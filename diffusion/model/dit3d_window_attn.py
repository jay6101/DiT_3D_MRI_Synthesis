#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Modified 3D DiT with Window-based Attention
- No voxelization
- Accepts (B, C, D, H, W) latent volumes
- Splits them via 3D patch embedding
- Uses window partitioning for memory-efficient 3D attention
- Includes optional relative position embeddings
References:
 - GLIDE: https://github.com/openai/glide-text2im
 - MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import Mlp

#################################################################################
#                              Utility Functions                                #
#################################################################################

def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (Tensor): shape [B, D, H, W, C].
        window_size (int): window size for each dimension.

    Returns:
        windows (Tensor): shape [B * num_windows, window_size, window_size, window_size, C].
        pad (tuple): (Dp, Hp, Wp) padded sizes so we can reverse it later.
    """
    B, D, H, W, C = x.shape

    pad_d = (window_size - D % window_size) % window_size
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))

    Dp, Hp, Wp = D + pad_d, H + pad_h, W + pad_w
    # Reshape into (B, Dp/window, window, Hp/window, window, Wp/window, window, C)
    x = x.view(
        B,
        Dp // window_size, window_size,
        Hp // window_size, window_size,
        Wp // window_size, window_size,
        C
    )
    # Move window dims next to each other and combine them as a single dimension
    # final shape: [B*(Dp//window * Hp//window * Wp//window), window_size, window_size, window_size, C]
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, window_size, window_size, window_size, C)
    return windows, (Dp, Hp, Wp)

def window_unpartition(windows, window_size, pad_xyz, orig_xyz):
    """
    Reverse of window_partition.
    Args:
        windows (Tensor): shape [B * num_windows, window_size, window_size, window_size, C].
        window_size (int): window size (same used in partition).
        pad_xyz (tuple): (Dp, Hp, Wp) size after padding.
        orig_xyz (tuple): (D, H, W) original size before padding.

    Returns:
        x (Tensor): shape [B, D, H, W, C].
    """
    Dp, Hp, Wp = pad_xyz
    D, H, W = orig_xyz
    B = windows.shape[0] // ((Dp // window_size) * (Hp // window_size) * (Wp // window_size))

    # Inverse reshape
    x = windows.view(
        B,
        Dp // window_size, Hp // window_size, Wp // window_size,
        window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(B, Dp, Hp, Wp, -1)

    # Remove any padding
    if (Dp > D) or (Hp > H) or (Wp > W):
        x = x[:, :D, :H, :W, :].contiguous()
    return x

def get_rel_pos(q_size, k_size, rel_pos):
    """
    Interpolate relative position embeddings if needed and
    return the slice corresponding to the relative positions.

    Args:
        q_size (int): query dimension size.
        k_size (int): key dimension size.
        rel_pos (Tensor): shape [2 * M - 1, C], where M = max(q_size, k_size).
    """
    max_rel_dist = 2 * max(q_size, k_size) - 1

    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate the rel_pos
        rel_pos_resized = F.interpolate(
            rel_pos[None].permute(0, 2, 1),  # (1, C, L)
            size=max_rel_dist,
            mode='linear',
            align_corners=False
        )
        rel_pos_resized = rel_pos_resized.permute(0, 2, 1)[0]  # (L, C)
    else:
        rel_pos_resized = rel_pos

    # coords in [0, M-1], shift by (M-1) for indexing
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]

def add_decomposed_rel_pos(attn, q, rel_pos_x, rel_pos_y, rel_pos_z, q_size, k_size):
    """
    Decomposed 3D relative position embeddings
    from MViTv2 approach extended to 3D.

    attn: [B*num_heads, qD*qH*qW, kD*kH*kW]
    q:    [B*num_heads, qD*qH*qW, dim_per_head]
    rel_pos_x: [2 * X - 1, dim_per_head]  (X = max(qD, kD))
    rel_pos_y, rel_pos_z similarly.
    q_size, k_size: (qD, qH, qW), (kD, kH, kW)
    """
    qD, qH, qW = q_size
    kD, kH, kW = k_size
    Bhw, _, dimh = q.shape

    # Reshape q to B,num_heads,qD,qH,qW,dimh
    # but we combined B & num_heads dimension => r_q is shape (Bhw, qD, qH, qW, dimh)
    r_q = q.reshape(Bhw, qD, qH, qW, dimh)

    Rx = get_rel_pos(qD, kD, rel_pos_x)  # shape [qD,kD, dim_per_head]
    Ry = get_rel_pos(qH, kH, rel_pos_y)  # shape [qH,kH, dim_per_head]
    Rz = get_rel_pos(qW, kW, rel_pos_z)  # shape [qW,kW, dim_per_head]

    # Combine the terms
    # each is shape [Bhw, qD, qH, qW, kD or kH or kW]
    rel_x = torch.einsum('bdhwc,dkc->bdhwk', r_q, Rx)  # [Bhw, qD, qH, qW, kD]
    rel_y = torch.einsum('bdhwc,hkc->bdhwk', r_q, Ry)
    rel_z = torch.einsum('bdhwc,wkc->bdhwk', r_q, Rz)

    # Add these onto attn
    # attn is [Bhw, qD*qH*qW, kD*kH*kW]
    attn_reshape = attn.view(Bhw, qD, qH, qW, kD, kH, kW)

    # Expand rel_x to shape [Bhw, qD, qH, qW, kD, 1, 1], etc.
    attn_reshape = attn_reshape + rel_x[:, :, :, :, :, None, None]
    attn_reshape = attn_reshape + rel_y[:, :, :, :, None, :, None]
    attn_reshape = attn_reshape + rel_z[:, :, :, :, None, None, :]

    attn = attn_reshape.view(Bhw, qD*qH*qW, kD*kH*kW)
    return attn

def modulate(x, shift, scale):
    """
    Adaptive layer norm shift/scale
    x: (B, N, dim)
    shift, scale: (B, dim)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: (B,) or (B,1)
        :param dim: dimension of the output
        :param max_period: controls the minimum frequency
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout (classifier-free guidance).
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        # if dropout_prob>0, we add an extra "dummy" class for the dropped label
        self.embedding_table = nn.Embedding(num_classes + 1 if dropout_prob>0 else num_classes, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        # replace dropped labels with the "num_classes" index
        return torch.where(drop_ids, torch.tensor(self.num_classes, device=labels.device), labels)

    def forward(self, labels, train, force_drop_ids=None):
        if (train and self.dropout_prob>0) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)

#################################################################################
#                                  3D Patch Embed                               #
#################################################################################

class PatchEmbed3D(nn.Module):
    """
    3D volume -> patch embeddings
    """
    def __init__(self, 
                 in_chans=4, 
                 in_size=(28,34,28),
                 patch_size=(2,2,2),
                 embed_dim=768):
        super().__init__()
        if isinstance(in_size, int):
            in_size = (in_size, in_size, in_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)

        # patch grid dims
        gd = in_size[0] // patch_size[0]
        gh = in_size[1] // patch_size[1]
        gw = in_size[2] // patch_size[2]
        self.num_patches = gd * gh * gw
        self.patch_size = patch_size
        self.in_size = in_size

        # simple 3D conv to embed
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True
        )

    def forward(self, x):
        # x: (B, C, D, H, W)
        x = x.float()
        x = self.proj(x)  # shape (B, embed_dim, D/pD, H/pH, W/pW)
        x = x.flatten(2)  # (B, embed_dim, N)
        x = x.transpose(1, 2)  # (B, N, embed_dim)
        return x

#################################################################################
#                             Positional Embeddings                              #
#################################################################################

def get_3d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: (d, h, w)
    Return: (d*h*w, embed_dim)
    """
    d, h, w = grid_size
    grid_d = np.arange(d, dtype=np.float32)
    grid_h = np.arange(h, dtype=np.float32)
    grid_w = np.arange(w, dtype=np.float32)
    grid = np.meshgrid(grid_d, grid_h, grid_w, indexing='ij')  # [3, d, h, w]
    grid = np.stack(grid, axis=0).reshape(3, -1)               # [3, d*h*w]

    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed  # shape (d*h*w, embed_dim)

def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    # grid: [3, d*h*w]
    assert embed_dim % 3 == 0
    emb_d = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])
    return np.concatenate([emb_d, emb_h, emb_w], axis=1)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    pos: shape (M,)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= (embed_dim / 2.)
    omega = 1. / (10000 ** omega)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, embed_dim/2)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)

#################################################################################
#                  Window-based Multi-head 3D Attention                         #
#################################################################################

class Attention(nn.Module):
    """
    Multi-head attention with optional window partitioning and 3D relative position.
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,   # (d, h, w) of patch-grid
        window_size=0      # if >0, we do local window partition
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        self.input_size = input_size  # (d//patchD, h//patchH, w//patchW)
        self.window_size = window_size

        if self.use_rel_pos:
            d, h, w = input_size
            # For local windows, you might only need (window_size) for each axis,
            # or you can keep it as (d,h,w) for global. In practice, we often do:
            #   max_d = window_size if window_size>0 else d
            #   self.rel_pos_x = ...
            # but for simplicity, we'll store them as if global. 
            # Adjust as needed for local if you want smaller embeddings.
            max_d = window_size if window_size>0 else d
            max_h = window_size if window_size>0 else h
            max_w = window_size if window_size>0 else w

            self.rel_pos_x = nn.Parameter(torch.zeros(2 * max_d - 1, head_dim))
            self.rel_pos_y = nn.Parameter(torch.zeros(2 * max_h - 1, head_dim))
            self.rel_pos_z = nn.Parameter(torch.zeros(2 * max_w - 1, head_dim))

            if not rel_pos_zero_init:
                nn.init.trunc_normal_(self.rel_pos_x, std=0.02)
                nn.init.trunc_normal_(self.rel_pos_y, std=0.02)
                nn.init.trunc_normal_(self.rel_pos_z, std=0.02)

    def forward(self, x):
        # x shape: (B, N, dim)
        B, N, C = x.shape
        # Reshape to (B, D, H, W, C)
        DHW = self.input_size
        d, h, w = DHW
        x_3d = x.reshape(B, d, h, w, C)

        # If using local windows, partition
        if self.window_size > 0:
            # B * nWindows, window_size, window_size, window_size, C
            windows, pad_xyz = window_partition(x_3d, self.window_size)
            b_, ws, _, _, c_ = windows.shape
            # Flatten to (b_, ws^3, c_)
            x_flat = windows.reshape(b_, ws*ws*ws, c_)
        else:
            # global attention
            windows = x_3d
            b_ = B
            ws = None
            x_flat = windows.view(B, d*h*w, C)
            pad_xyz = None

        # QKV
        qkv = self.qkv(x_flat)  # (b_, n, 3*dim)
        qkv = qkv.reshape(b_, -1, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, b_, num_heads, n, head_dim)
        q, k, v = qkv.unbind(dim=0)

        # scaled dot-product
        attn = (q * self.scale) @ k.transpose(-2, -1)  # shape (b_, num_heads, n, n)

        # Optionally add 3D relative pos
        if self.use_rel_pos:
            # Flatten q to shape (b_ * num_heads, n, head_dim)
            bh = b_ * self.num_heads
            # We want to figure out the 3D shape for n => could be (ws, ws, ws) or (d, h, w)
            if self.window_size > 0:
                dd = hh = ww = self.window_size
                kd = kh = kw = self.window_size
            else:
                dd, hh, ww = d, h, w
                kd, kh, kw = d, h, w

            q_reshape = q.reshape(bh, dd*hh*ww, -1)  # (bh, n, head_dim)
            attn = add_decomposed_rel_pos(
                attn.view(bh, dd*hh*ww, dd*hh*ww),
                q_reshape,
                self.rel_pos_x, self.rel_pos_y, self.rel_pos_z,
                (dd, hh, ww), (kd, kh, kw)
            ).view(b_, self.num_heads, dd*hh*ww, dd*hh*ww)

        attn = attn.softmax(dim=-1)
        out = attn @ v  # shape (b_, num_heads, n, head_dim)
        out = out.transpose(1, 2).reshape(b_, -1, C)
        out = self.proj(out)

        # Unpartition
        if self.window_size > 0:
            # out shape (b_, n, C), we map it back to (b_, ws,ws,ws, C)
            out_3d = out.reshape(b_, ws, ws, ws, C)
            x_3d = window_unpartition(out_3d, self.window_size, pad_xyz, (d,h,w))
            x = x_3d.reshape(B, d*h*w, C)
        else:
            # global
            x = out

        return x

#################################################################################
#                                   DiT Blocks                                  #
#################################################################################

class DiTBlock(nn.Module):
    """
    DiT block with adaptive LayerNorm zero conditioning,
    and optional window-based attention.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        input_size=None
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size,  # (d//pD, h//pH, w//pW)
            window_size=window_size
        )
        self.input_size = input_size
        self.window_size = window_size

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

        # 6 * hidden_size => shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # x: (B, N, hidden_size)
        # c: (B, hidden_size)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        # MSA
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        x_attn = self.attn(x_norm)
        x = x + gate_msa.unsqueeze(1)*x_attn

        # MLP
        x_norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x_mlp = self.mlp(x_norm2)
        x = x + gate_mlp.unsqueeze(1)*x_mlp

        return x

class FinalLayer(nn.Module):
    """
    Final projection from hidden vectors -> 3D patchified output
    """
    def __init__(self, hidden_size, patch_size=(2,2,2), out_channels=4):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size,
            patch_size[0]*patch_size[1]*patch_size[2]*out_channels,
            bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2*hidden_size, bias=True)
        )
        self.patch_size = patch_size
        self.out_channels = out_channels

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)  # (B, N, pD*pH*pW*C_out)
        return x

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiT(nn.Module):
    """
    3D Diffusion Model with a Transformer backbone and window-based attention.
    """
    def __init__(
        self,
        in_size=(28,34,28),
        patch_size=(2,2,2),
        in_channels=4,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1,
        learn_sigma=False,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        window_block_indexes=()
    ):
        """
        window_block_indexes: list of layer indices that should use local window-attn.
                             e.g. [0,1,2,3] or all blocks if you want all to have local windows.
        """
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels*2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.in_size = in_size

        # Patch Embedding
        self.patch_embed = PatchEmbed3D(
            in_chans=in_channels,
            in_size=in_size,
            patch_size=patch_size,
            embed_dim=hidden_size
        )
        num_patches = self.patch_embed.num_patches

        # Timestep & Label embeddings
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        # Positional Embedding
        # shape => (D//pD, H//pH, W//pW)
        d_ = in_size[0] // patch_size[0]
        h_ = in_size[1] // patch_size[1]
        w_ = in_size[2] // patch_size[2]

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        # Build DiT blocks
        blocks = []
        for i in range(depth):
            wsize = window_size if i in window_block_indexes else 0
            block = DiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=wsize,
                input_size=(d_, h_, w_)
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights((d_, h_, w_))

    def initialize_weights(self, patch_grid):
        # Basic init
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_basic_init)

        # Sine-cos pos embedding
        embed_dim = self.pos_embed.shape[-1]
        pos_embed = get_3d_sincos_pos_embed(embed_dim, patch_grid)  # shape (d*h*w, embed_dim)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # PatchEmbed conv init
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Zero-out AdaLN mods
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        Inverse of PatchEmbed:
        x: (B, N, pD*pH*pW*C_out)
        returns volume: (B, C_out, D, H, W)
        """
        B, N, outdim = x.shape
        pD, pH, pW = self.patch_size
        C_out = self.out_channels

        # Patch grid size
        gD = self.in_size[0] // pD
        gH = self.in_size[1] // pH
        gW = self.in_size[2] // pW
        assert gD*gH*gW == N, "Mismatch in # of tokens vs. patch grid"

        x = x.reshape(B, gD, gH, gW, pD, pH, pW, C_out)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)  # B, C_out, gD,pD, gH,pH, gW,pW
        D, H, W = gD*pD, gH*pH, gW*pW
        x = x.reshape(B, C_out, D, H, W)
        return x

    def forward(self, x, t, y):
        """
        x: (B, C, D, H, W)  your 3D latent
        t: (B,) diffusion timestep
        y: (B,) class label
        """
        # Patchify
        x = self.patch_embed(x)  # (B, N, hidden_size)
        x = x + self.pos_embed   # add sin-cos positional embedding

        t_emb = self.t_embedder(t)                # (B, hidden_size)
        y_emb = self.y_embedder(y, self.training) # (B, hidden_size)
        c = t_emb + y_emb

        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)  # (B, N, pD*pH*pW*C_out)

        # Unpatchify
        x = self.unpatchify(x)     # (B, C_out, D, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Classifier-free guidance: feed [uncond + cond] in one batch of double-size.
        """
        half = x[:len(x)//2]
        combined = torch.cat([half, half], dim=0)
        out = self.forward(combined, t, y)
        # Suppose we only apply guidance to first 3 channels (like some repos do).
        # Or you can do all channels. Below, only 3 channels are guided:
        eps, rest = out[:, :3], out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps)//2, dim=0)
        half_eps = uncond_eps + cfg_scale*(cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

#################################################################################
#                      Example Model Constructors (Optional)                    #
#################################################################################

def DiT_XL_2(**kwargs):
    # Example config with patch_size=2, big hidden_size=1152
    return DiT(depth=28, hidden_size=1152, patch_size=(2,2,2), num_heads=16, **kwargs)

def DiT_B_4(**kwargs):
    # Example with patch_size=4, hidden_size=768
    return DiT(depth=12, hidden_size=768, patch_size=(4,4,4), num_heads=12, **kwargs)

def DiT_S_4(**kwargs):
    # Example with patch_size=4, hidden_size=768
    return DiT(depth=12, hidden_size=384, patch_size=(4,2,4), num_heads=6, **kwargs)

# etc. You can add more to a dictionary if you like.
DiT3D_models_WindAttn = {
    'DiT-XL/2': DiT_XL_2,
    'DiT-B/4':  DiT_B_4,
    'DiT-S/4':  DiT_S_4,
    # ...
}
