import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import Attention, Mlp

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: tuple of three ints (D, H, W)
    return:
    pos_embed: [D*H*W, embed_dim]
    """
    # grid_size is now a tuple (D,H,W)
    d, h, w = grid_size

    # Ranges for each dimension
    grid_d = np.arange(d, dtype=np.float32)
    grid_h = np.arange(h, dtype=np.float32)
    grid_w = np.arange(w, dtype=np.float32)

    # Create (3,D,H,W) 
    grid = np.meshgrid(grid_d, grid_h, grid_w, indexing='ij')  # shape [3, D, H, W]
    grid = np.stack(grid, axis=0)                              # [3, D, H, W]
    grid = grid.reshape([3, -1])                               # [3, D*H*W]

    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    # grid: [3, D*H*W]
    assert embed_dim % 3 == 0
    emb_d = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (D*H*W, D/3)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (D*H*W, D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (D*H*W, D/3)
    emb = np.concatenate([emb_d, emb_h, emb_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each axis
    pos: a list of positions to be encoded: size (M,)  e.g. D*H*W in flatten
    out: (M, embed_dim)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= (embed_dim / 2.)
    omega = 1. / (10000**omega)  # (embed_dim/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # [M, embed_dim/2]

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

# ------------------------------------------------------------------------------
# Timestep + Label Embedding
# ------------------------------------------------------------------------------
class TimestepEmbedder(nn.Module):
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
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also does label dropout (classifier-free guidance).
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        self.embedding_table = nn.Embedding(
            num_classes + 1 if dropout_prob > 0 else num_classes,
            hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids, torch.tensor(self.num_classes, device=labels.device), labels)

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)

# ------------------------------------------------------------------------------
# 3D Patch Embedding (no voxelization)
# ------------------------------------------------------------------------------
class PatchEmbed3D(nn.Module):
    """
    Simple 3D convolutional patch-embedding that splits (C,D,H,W) into smaller patches,
    flattens them, then produces a sequence of shape (N, num_patches, embed_dim).
    """
    def __init__(self, in_chans=4,  # for your latent channels
                 in_size=(28,34,28),  # (D,H,W)
                 patch_size=(2,2,2),
                 embed_dim=768):
        super().__init__()
        # Make sure in_size and patch_size are each length 3
        if isinstance(in_size, int):
            in_size = (in_size, in_size, in_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)

        # Number of patches in each dimension
        self.grid_d = in_size[0] // patch_size[0]
        self.grid_h = in_size[1] // patch_size[1]
        self.grid_w = in_size[2] // patch_size[2]
        self.num_patches = self.grid_d * self.grid_h * self.grid_w

        self.patch_size = patch_size
        # A simple 3D conv that reduces the D,H,W by patch factors
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True
        )

    def forward(self, x):
        # x: shape (B, C, D, H, W)
        x = x.float()
        x = self.proj(x)               # (B, embed_dim, D/pD, H/pH, W/pW)
        x = x.flatten(2)               # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)          # (B, num_patches, embed_dim)
        return x


# ------------------------------------------------------------------------------
# Transformer Blocks
# ------------------------------------------------------------------------------
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.GELU,  # or approx
            drop=0.,
            eta=None
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        """
        x: (B, N, hidden_size)
        c: (B, hidden_size)  [the combined t & class embeddings]
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        # Attention
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        x_attn = self.attn(x_norm)
        x = x + gate_msa.unsqueeze(1) * x_attn

        # MLP
        x_norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x_mlp = self.mlp(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * x_mlp
        return x


class FinalLayer(nn.Module):
    """
    The final projection from hidden size back to (patch_size^3 * out_channels) per token.
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
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.patch_size = patch_size
        self.out_channels = out_channels

    def forward(self, x, c):
        # x: (B, N, hidden_size)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x  # shape (B, N, patch_size^3 * out_channels)


# ------------------------------------------------------------------------------
# DiT for 3D volumes (no voxelization)
# ------------------------------------------------------------------------------
class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone for 3D data.
    """
    def __init__(
        self,
        in_size=(28,34,28),   # (D,H,W)
        in_channels=4,        # for your latent
        patch_size=(2,2,2),
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1,
        learn_sigma=False
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels

        if isinstance(in_size, int):
            in_size = (in_size, in_size, in_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)

        # This is the flattened patch dimension (D/pD * H/pH * W/pW).
        self.patch_embed = PatchEmbed3D(
            in_chans=in_channels,
            in_size=in_size,
            patch_size=patch_size,
            embed_dim=hidden_size
        )
        num_patches = self.patch_embed.num_patches

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        # -- Sine-cos pos embedding
        pos_embed_shape = (in_size[0]//patch_size[0], 
                           in_size[1]//patch_size[1],
                           in_size[2]//patch_size[2])
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size),
            requires_grad=False
        )

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(
            hidden_size,
            patch_size=patch_size,
            out_channels=self.out_channels
        )

        # Store shapes, to unpatchify
        self.in_size = in_size
        self.patch_size = patch_size
        self.initialize_weights(pos_embed_shape)

    def initialize_weights(self, pos_embed_shape):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        embed_dim = self.pos_embed.shape[-1]
        pos_embed_np = get_3d_sincos_pos_embed(embed_dim, pos_embed_shape)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed_np).float().unsqueeze(0)
        )

        # Initialize patch_embed conv like nn.Linear
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Zero out final layers in blocks
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(
            self.final_layer.adaLN_modulation[-1].weight, 0
        )
        nn.init.constant_(
            self.final_layer.adaLN_modulation[-1].bias, 0
        )
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        Reverse the PatchEmbed step.
        x: (B, num_patches, pD*pH*pW*C_out)
        Return shape: (B, C_out, D, H, W)
        """
        B, N, out_tokens = x.shape
        pD, pH, pW = self.patch_size
        C_out = self.out_channels
        # Dimensions of the patch grid
        gD = self.in_size[0] // pD
        gH = self.in_size[1] // pH
        gW = self.in_size[2] // pW
        assert gD*gH*gW == N, "Mismatch in number of patches"

        x = x.reshape(B, gD, gH, gW, pD, pH, pW, C_out)
        # rearrange the block back to 3D
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)  # B,C, gD,pD, gH,pH, gW,pW
        D = gD * pD
        H = gH * pH
        W = gW * pW
        x = x.reshape(B, C_out, D, H, W)
        return x

    def forward(self, x, t, y):
        """
        x: shape (B, C, D, H, W)  — your 3D latent
        t: (B,) diffusion timesteps
        y: (B,) class labels
        """
        # Patchify
        x = self.patch_embed(x)  # (B, N, hidden_size)
        x = x + self.pos_embed   # (B, N, hidden_size)

        # Combine time + label embeddings
        t_emb = self.t_embedder(t)                # (B, hidden_size)
        y_emb = self.y_embedder(y, self.training) # (B, hidden_size)
        c = t_emb + y_emb

        # Transformer
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)  # (B, N, pD*pH*pW*C_out)

        # Unpatchify
        x = self.unpatchify(x)      # (B, C_out, D, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Classifier-free guidance sampling: feed (cond + uncond) in a single batch.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)  # uncond+cond
        model_out = self.forward(combined, t, y)
        # By default, we apply guidance only to first 3 channels, or all channels if you prefer:
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps)//2, dim=0)
        half_eps = uncond_eps + cfg_scale*(cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
