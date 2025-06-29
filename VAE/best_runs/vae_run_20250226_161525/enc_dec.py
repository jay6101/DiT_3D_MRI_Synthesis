# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from model.resnet_blocks import ResBlock3D, ResBlock3DTranspose

# class Encoder(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         # ----------- Stage 1 -----------
#         # Downsample from 1 -> 64 channels, stride=2
#         self.block1_down = ResBlock3D(in_channels=1, out_channels=64, stride=2)
#         # Extra residual block, keeps channels=64, stride=1
#         self.block1_res = ResBlock3D(in_channels=64, out_channels=64, stride=1)
        
#         # ----------- Stage 2 -----------
#         # Downsample 64 -> 128 channels, stride=2
#         self.block2_down = ResBlock3D(in_channels=64, out_channels=128, stride=2)
#         # Extra residual block, keeps channels=128, stride=1
#         self.block2_res = ResBlock3D(in_channels=128, out_channels=128, stride=1)
        
#         self.block3_down = ResBlock3D(in_channels=128, out_channels=256, stride=2)
#         # Extra residual block, keeps channels=128, stride=1
#         self.block3_res = ResBlock3D(in_channels=256, out_channels=256, stride=1)
        
#         # Convert 128 -> 64 at the bottleneck
#         self.conv_reduce = nn.Conv3d(256, 64, kernel_size=1, stride=1)
#         self.bn_reduce = nn.BatchNorm3d(64)

#     def forward(self, x):
#         # x: [B, 1, 113, 137, 113]
        
#         # ---- Stage 1 ----
#         x = self.block1_down(x)  # => [B, 64, 57, 69, 57]
#         x = self.block1_res(x)   # => [B, 64, 57, 69, 57]
        
#         # ---- Stage 2 ----
#         x = self.block2_down(x)  # => [B, 128, 29, 35, 29]
#         x = self.block2_res(x)   # => [B, 128, 29, 35, 29]
        
#         # ---- Stage 3 ----
#         x = self.block3_down(x)  # => [B, 128, 29, 35, 29]
#         x = self.block3_res(x)   # => [B, 128, 29, 35, 29]
        
#         # Reduce to 64 channels
#         x = self.conv_reduce(x)  # => [B, 64, 29, 35, 29]
#         x = self.bn_reduce(x)
        
#         return x

# class Decoder(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         # Expand 64 -> 128 before upsampling
#         self.conv_expand = nn.Conv3d(64, 256, kernel_size=1, stride=1)
#         self.bn_expand = nn.BatchNorm3d(256)
        
#         # ----------- Stage 1 -----------
#         # Upsample 128 -> 64 channels, stride=2
#         self.block1_up = ResBlock3DTranspose(in_channels=256, out_channels=128, stride=2)
#         # Extra residual block, keeps channels=64, stride=1
#         self.block1_res = ResBlock3DTranspose(in_channels=128, out_channels=128, stride=1)
        
#        # ----------- Stage 2 -----------
#         # Upsample 128 -> 64 channels, stride=2
#         self.block2_up = ResBlock3DTranspose(in_channels=128, out_channels=64, stride=2)
#         # Extra residual block, keeps channels=64, stride=1
#         self.block2_res = ResBlock3DTranspose(in_channels=64, out_channels=64, stride=1)
        
#         # ----------- Stage 2 -----------
#         # Upsample 64 -> 1 channel, stride=2
#         self.block3_up = ResBlock3DTranspose(in_channels=64, out_channels=1, stride=2)

#     def forward(self, z):
#         # z: [B, 64, 29, 35, 29]
        
#         x = self.conv_expand(z)  # => [B, 128, 29, 35, 29]
#         x = self.bn_expand(x)
#         x = F.relu(x, inplace=True)
        
#         # ---- Stage 1 ----
#         x = self.block1_up(x)    # => [B, 64, 57, 69, 57]
#         x = self.block1_res(x)   # => [B, 64, 57, 69, 57]
        
#         # ---- Stage 2 ----
#         x = self.block2_up(x)    # => [B, 64, 57, 69, 57]
#         x = self.block2_res(x)   # => [B, 64, 57, 69, 57]
        
#         # ---- Stage 2 ----
#         x = self.block3_up(x)    # => [B, 1, 113, 137, 113]
        
#         return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet_blocks import ResBlock3D, ResBlock3DTranspose

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # ----------- Stage 1 -----------
        # Downsample from 1 -> 64 channels, stride=2
        self.block1_down = ResBlock3D(in_channels=1, out_channels=64, stride=2)
        # Extra residual block, keeps channels=64, stride=1
        self.block1_res = ResBlock3D(in_channels=64, out_channels=64, stride=1)
        
        # ----------- Stage 2 -----------
        # Downsample 64 -> 128 channels, stride=2
        self.block2_down = ResBlock3D(in_channels=64, out_channels=128, stride=2)
        # Extra residual block, keeps channels=128, stride=1
        self.block2_res = ResBlock3D(in_channels=128, out_channels=128, stride=1)
        
        # Convert 128 -> 64 at the bottleneck
        self.conv_reduce = nn.Conv3d(128, 64, kernel_size=1, stride=1)
        self.bn_reduce = nn.BatchNorm3d(64)

    def forward(self, x):
        # x: [B, 1, 113, 137, 113]
        
        # ---- Stage 1 ----
        x = self.block1_down(x)  # => [B, 64, 57, 69, 57]
        x = self.block1_res(x)   # => [B, 64, 57, 69, 57]
        
        # ---- Stage 2 ----
        x = self.block2_down(x)  # => [B, 128, 29, 35, 29]
        x = self.block2_res(x)   # => [B, 128, 29, 35, 29]
        
        # Reduce to 64 channels
        x = self.conv_reduce(x)  # => [B, 64, 29, 35, 29]
        x = self.bn_reduce(x)
        
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Expand 64 -> 128 before upsampling
        self.conv_expand = nn.Conv3d(64, 128, kernel_size=1, stride=1)
        self.bn_expand = nn.BatchNorm3d(128)
        
        # ----------- Stage 1 -----------
        # Upsample 128 -> 64 channels, stride=2
        self.block1_up = ResBlock3DTranspose(in_channels=128, out_channels=64, stride=2)
        # Extra residual block, keeps channels=64, stride=1
        self.block1_res = ResBlock3DTranspose(in_channels=64, out_channels=64, stride=1)
        
        # ----------- Stage 2 -----------
        # Upsample 64 -> 1 channel, stride=2
        self.block2_up = ResBlock3DTranspose(in_channels=64, out_channels=1, stride=2, bn=True)

    def forward(self, z):
        # z: [B, 64, 29, 35, 29]
        
        x = self.conv_expand(z)  # => [B, 128, 29, 35, 29]
        x = self.bn_expand(x)
        x = F.relu(x, inplace=True)
        
        # ---- Stage 1 ----
        x = self.block1_up(x)    # => [B, 64, 57, 69, 57]
        x = self.block1_res(x)   # => [B, 64, 57, 69, 57]
        
        # ---- Stage 2 ----
        x = self.block2_up(x)    # => [B, 1, 113, 137, 113]
        x = torch.tanh(x)
        
        return x

