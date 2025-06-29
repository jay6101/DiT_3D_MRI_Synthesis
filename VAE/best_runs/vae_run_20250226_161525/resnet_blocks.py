import torch
import torch.nn as nn
import torch.nn.functional as F

# --- ResNet Blocks ---

class ResBlock3D(nn.Module):
    """
    Standard 3D Residual Block with optional stride=2 for downsampling.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # 1x1 conv to match dimensions if needed
        self.shortcut = nn.Sequential()
        if (stride != 1) or (in_channels != out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.shortcut(identity)
        out += identity
        return F.relu(out, inplace=True)

class ResBlock3DTranspose(nn.Module):
    """
    3D Residual Block for upsampling using ConvTranspose3d.
    """
    def __init__(self, in_channels, out_channels, stride=2, bn=True):
        super().__init__()
        self.deconv1 = nn.ConvTranspose3d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.is_bn = bn
        self.bn1 = nn.BatchNorm3d(out_channels)

        self.conv2 = nn.Conv3d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # Shortcut for matching dimensions using the same transposed conv parameters
        if self.is_bn:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose3d(
                    in_channels, out_channels,
                    kernel_size=3, stride=stride, padding=1, bias=False
                ),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose3d(
                    in_channels, out_channels,
                    kernel_size=3, stride=stride, padding=1, bias=False
                )
            )


    def forward(self, x):
        identity = x
        out = self.deconv1(x)
        if self.is_bn:
            out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        if self.is_bn:
            out = self.bn2(out)

        identity = self.shortcut(identity)
        out += identity
        return F.relu(out, inplace=True)