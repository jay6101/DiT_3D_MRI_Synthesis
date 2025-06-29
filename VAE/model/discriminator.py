import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            #spectral_norm(nn.Conv3d(1, 32, 4, 2, 1)),  # [B, 1, 113,137,113] -> [B, 32, 57,69,57]
            nn.Conv3d(1, 32, 4, 2, 1), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(0.3),
            
            #spectral_norm(nn.Conv3d(32, 64, 4, 2, 1)),  # -> [B, 64, 29,35,29]
            nn.Conv3d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(0.3),
            
            #spectral_norm(nn.Conv3d(64, 128, 4, 2, 1)), # -> [B, 128,15,18,15]
            nn.Conv3d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(0.3),
            
            #spectral_norm(nn.Conv3d(128, 256, 4, 2, 1)),# -> [B, 256,7,9,7]
            nn.Conv3d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(0.3),
            
            #spectral_norm(nn.Conv3d(256, 512, 4, 2, 1)),# -> [B, 512,3,4,3]
            nn.Conv3d(256, 512, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(0.3),
            
            # spectral_norm(nn.Conv3d(512, 512, 4, 2, 1)),# -> [B, 512,1,2,1]
            nn.Conv3d(512, 512, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(0.3),
            
            nn.Flatten(),
            spectral_norm(nn.Linear(512 * 2, 1))
            # Optionally, add a Sigmoid activation for standard GANs or keep it linear for WGAN variants.
        )

    def forward(self, x):
        return self.main(x)
