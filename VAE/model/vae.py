import torch
import torch.nn as nn
from model.enc_dec import Encoder, Decoder

class VAE(nn.Module):
    def __init__(self, use_reparam=True):
        super().__init__()
        self.encoder = Encoder()  # Our new, larger encoder
        self.decoder = Decoder()  # Our new, larger decoder
        self.use_reparam = use_reparam
        
        if use_reparam:
            # Adjust these to match new latent channels (64 -> 64)
            self.mu_conv = nn.Conv3d(64, 64, kernel_size=1)
            self.logvar_conv = nn.Conv3d(64, 64, kernel_size=1)

    def forward(self, x):
        h = self.encoder(x)
        if self.use_reparam:
            mu = self.mu_conv(h)
            logvar = self.logvar_conv(h)
            z = reparameterize(mu, logvar)
            x_rec = self.decoder(z)
            return x_rec, mu, logvar
        else:
            return self.decoder(h), None, None
        
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
