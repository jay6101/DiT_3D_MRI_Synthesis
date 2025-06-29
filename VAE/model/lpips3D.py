import torch
import torch.nn as nn
import lpips

class LPIPSLoss3D(nn.Module):
    def __init__(self, net='vgg'):
        """
        Initializes the LPIPSLoss3D module.
        
        Args:
            net: Specifies the backbone network to use ('vgg' in this case).
        """
        super(LPIPSLoss3D, self).__init__()
        self.lpips_fn = lpips.LPIPS(net=net)

    def forward(self, volume1, volume2):
        B, C, D, H, W = volume1.shape
        total_loss = 0.0
        count = 0

        # Axial view (D-axis)
        for d in range(D):
            slice1 = volume1[:, :, d, :, :]
            slice2 = volume2[:, :, d, :, :]
            loss = self.lpips_fn(slice1, slice2)
            total_loss += loss
            count += 1

        # Sagittal view (W-axis)
        for w in range(W):
            slice1 = volume1[:, :, :, :, w]
            slice2 = volume2[:, :, :, :, w]
            loss = self.lpips_fn(slice1, slice2)
            total_loss += loss
            count += 1

        # Coronal view (H-axis)
        for h in range(H):
            slice1 = volume1[:, :, :, h, :]
            slice2 = volume2[:, :, :, h, :]
            loss = self.lpips_fn(slice1, slice2)
            total_loss += loss
            count += 1

        aggregated_loss = total_loss / count
        return aggregated_loss


# Example usage:
# model = LPIPSLoss3D(net='vgg')
# loss = model(volume1, volume2)
