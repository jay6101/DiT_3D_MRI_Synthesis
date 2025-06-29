import torch
import torch.nn as nn
from monai.networks.nets import SegResNet

class BrainTumor3DPerceptualLoss(nn.Module):
    def __init__(self, pretrained_path=None, target_layer_name="conv1",
                 spatial_dims=3, in_channels=1, out_channels=3, init_filters=32,
                 loss_weight=1.0):
        """
        Bundles a SegResNet model, extracts features from an early layer via a hook,
        and computes a 3D perceptual loss (L2 loss on extracted features) between prediction and target volumes.
        
        Parameters:
            pretrained_path (str): Path to the pretrained checkpoint (if any).
            target_layer_name (str): Name of the layer from which to extract features.
            spatial_dims (int): Dimensionality (e.g., 3 for 3D data).
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            init_filters (int): Initial number of filters in SegResNet.
            loss_weight (float): Scalar weight for the loss.
        """
        super(BrainTumor3DPerceptualLoss, self).__init__()
        self.loss_weight = loss_weight
        
        # Initialize the 3D SegResNet model
        self.seg_model = SegResNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=init_filters
        )
        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            self.seg_model.load_state_dict(checkpoint)
        self.seg_model.eval()  # set to eval mode
        
        # Set up the forward hook to capture features from the specified layer.
        self.feature = None
        self.target_layer_name = target_layer_name
        self._register_hook()
        
        # Use MSELoss as the perceptual loss.
        self.l2_loss = nn.MSELoss()
    
    def _hook_fn(self, module, input, output):
        self.feature = output
        
    def _register_hook(self):
        # Register hook on the target layer.
        for name, module in self.seg_model.named_modules():
            if name == self.target_layer_name:
                module.register_forward_hook(self._hook_fn)
                break
    
    def forward(self, x, target):
        """
        x and target: input 3D volumes (N, C, D, H, W) with values in the same range.
        Computes the loss on the features extracted at the specified layer.
        """
        # Extract features for the prediction.
        self.feature = None
        _ = self.seg_model(x)
        features_pred = self.feature

        # Extract features for the target.
        self.feature = None
        _ = self.seg_model(target)
        features_target = self.feature

        # Compute L2 loss between the features.
        loss = self.l2_loss(features_pred, features_target)
        return self.loss_weight * loss

# Example usage:
if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # If you have a pretrained checkpoint, provide its path.
    pretrained_model_path = "/space/mcdonald-syn01/1/projects/jsawant/DSC250/VAE_GAN/model/model.pt"  # e.g., "path/to/checkpoint.pth"
    model = BrainTumor3DPerceptualLoss(
        pretrained_path=None,
        target_layer_name="conv1",  # change as needed
        spatial_dims=3, in_channels=1, out_channels=3, init_filters=32
    ).to(device)
    
    # Create dummy input and target volumes
    dummy_input = torch.rand(1, 1, 240, 240, 160).to(device)
    dummy_target = torch.rand(1, 1, 240, 240, 160).to(device)
    
    # Calculate and print the final loss.
    final_loss = model(dummy_input, dummy_target)
    print("Final 3D Perceptual Loss:", final_loss.item())
