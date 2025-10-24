import os
import torch
import json
import numpy as np
from model.maisi_vae import VAE_Lite
from tqdm import tqdm

def load_vae_model(run_folder, device):
    """Load the trained VAE model."""
    # Load hyperparameters
    with open(os.path.join(run_folder, 'hparams.json'), 'r') as f:
        hparams = json.load(f)
    hparams['device'] = device
    
    # Initialize VAE model with the same parameters as training
    vae = VAE_Lite(
    spatial_dims=3,           # 3D model
    in_channels=1,            # e.g., single-channel input
    out_channels=1,           # single-channel reconstruction
    channels=(32, 64, 128),   # downsampling channels
    num_res_blocks=(1, 1, 1),    # one ResBlock per level
    attention_levels=(False, False, False),
    latent_channels=4,
    norm_num_groups=16,
    norm_eps=1e-5,
    with_encoder_nonlocal_attn=False,
    with_decoder_nonlocal_attn=False,
    include_fc=False,
    use_combined_linear=False,
    use_flash_attention=False,
    use_convtranspose=False,
    num_splits=8,
    dim_split=0,
    norm_float16=False,
    print_info=False,
    save_mem=True,
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(os.path.join(run_folder, 'best_vae.pth'), 
                          map_location=device)
    
    vae.load_state_dict(checkpoint['vae_state_dict'])
    vae.eval()
    
    return vae, hparams

def determine_latent_dimensions(vae, device):
    """Determine the latent space dimensions by doing a forward pass with dummy input."""
    # Create dummy input with same dimensions as training data: [1, 112, 136, 112]
    dummy_input = torch.randn(1, 1, 112, 136, 112).to(device)
    
    with torch.no_grad():
        mean, logvar = vae.encode(dummy_input)
    
    return mean.shape[1:]  # Remove batch dimension

def generate_synthetic_samples(vae, latent_shape, num_samples, device, output_dir):
    """Generate synthetic samples from Gaussian latents using the VAE decoder."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_samples} synthetic samples...")
    print(f"Latent shape: {latent_shape}")
    
    # Generate samples in batches to avoid memory issues
    batch_size = 8  # Adjust based on available GPU memory
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    sample_idx = 0
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
            # Determine current batch size
            current_batch_size = min(batch_size, num_samples - sample_idx)
            
            # Sample from standard Gaussian distribution
            latent_samples = torch.randn(current_batch_size, *latent_shape).to(device)
            
            # Generate samples using decoder
            generated_samples = vae.decode(latent_samples)
            
            # Save each sample
            for i in range(current_batch_size):
                sample = generated_samples[i].cpu().numpy()
                sample_filename = f"synthetic_sample_{sample_idx:04d}.npy"
                np.save(os.path.join(output_dir, sample_filename), sample)
                sample_idx += 1
            
            # Clear GPU memory
            del latent_samples, generated_samples
            torch.cuda.empty_cache() if device.startswith('cuda') else None
    
    print(f"Generated {num_samples} synthetic samples and saved to {output_dir}")

def main():
    # Configuration
    run_folder = "/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/VAE/best_runs/vae_run_20250226_161525"
    output_dir = "/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/synthetic_data_VAE"
    device = 'cuda:1'  # Adjust as needed
    num_samples = 481
    
    print(f"Loading VAE model from {run_folder}")
    
    # Load the VAE model
    vae, hparams = load_vae_model(run_folder, device)
    
    # Determine latent dimensions
    latent_shape = determine_latent_dimensions(vae, device)
    print(f"Determined latent shape: {latent_shape}")
    
    # Generate synthetic samples
    generate_synthetic_samples(vae, latent_shape, num_samples, device, output_dir)
    
    print("Synthetic sample generation completed!")

if __name__ == "__main__":
    main() 