import os
import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from dataset import MRIDataset
from model.maisi_vae import VAE_Lite
#from model.discriminator import Discriminator
from tqdm import tqdm
import pickle

def load_models(run_folder, device):
    """Load the trained VAE and Discriminator models."""
    # Load hyperparameters
    with open(os.path.join(run_folder, 'hparams.json'), 'r') as f:
        hparams = json.load(f)
    hparams['device'] = device
    
    # Initialize models
    #vae = VAE(use_reparam=True).to(device)
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
    ).to(hparams['device'])
    #disc = Discriminator().to(device)
    
    # Load checkpoint
    checkpoint = torch.load(os.path.join(run_folder, 'best_vae_gan.pth'), 
                          map_location=device)
    
    vae.load_state_dict(checkpoint['vae_state_dict'])
    #disc.load_state_dict(checkpoint['disc_state_dict'])
    
    return vae, hparams

def infer(run_folder, latent_dir, device='cuda:1'):
    """Run inference and save reconstructions."""
    print(f"Running inference for {run_folder}")
    
    # Create output directory
    # recon_dir = os.path.join(run_folder, 'reconstructions')
    # os.makedirs(recon_dir, exist_ok=True)
    
    # Load models
    vae, hparams = load_models(run_folder, device)
    vae.eval()
    #disc.eval()
    
    # Create dataset and dataloader
    val_dataset = MRIDataset(hparams['val_csv_file'], train=False)
    val_loader = DataLoader(val_dataset, 
                          batch_size=hparams['batch_size'],
                          shuffle=False,
                          num_workers=hparams['num_workers'],
                          pin_memory=True)
    
    print("Generating reconstructions...")
    with torch.no_grad():
        for batch_idx, (images, labels, paths) in enumerate(tqdm(val_loader)):
            images = images.float().to(device)
            
            # Generate reconstructions
            recon_images, mus, logvars = vae(images)
            
            # Convert to numpy and save
            for i, (mu, logvar, label, path) in enumerate(zip(mus,logvars,labels, paths)):
                # Get filename from path
                filename = os.path.basename(path).split('.')[0]
                
                # Convert to numpy arrays
                mu = mu.cpu().numpy()
                logvar = logvar.cpu().numpy()
                
                latent = {'mu': mu, 'logvar':logvar, 'label': label.item()}
                # print(mu.shape)
                # print(logvar.shape)
                # print(label.item())
                
                with open(os.path.join(latent_dir, f'{filename}_latent.pkl'), 'wb') as f:
                    pickle.dump(latent, f)
                
                
                # Optionally save latent vectors
                # if mu is not None:
                #     latent = mu[i].cpu().numpy()
                #     np.save(os.path.join(recon_dir, f'{filename}_latent.npy'), latent)

    print(f"Latents saved to {latent_dir}")

if __name__ == "__main__":
    hparams = {
        'run_folder': "/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/VAE/best_runs/vae_run_20250226_161525",
        'device': 'cuda:0'
    }
    latent_dir = "/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/diffusion/latent_data/val"
    
    infer(hparams['run_folder'],latent_dir, hparams['device'])
