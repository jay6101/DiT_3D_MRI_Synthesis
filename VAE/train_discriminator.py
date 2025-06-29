import os
import torch
import json
from datetime import datetime
from torch.utils.data import DataLoader

from dataset import MRIDataset
from model.maisi_vae import VAE_Lite
from model.patchgan_discriminator import PatchDiscriminator
from model.discriminator import Discriminator
from model.adversarial_loss import PatchAdversarialLoss
from utils import set_random_seeds, compute_gan_loss

def train_discriminator(hparams):
    # Set random seeds for reproducibility
    #set_random_seeds(hparams['random_seed'])

    # Create a directory for this discriminator training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #run_dir = os.path.join(hparams['runs_dir'], f'disc_run_{timestamp}')
    run_dir = "/space/mcdonald-syn01/1/projects/jsawant/DSC250/VAE_GAN/gan"
    #os.makedirs(run_dir, exist_ok=True)

    # Save hyperparameters used for this run
    hparams_save = hparams.copy()
    hparams_save['device'] = str(hparams['device'])
    with open(os.path.join(run_dir, 'hparams.json'), 'w') as f:
        json.dump(hparams_save, f, indent=4)

    # Prepare datasets and dataloaders
    train_dataset = MRIDataset(hparams["train_csv_file"], train=True)
    val_dataset   = MRIDataset(hparams["val_csv_file"], train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams['batch_size'],
        shuffle=True,
        num_workers=hparams['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=hparams['batch_size'],
        shuffle=False,
        num_workers=hparams['num_workers'],
        pin_memory=True
    )

    device = hparams['device']

    # Initialize the VAE and load its checkpoint; then freeze its parameters.
    vae = VAE_Lite(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(32, 64, 128),
        num_res_blocks=(1, 1, 1),
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
    
    checkpoint = torch.load(hparams['vae_checkpoint'], map_location=device)
    vae.load_state_dict(checkpoint['vae_state_dict'])
    vae.eval()  # Set VAE to evaluation mode
    for param in vae.parameters():
        param.requires_grad = False  # Freeze VAE parameters

    # Initialize the discriminator and its optimizer
    disc = Discriminator().to(hparams['device'])
    checkpoint = torch.load(hparams['disc_checkpoint'], map_location=device)
    disc.load_state_dict(checkpoint['disc_state_dict'])
    del checkpoint
    disc.train()
    # disc = PatchDiscriminator(
    #     spatial_dims=3,
    #     num_layers_d=3,
    #     channels=32,
    #     in_channels=1,
    #     out_channels=1,
    #     norm="INSTANCE",
    # ).to(device)
    # disc.train()
    optimizer_disc = torch.optim.Adam(
        disc.parameters(),
        lr=hparams['disc_lr'],
        weight_decay=hparams['weight_decay']
    )
    # adv_loss = PatchAdversarialLoss(criterion="least_squares")

    best_val_loss = float('inf')
    patience_counter = 0

    # Main training loop for the discriminator
    for epoch in range(hparams['num_epochs']):
        disc.train()
        running_loss = 0.0
        running_d_fake = 0.0
        running_d_real = 0.0
        num_batches = 0
        
        for batch_idx, (images, _, _) in enumerate(train_loader):
            images = images.float().to(device)

            # Generate reconstructions from the frozen VAE
            with torch.no_grad():
                x_recon, _, _ = vae(images)
            
            # Forward pass on real images
            real_preds = disc(images)
            d_loss_real = compute_gan_loss(real_preds, True, hparams['gan_loss_type'])
            
            # Forward pass on fake (reconstructed) images
            fake_preds = disc(x_recon)
            d_loss_fake = compute_gan_loss(fake_preds, False, hparams['gan_loss_type'])
            
            # Total discriminator loss (optionally weighted by lambda_disc)
            d_loss = hparams['lambda_disc'] * (d_loss_real + d_loss_fake)
            
            optimizer_disc.zero_grad()
            d_loss.backward()
            optimizer_disc.step()

            running_loss += d_loss.item()
            running_d_fake += d_loss_fake.item()
            running_d_real += d_loss_real.item()
            num_batches += 1

            if (batch_idx)%10==0:
                print(f"Epoch [{epoch+1}/{hparams['num_epochs']}], Running Loss: {running_loss/10:.4f}, D_real: {running_d_real/10:.4f}, D_fake:  {running_d_fake/10:.4f}")
                running_loss = 0.0
                running_d_real = 0.0
                running_d_fake = 0.0
                # torch.save({
                # 'epoch': epoch,
                # 'disc_state_dict': disc.state_dict(),
                # }, os.path.join(run_dir, 'best_disc.pth'))
                # print("disc saved")
        avg_train_loss = running_loss / num_batches
        print(f"Epoch [{epoch+1}/{hparams['num_epochs']}], Average Training Loss: {avg_train_loss:.4f}")

        # Validation loop (no optimization)
        disc.eval()
        val_loss = 0.0
        running_loss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for i,(images, _, _ )in enumerate(val_loader):
                images = images.float().to(device)
                x_recon, _, _ = vae(images)
                
                real_preds = disc(images)
                d_loss_real = compute_gan_loss(real_preds, True, hparams['gan_loss_type'])
                
                fake_preds = disc(x_recon)
                d_loss_fake = compute_gan_loss(fake_preds, False, hparams['gan_loss_type'])
                
                loss = hparams['lambda_disc'] * (d_loss_real + d_loss_fake)
                val_loss += loss.item()
                num_val_batches += 1
                
        avg_val_loss = val_loss / num_val_batches
        print(f"-------------------Epoch [{epoch+1}/{hparams['num_epochs']}], Validation Loss: {avg_val_loss:.4f}-------------")

        # Save checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'disc_state_dict': disc.state_dict(),
                'val_loss': avg_val_loss,
            }, os.path.join(run_dir, 'best_disc.pth'))
            print(f"** Best discriminator model saved at epoch {epoch+1}")
        else:
            patience_counter += 1
            if hparams.get('use_early_stopping', False) and patience_counter >= hparams['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("Discriminator training complete.")

if __name__ == "__main__":
    hparams = {
        'train_csv_file': "/space/mcdonald-syn01/1/projects/jsawant/DSC250/data_csvs/train.csv",
        'val_csv_file':   "/space/mcdonald-syn01/1/projects/jsawant/DSC250/data_csvs/val.csv",
        'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        'batch_size': 8,
        'disc_lr': 1e-4,
        'weight_decay': 1e-6,
        'num_epochs': 1,
        'runs_dir': '/space/mcdonald-syn01/1/projects/jsawant/DSC250/VAE_GAN/runs',
        'random_seed': 42,
        'use_early_stopping': True,
        'early_stopping_patience': 35,
        'lambda_disc': 1.0,         # Weight for the discriminator loss
        'gan_loss_type': 'lsgan',     # Options: 'bce', 'lsgan', 'hinge', etc.
        'num_workers': 4,
        'vae_checkpoint': '/space/mcdonald-syn01/1/projects/jsawant/DSC250/VAE_GAN/gan/best_vae_gan.pth',
        'disc_checkpoint': '/space/mcdonald-syn01/1/projects/jsawant/DSC250/VAE_GAN/gan/best_disc.pth'
    }
    train_discriminator(hparams)
 