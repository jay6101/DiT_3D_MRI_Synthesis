import torch
import torch.nn.functional as F
import random, os, numpy as np

def set_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def kl_divergence(mu, logvar):
    # KL( q(z|x) || p(z)=N(0,I) )
    kld = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1, dim=[1,2,3,4])
    return kld.mean()

def compute_gan_loss(preds, is_real, loss_type):
    """Helper function to compute GAN losses based on specified type"""
    if loss_type == 'wgan':
        return -torch.mean(preds) if is_real else torch.mean(preds)
    elif loss_type == 'bce':
        target = torch.ones_like(preds) if is_real else torch.zeros_like(preds)
        return F.binary_cross_entropy_with_logits(preds, target)
    elif loss_type == 'lsgan':
        target = torch.ones_like(preds) if is_real else torch.zeros_like(preds)
        return F.mse_loss(preds, target)
    elif loss_type == 'hinge':
        if is_real:
            return torch.mean(F.relu(1 - preds))
        else:
            return torch.mean(F.relu(1 + preds))
    else:
        raise ValueError(f"Unsupported GAN loss type: {loss_type}")

def compute_recon_loss(x_recon, images, loss_type):
    """Helper function to compute reconstruction losses based on specified type"""
    if loss_type == 'l1':
        return F.l1_loss(x_recon, images)
    elif loss_type == 'mse':
        return F.mse_loss(x_recon, images)
    else:
        raise ValueError(f"Unsupported reconstruction loss type: {loss_type}")

def train_one_epoch(vae, disc, lpips_model, train_loader, opt_vae, opt_disc, hparams, current_epoch):
    vae.train()
    # if disc is not None:
    #     disc.train()
    device = hparams['device']

    # Add moving average tracking for loss adjustment
    ma_window = 20  # Window size for moving average
    d_fake_losses = []
    g_adv_losses = []
    
    # Track current lambda values
    current_lambda_adv = hparams['lambda_adv']
    current_lambda_disc = hparams['lambda_disc']
    
    # Define separate adjustment factors and bounds
    disc_adjustment_factor = 5  # More conservative adjustment for discriminator
    gen_adjustment_factor = 5   # More aggressive adjustment for generator
    
    # Separate min/max bounds for each lambda
    min_lambda_disc = 1e-15
    max_lambda_disc = 1e-4
    min_lambda_adv = 0.001
    max_lambda_adv = 0.2

    # Calculate total warmup steps
    total_warmup_steps = hparams['warmup_epochs'] * len(train_loader)
    current_step = current_epoch * len(train_loader)

    running_g, running_d = 0.0, 0.0
    running_recon, running_kl, running_lpips, running_adv = 0.0, 0.0, 0.0, 0.0
    running_d_real, running_d_fake = 0.0, 0.0

    epoch_running_recon = 0.0
    epoch_running_kl = 0.0
    epoch_running_lpips = 0.0
    epoch_running_adv = 0.0
    epoch_running_d_real = 0.0
    epoch_running_d_fake = 0.0
    epoch_running_g = 0.0
    epoch_running_d = 0.0
    
    for batch_idx, (images, labels, paths) in enumerate(train_loader):
        # Update current step
        current_step += 1
        
        # Calculate ramping factors for adversarial losses
        if current_step < total_warmup_steps:
            # Linear ramp from 0 to target value
            ramp_factor = current_step / total_warmup_steps
            current_lambda_adv = hparams['lambda_adv'] * ramp_factor
            current_lambda_disc = hparams['lambda_disc'] * ramp_factor
        else:
            # Use full values after warmup
            current_lambda_adv = hparams['lambda_adv']
            current_lambda_disc = hparams['lambda_disc']

        images = images.float().to(device)
        
        # Only run discriminator updates if disc exists
        # if disc is not None and batch_idx % 10 == 0 and current_step > 0:
        #     opt_disc.zero_grad()
        #     # Real
        #     real_preds = disc(images)
        #     d_loss_real = compute_gan_loss(real_preds, True, hparams['gan_loss_type'])

        #     # Fake
        #     x_recon, mu, logvar = vae(images)
        #     fake_preds = disc(x_recon.detach())
        #     d_loss_fake = compute_gan_loss(fake_preds, False, hparams['gan_loss_type'])

        #     # Total discriminator loss with ramped weight
        #     d_loss = current_lambda_disc * (d_loss_real + d_loss_fake)
        #     d_loss.backward()
        #     opt_disc.step()
        # else:
        #     d_loss = torch.tensor(0.0).to(device)
        #     d_loss_real = torch.tensor(0.0).to(device)
        #     d_loss_fake = torch.tensor(0.0).to(device)

        # 2) Update VAE (Generator)
        opt_vae.zero_grad()
        x_recon, mu, logvar = vae(images)
        # print("mu",mu.size())
        # print("logvar",logvar.size())

        # Recon + KL + LPIPS losses (always active)
        recon_loss = compute_recon_loss(x_recon, images, hparams['recon_loss_type'])
        kld = 0.0
        if mu is not None and logvar is not None:
            kld = kl_divergence(mu, logvar)

        with torch.no_grad():
            lpips_model.eval()
        lpips_val = lpips_model(x_recon, images)
        if lpips_val.dim() > 0:
            lpips_val = lpips_val.mean()
        
        # Only compute adversarial loss if using discriminator
        adv_loss = torch.tensor(0.0).to(device)
        if disc is not None and current_step > 0:
            disc.eval()
            gen_preds = disc(x_recon)
            if hparams['gan_loss_type'] == 'wgan':
                adv_loss = -torch.mean(gen_preds)
            else:
                # For non-WGAN losses, we want the generator to maximize P(real)
                adv_loss = compute_gan_loss(gen_preds, True, hparams['gan_loss_type'])

        # Weighted sum with ramped adversarial term
        g_loss = (hparams['lambda_recon'] * recon_loss
                  + hparams['lambda_kl'] * kld
                  + hparams['lambda_lpips'] * lpips_val
                  + current_lambda_adv * adv_loss)
        
        g_loss.backward()
        opt_vae.step()

        # # Track losses for lambda adjustment
        # d_fake_losses.append(d_loss_fake.item())
        # g_adv_losses.append(adv_loss.item())
        
        # # Adjust lambda values if we have enough samples
        # if len(d_fake_losses) % ma_window==0 and len(d_fake_losses)!=0:
        #     # Calculate moving averages
        #     avg_d_fake = sum(d_fake_losses[-ma_window:]) / ma_window
        #     avg_g_adv = sum(g_adv_losses[-ma_window:]) / ma_window
            
        #     if avg_d_fake < sum(d_fake_losses[-ma_window-10:-ma_window])/10: #and \
        #        #avg_g_adv > sum(g_adv_losses[-ma_window-10:-ma_window])/10:
        #         current_lambda_disc = 0#current_lambda_disc / disc_adjustment_factor
        #         current_lambda_adv = min(current_lambda_adv * gen_adjustment_factor, max_lambda_adv)
        #         print(f"Adjusting lambdas - Strengthening generator: λ_disc={current_lambda_disc:.2e}, λ_adv={current_lambda_adv:.2e}")
            
        #     else: #avg_d_fake > sum(d_fake_losses[-ma_window-10:-ma_window])/10: #and \
        #          #avg_g_adv < sum(g_adv_losses[-ma_window-10:-ma_window])/10:
        #         current_lambda_disc = hparams['lambda_disc']
        #         current_lambda_adv = max(current_lambda_adv / gen_adjustment_factor, min_lambda_adv)
        #         print(f"Adjusting lambdas - Strengthening discriminator: λ_disc={current_lambda_disc}, λ_adv={current_lambda_adv}")

        # Update running averages for both 10-batch and epoch-level metrics
        epoch_running_g += g_loss.item()
        # epoch_running_d += d_loss.item()
        epoch_running_recon += recon_loss.item()
        epoch_running_kl += kld
        epoch_running_lpips += lpips_val.item()
        epoch_running_adv += adv_loss.item()
        # epoch_running_d_real += d_loss_real.item()
        # epoch_running_d_fake += d_loss_fake.item()

        # Update running averages for 10-batch metrics
        running_g += g_loss.item()
        # running_d += d_loss.item()
        running_recon += recon_loss.item()
        running_kl += kld
        running_lpips += lpips_val.item()
        running_adv += adv_loss.item()
        # running_d_real += d_loss_real.item()
        # running_d_fake += d_loss_fake.item()

        # Print and reset running averages every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"Train Batch [{batch_idx + 1}/{len(train_loader)}] Average of last 10 - "
                  f"Recon: {running_recon/10:.4f}, "
                  f"KL: {running_kl/10:.4f}, "
                  f"LPIPS: {running_lpips/10:.4f}, "
                  f"Adv: {running_adv/10:.4f}, "
                  f"D_real: {running_d_real/1:.4f}, "
                  f"D_fake: {running_d_fake/1:.4f}")
            if running_adv/10 < 0.6:
                break
            
            # Reset running losses
            running_recon, running_kl = 0.0, 0.0
            running_lpips, running_adv = 0.0, 0.0
            running_d_real, running_d_fake = 0.0, 0.0
            running_g, running_d = 0.0, 0.0

    # Calculate epoch averages using the actual number of batches
    n_batches = len(train_loader)
    avg_recon = epoch_running_recon / n_batches
    avg_kl = epoch_running_kl / n_batches
    avg_lpips = epoch_running_lpips / n_batches
    avg_adv = epoch_running_adv / n_batches
    avg_d_real = epoch_running_d_real / n_batches
    avg_d_fake = epoch_running_d_fake / n_batches
    
    print(f"\nTrain Epoch Averages:")
    print(f"Generator Total Loss: {epoch_running_g / n_batches:.4f}")
    print(f"Discriminator Total Loss: {epoch_running_d / n_batches:.4f}")
    print(f"Individual Losses - Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}, "
          f"LPIPS: {avg_lpips:.4f}, Adv: {avg_adv:.4f}")
    print(f"Discriminator Losses - Real: {avg_d_real:.4f}, Fake: {avg_d_fake:.4f}")
    
    return epoch_running_g / n_batches, epoch_running_d / n_batches

def validate_one_epoch(vae, disc, lpips_model, val_loader, hparams):
    vae.eval()
    if disc is not None:
        disc.eval()
    device = hparams['device']
    
    running_g, running_d = 0.0, 0.0
    running_recon, running_kl, running_lpips, running_adv = 0.0, 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for images, labels, paths in val_loader:
            images = images.float().to(device)
            
            # Always generate reconstructions first
            x_recon, mu, logvar = vae(images)
            
            # Discriminator evaluation only if it exists
            d_loss = torch.tensor(0.0).to(device)
            if disc is not None:
                real_preds = disc(images)
                d_loss_real = compute_gan_loss(real_preds, True, hparams['gan_loss_type'])
                
                fake_preds = disc(x_recon)
                d_loss_fake = compute_gan_loss(fake_preds, False, hparams['gan_loss_type'])
                
                d_loss = d_loss_real + d_loss_fake
            
            # Generator losses
            recon_loss = compute_recon_loss(x_recon, images, hparams['recon_loss_type'])
            kld = 0.0
            if mu is not None and logvar is not None:
                kld = kl_divergence(mu, logvar)

            lpips_val = lpips_model(x_recon, images)
            if lpips_val.dim() > 0:
                lpips_val = lpips_val.mean()
            
            # Generator adversarial loss
            if disc is not None:
                if hparams['gan_loss_type'] == 'wgan':
                    adv_loss = -torch.mean(fake_preds)
                else:
                    adv_loss = compute_gan_loss(fake_preds, True, hparams['gan_loss_type'])
            else:
                adv_loss = torch.tensor(0.0).to(device)
            if disc is not None:
                g_loss = (hparams['lambda_recon'] * recon_loss
                        + hparams['lambda_kl'] * kld
                        + hparams['lambda_lpips'] * lpips_val
                        + hparams['lambda_adv'] * adv_loss)
            else:
                g_loss = (hparams['lambda_recon'] * recon_loss
                        + hparams['lambda_kl'] * kld
                        + hparams['lambda_lpips'] * lpips_val)
                
            running_d += d_loss.item()
            running_g += g_loss.item()
            running_recon += recon_loss.item()
            running_kl += kld
            running_lpips += lpips_val.item()
            running_adv += adv_loss.item()
    
    n_batches = len(val_loader)
    avg_recon = running_recon / n_batches
    avg_kl = running_kl / n_batches
    avg_lpips = running_lpips / n_batches
    avg_adv = running_adv / n_batches
    
    print(f"\nValidation Epoch Averages:")
    print(f"Generator Total Loss: {running_g / n_batches:.4f}")
    print(f"Discriminator Total Loss: {running_d / n_batches:.4f}")
    print(f"Individual Losses - Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}, "
          f"LPIPS: {avg_lpips:.4f}, Adv: {avg_adv:.4f}")
    
    return running_g / n_batches, running_d / n_batches
