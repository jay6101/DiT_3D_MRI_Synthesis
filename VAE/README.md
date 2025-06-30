# VAE (Variational Autoencoder) Component

This directory contains the implementation of a 3D Variational Autoencoder for encoding T1-weighted MRI brain images into a compact latent representation.

## Purpose

The VAE serves as the first stage in our diffusion pipeline:
1. **Encode** 3D MRI volumes into a lower-dimensional latent space
2. **Decode** latent representations back to image space
3. **Enable** efficient diffusion modeling in latent space rather than high-dimensional image space

## Directory Structure

```
VAE/
├── model/
│   ├── maisi_vae.py              # Main VAE architecture (MAISI-based)
│   ├── discriminator.py          # Discriminator for adversarial training
│   ├── lpips3D.py               # 3D LPIPS perceptual loss
│   ├── resnet_blocks.py         # ResNet building blocks
│   └── enc_dec.py               # Encoder/decoder components
├── train.py                      # Main training script
├── infer.py                     # Inference script for encoding/decoding
├── save_latent.py               # Save latent representations
├── dataset.py                   # Dataset handling for VAE training
├── utils.py                     # Training utilities and helper functions
├── visualize_recon.ipynb        # Reconstruction visualization notebook
├── runs/                        # Training run outputs
└── best_runs/                   # Best model checkpoints
```

## Architecture Details

### VAE Model (`maisi_vae.py`)
- **Type**: 3D Variational Autoencoder based on MAISI architecture
- **Input**: 3D MRI volumes (typically 91×109×91×1)
- **Latent Space**: Compact representation (configurable dimensions)
- **Features**:
  - ResNet blocks for stable training
  - Optional attention mechanisms
  - Configurable channel dimensions
  - Memory-efficient implementation

### Discriminator (`discriminator.py`)
- **Purpose**: Adversarial training to improve reconstruction quality
- **Architecture**: 3D convolutional discriminator
- **Loss**: Helps VAE generate more realistic reconstructions

### Perceptual Loss (`lpips3D.py`)
- **Purpose**: 3D extension of LPIPS (Learned Perceptual Image Patch Similarity)
- **Function**: Measures perceptual similarity in reconstruction loss
- **Benefit**: Improves visual quality of reconstructions

## Usage

### Training

```bash
python train.py --config_file config.json
```

### Key Training Parameters

- `spatial_dims`: 3 (for 3D volumes)
- `in_channels`: 1 (single-channel T1w images)
- `out_channels`: 1 (reconstructed single-channel)
- `channels`: (32, 64, 128) - encoder/decoder channel progression
- `latent_channels`: 4 - latent space dimensionality
- `num_res_blocks`: (1, 1, 1) - ResNet blocks per level

### Inference

```bash
python infer.py --model_checkpoint best_model.pth --input_data data.nii
```

### Saving Latent Representations

```bash
python save_latent.py --model_checkpoint best_model.pth --dataset train_data.csv
```

## Training Process

### Loss Components

1. **Reconstruction Loss**: MSE between input and reconstructed images
2. **KL Divergence**: Regularization term for latent space
3. **Perceptual Loss**: LPIPS-based perceptual similarity


## Configuration

Key hyperparameters in training:

```python
hyperparams = {
    'batch_size': 2,           # Small batch size due to 3D memory requirements
    'learning_rate': 1e-4,     # Conservative learning rate
    'num_epochs': 100,         # Extended training for convergence
    'weight_decay': 1e-5,      # L2 regularization
    'beta': 1.0,              # KL divergence weight
}
```

## Evaluation

### Metrics

- **Reconstruction Quality**: L1 loss, LPIPS loss
- **Latent Space Quality**: KL divergence
- **Perceptual Quality**: LPIPS scores, visual assessment

### Visualization

Use `visualize_recon.ipynb` to:
- Compare original vs reconstructed images
- Visualize latent space interpolations
- Assess reconstruction quality across different brain regions


