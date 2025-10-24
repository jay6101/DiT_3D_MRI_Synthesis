# Classifier Component

This directory contains the implementation of deep learning classifiers for HC (Healthy Controls) vs TLE (Temporal Lobe Epilepsy) classification, along with comprehensive saliency analysis tools for model interpretability.

## Purpose

The classifier component serves multiple roles in our pipeline:
1. **Baseline Performance**: Establish classification performance on real data
2. **Synthetic Data Evaluation**: Assess quality of generated synthetic images
3. **Interpretability**: Generate saliency maps to understand model decisions

## Directory Structure

```
classifier/
├── model/                        # Neural network architectures
├── train.py                     # Main training script with k-fold CV
├── run.py                       # Training orchestration script  
├── dataset.py                   # Dataset handling with synthetic data support
├── utils.py                     # Training utilities and metrics
├── generate_saliency_maps.py    # Comprehensive saliency analysis
├── run_saliency_generation.py   # Saliency generation orchestration
├── vanilla_backprop.py          # Vanilla backpropagation saliency
├── SALIENCY_README.md           # Detailed saliency analysis documentation
├── saliency_requirements.txt    # Saliency-specific dependencies
└── new_runs/                    # Training run outputs and checkpoints
```

## Model Architectures

### Supported Models

The framework supports multiple architectures through dynamic model loading:

```python
# Available model architectures
models = [
    'efficientNetV2',    # 2D EfficientNet with modified first layer
]
```

### Model Selection

```python
def get_model_class(modelname):
    """Dynamically import model based on name"""
    module = __import__(f'model.{modelname}', fromlist=['MRIClassifier'])
    return getattr(module, 'MRIClassifier')
```

## Usage

### Key Training Features

#### Synthetic Data Integration
```python
# Dataset supports mixing real and synthetic data
train_dataset = MRIDataset(
    train_df, 
    hyperparams, 
    train=True, 
    num_samples=2723,           # Total samples
    num_synth_samples=1361      # 50% synthetic data
)
```

#### Balanced Sampling
```python
# Ensures balanced batches across HC/TLE classes
train_sampler = BalancedBatchSampler(train_dataset, batch_size)
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
```

## Training Configuration

### Hyperparameters

```python
hyperparams = {
    # Model architecture
    'dropout_rate': 0.5,
    
    # Training parameters
    'batch_size': 128,                  
    'learning_rate': 1e-3,
    'weight_decay': 0,
    'num_epochs': 100,
    
    # Scheduler settings
    'scheduler_type': 'cosine',         # 'cosine' or 'plateau'
    'cosine_t0': 25,                   # Cosine annealing period
    'cosine_t_mult': 2,                # Period multiplier
    'cosine_eta_min': 1e-5,            # Minimum learning rate
    
    # Data settings
    'val_split': 0.37,                  # Validation split ratio
    'num_workers': 4,                  # DataLoader workers
    'pin_memory': True,                # GPU memory optimization
    
}
```

### Loss Function

```python
# Binary cross-entropy with logits
criterion = nn.BCEWithLogitsLoss()

# Optional: Weighted loss for class imbalance
pos_weight = torch.tensor(neg_samples / pos_samples)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

## Evaluation Metrics

### Classification Metrics

```python
def calculate_metrics(y_true, y_pred, y_scores):
    """
    Comprehensive metric calculation
    
    Returns:
    - accuracy: Overall classification accuracy
    - sensitivity: True positive rate (recall)
    - specificity: True negative rate  
    - precision: Positive predictive value
    - f1_score: Harmonic mean of precision and recall
    - auc: Area under ROC curve
    """
```

## Saliency Analysis

For detailed saliency analysis documentation, see [SALIENCY_README.md](SALIENCY_README.md).

### Quick Start: Saliency Generation

```bash
python run_saliency_generation.py \
    --model_checkpoint /path/to/best_model.pth \
    --data_csv /path/to/test_data.csv \
    --output_dir ./saliency_outputs \
    --methods vanilla_backprop guided_backprop grad_cam
```

## Implementation Details

### Dataset Handling (`dataset.py`)

```python
class MRIDataset(Dataset):
    """
    Handles both real and synthetic MRI data
    
    Features:
    - NIfTI file loading with nibabel
    - Automatic synthetic data mixing
    - Data augmentation (optional)
    - Label encoding (HC=0, TLE=1)
    - Memory-efficient loading
    """
```

### Balanced Batch Sampling

```python
class BalancedBatchSampler:
    """
    Ensures each batch has balanced HC/TLE samples
    
    Benefits:
    - Stable training dynamics
    - Consistent gradient signals
    - Reduced class imbalance effects
    """
```

## Training Pipeline

### 1. Data Preparation
```python
# Load and prepare datasets
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# Filter to HC/TLE classes only
train_df = train_df[train_df["HC_vs_LTLE_vs_RTLE_string"].isin(["right", "left", "HC"])]
```


### 3. Model Training
```python
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # Initialize model for current fold
    model = MRIClassifier(dropout_rate=hyperparams['dropout_rate'])
    
    # Train with early stopping
    best_val_loss = train_fold(model, train_loader, val_loader)
    
    # Save fold results
    save_fold_metrics(fold_metrics, fold_dir)
```

### 4. Final Evaluation
```python
# Test on held-out test set
test_metrics = evaluate_model(best_model, test_loader)

# Generate saliency maps for interpretability
generate_saliency_maps(best_model, test_loader, output_dir)
```

## Synthetic Data Evaluation

### Integration Strategy

1. **Mixed Training**: Train on real + synthetic data
2. **Ablation Studies**: Compare real-only vs mixed training
3. **Quality Assessment**: Measure performance drop/gain with synthetic data

## Output Structure

```
new_runs/
└── run_YYYYMMDD_HHMMSS_modelname/
    ├── parameters.json           # Hyperparameters
    ├── fold_1/
        ├── best_model.pth       # Best model checkpoint
        ├── fold_metrics.json    # Fold performance metrics
        └── saliency_maps/       # Generated saliency maps
```

## Integration with Pipeline

### VAE Integration
```python
# Evaluate VAE reconstructions
vae_reconstructions = vae.decode(latent_codes)
classification_performance = classifier.evaluate(vae_reconstructions)
```

### Diffusion Integration  
```python
# Evaluate diffusion-generated samples
synthetic_samples = diffusion_model.sample(num_samples=1000)
synthetic_performance = classifier.evaluate(synthetic_samples)
```