import torch
from train import train_model
import copy

hyperparams = {
    'csv_file': "/space/mcdonald-syn01/1/projects/jsawant/2D_supervised_HC_vs_TLE/csv/pickle_prep_2025_01_17no_duplicates.csv",
    'device': torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'),
    'batch_size': 64,
    'learning_rate': 0.001,
    'num_epochs': 100,
    'dropout_rate': 0.3,
    'runs_dir': '/space/mcdonald-syn01/1/projects/jsawant/2D_supervised_HC_vs_TLE/runs',
    'val_split': 0.15,
    'use_early_stopping': True,
    'early_stopping_patience': 45,
    'scheduler_patience': 5,
    'scheduler_factor': 0.5,
    'num_workers': 4,
    'scheduler_type': 'cosine',  # or 'cosine'
    'cosine_t0': 30,  # Number of epochs for first restart
    'cosine_t_mult': 2,  # Multiply T_0 by this factor after each restart
    'cosine_eta_min': 5e-5,  # Minimum learning rate
    'use_best_threshold': False,
    'random_seed': 42,
    'split_seed':7,
    'weight_decay': 0,
    'use_mixed_precision': False,
    'supcon_temperature': 0.1,
    'warmup_epochs': 15,          # Number of epochs for warmup
    'final_lambda': 2,          # Final weight for BCE loss
    'lambda_schedule': 'linear',   # 'linear', 'cosine', or 'step'
    'generate_saliency_maps':True
}

# List of split seeds to try
split_seeds = [11,23,43]#, 123, 256, 789, 1024, 2048, 3072, 4096, 5000]

# Store results for each seed
results = []

for seed in split_seeds:
    print(f"\n=== Training with split_seed: {seed} ===")
    
    # Create a copy of hyperparams and update the split_seed
    current_hyperparams = copy.deepcopy(hyperparams)
    current_hyperparams['split_seed'] = seed
    
    # Train the model with current seed
    train_model(current_hyperparams, 'efficientNetV2')

# Print summary of all results
print("\nDone")
# for result in results:
#     print(f"Split seed {result['split_seed']}: {result['metrics']}")