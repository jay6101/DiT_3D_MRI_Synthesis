import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from dataset import MRIDataset, BalancedBatchSampler
from utils import calculate_metrics, validate, prepare_fold_data, save_fold_metrics, calculate_overall_metrics
import pandas as pd
import os
import numpy as np
from datetime import datetime
import json
import random
import torch.backends.cudnn as cudnn
from torch.amp import autocast
import shutil
import math
from sklearn.model_selection import train_test_split

def set_random_seeds(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    # cudnn.deterministic = True
    # cudnn.benchmark = False
    
def get_model_class(modelname):
    """Dynamically import and return the model class based on model name."""
    try:
        module = __import__(f'model.{modelname}', fromlist=['MRIClassifier'])
        return getattr(module, 'MRIClassifier')
    except ImportError:
        raise ImportError(f"Could not import model {modelname} from models package")
    except AttributeError:
        raise AttributeError(f"Model {modelname} does not contain MRIClassifier class")

def train_model(hyperparams, modelname):
    # Get the model class dynamically
    MRIClassifier = get_model_class(modelname)
    # Set seeds at the start
    set_random_seeds(hyperparams['random_seed'])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(hyperparams['runs_dir'], f'run_{timestamp}_{modelname}')
    os.makedirs(run_dir, exist_ok=True)
    
    with open(os.path.join(run_dir, 'parameters.json'), 'w') as f:
        # Convert torch.device to string for JSON serialization
        hyperparams_to_save = hyperparams.copy()
        hyperparams_to_save['device'] = str(hyperparams_to_save['device'])
        json.dump(hyperparams_to_save, f, indent=4)
    # Save copies of source files
    source_files = [
        'run.py',
        'train.py',
        'utils.py',
        'dataset.py',
        f'model/{modelname}.py'  # Assuming this is the model file
    ]
    
    for file in source_files:
        src_path = os.path.join(os.path.dirname(__file__), file)
        dst_path = os.path.join(run_dir, os.path.basename(file))
        shutil.copy2(src_path, dst_path)
    
    print(f"Using device: {hyperparams['device']}")
    # df = pd.read_csv(hyperparams['csv_file'])
    # lr_df = df.loc[df["HC_vs_LTLE_vs_RTLE_string"].isin(["right","left"])]
    # hc_df = df.loc[df["HC_vs_LTLE_vs_RTLE_string"].isin(["HC"])]
    #df["HC_vs_TLE"] = np.where(df["HC_vs_LTLE_vs_RTLE_string"].isin(["right", "left"]), 1, 0)
    
    # Initialize StratifiedKFold with 5 folds
    #skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=hyperparams['split_seed'])
    all_folds_metrics = {'fold_metrics': [], 'hyperparameters': hyperparams}

    # hc_fold_size = len(hc_df) // 5
    # hc_indices = np.arange(len(hc_df))
    # np.random.RandomState(hyperparams['split_seed']).shuffle(hc_indices)
    # hc_folds = np.array_split(hc_indices, 5)
    
    # Run training for each fold
    for fold_idx in range(1):
        print(f"\nTraining Fold {fold_idx + 1}/5")
        fold_dir = os.path.join(run_dir, f'fold_{fold_idx + 1}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # # Get HC indices for current fold
        # hc_test_idx = hc_folds[fold_idx]
        # hc_train_idx = np.concatenate([hc_folds[i] for i in range(5) if i != fold_idx])
        
        # Split HC training data into train and validation with same ratio as LR data
        # hc_train_df, hc_val_df = train_test_split(
        #     hc_df.iloc[hc_train_idx],
        #     test_size=hyperparams['val_split'],
        #     random_state=hyperparams['split_seed']
        # )
        
        # Prepare data using the split dataframes
        # train_csv, val_csv, test_csv = prepare_fold_data(
        #     lr_df.iloc[train_idx], 
        #     lr_df.iloc[test_idx],
        #     hc_train_df,
        #     hc_val_df,
        #     hc_df.iloc[hc_test_idx],
        #     hyperparams, 
        #     fold_dir
        # )
        
        # Prepare data
        #train_csv, val_csv, test_csv = prepare_fold_data(lr_df, train_idx, test_idx, hyperparams, fold_dir, hc_df)
        full_df = pd.read_csv(hyperparams['train_csv'])
        test_df = pd.read_csv(hyperparams['test_csv'])
        #full_df= full_df[~full_df['file'].str.contains("/space/mcdonald-syn01/1/BIDS//enigma_conglom//derivatives//cat12_copy/sub-upenn", na=False)]
        full_df = full_df.loc[full_df["HC_vs_LTLE_vs_RTLE_string"].isin(["right","left","HC"])]
        #test_df= test_df[~test_df['file'].str.contains("/space/mcdonald-syn01/1/BIDS//enigma_conglom//derivatives//cat12_copy/sub-upenn", na=False)]
        test_df = test_df.loc[test_df["HC_vs_LTLE_vs_RTLE_string"].isin(["right","left","HC"])]
        
        neg_labels = len(full_df[full_df['HC_vs_LTLE_vs_RTLE_string']=="HC"])
        pos_labels = len(full_df) - neg_labels
        #pos_weight = torch.tensor(neg_labels/pos_labels)
        print(f"negative labels:{neg_labels}")
        print(f"positive labels:{pos_labels}")
        
        
        train_df = full_df
        test_df, val_df = train_test_split(
            test_df,
            test_size=hyperparams['val_split'],
            stratify=test_df['HC_vs_LTLE_vs_RTLE_string'],
            random_state=hyperparams['split_seed']
        )
        #print(f"Length of test dataset: {len(test_df)}")
        # Initialize datasets and dataloaders
        train_dataset = MRIDataset(train_df, hyperparams, train=True, num_samples=500, num_synth_samples=int(1*500))
        val_dataset = MRIDataset(val_df, hyperparams, train=False)
        test_dataset = MRIDataset(test_df, hyperparams, train=False)
        print(f"Length of train dataset: {len(train_dataset)}")
        print(f"Length of val dataset: {len(val_dataset)}")
        print(f"Length of test dataset: {len(test_dataset)}")
        
        train_sampler = BalancedBatchSampler(train_dataset, hyperparams['batch_size'])
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, # batch_size=hyperparams['batch_size']
                                num_workers=hyperparams['num_workers'], pin_memory=True, prefetch_factor=2)
        val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], 
                              shuffle=False, num_workers=hyperparams['num_workers'], pin_memory=True, prefetch_factor=2)
        test_loader = DataLoader(test_dataset, batch_size=hyperparams['batch_size'], 
                               shuffle=False, num_workers=hyperparams['num_workers'], pin_memory=True)
        
        # Initialize model and training components
        model = MRIClassifier(dropout_rate=hyperparams['dropout_rate']).to(hyperparams['device'])
        bce_criterion = nn.BCEWithLogitsLoss()           #pos_weight=pos_weight)
        #criterion = nn.BCEWithLogitsLoss()
        # supcon_criterion = SupConLoss(
        #     temperature=hyperparams['supcon_temperature'],
        #     device=hyperparams['device']
        # )
        optimizer = optim.Adam(
            model.parameters(), 
            lr=hyperparams['learning_rate'],
            weight_decay=hyperparams['weight_decay']
        )
        
        # Modified scheduler initialization
        if hyperparams['scheduler_type'].lower() == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',
                patience=hyperparams['scheduler_patience'],
                factor=hyperparams['scheduler_factor']
            )
        elif hyperparams['scheduler_type'].lower() == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=hyperparams['cosine_t0'],
                T_mult=hyperparams['cosine_t_mult'],
                eta_min=hyperparams['cosine_eta_min']
            )
        else:
            raise ValueError("scheduler_type must be either 'plateau' or 'cosine'")
        
        # Initialize metrics
        fold_metrics = {
            'train_losses': [], 'val_losses': [], 'train_metrics': [],
            'val_metrics': [], 'epochs': [], 'best_val_loss': float('inf'),
            'best_val_metrics': None,
            'test_metrics': None
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        scaler = torch.amp.GradScaler()  # Updated GradScaler initialization
        
        warmup_epochs = hyperparams.get('warmup_epochs', 10)
        final_lambda = hyperparams.get('final_lambda', 0.5)
        lambda_schedule = hyperparams.get('lambda_schedule', 'linear')
        
        # Training loop
        for epoch in range(hyperparams['num_epochs']):
            model.train()
            running_loss = 0.0
            epoch_loss = 0.0
            all_train_labels, all_train_predictions, all_train_scores = [], [], []
            
            # # Calculate lambda based on schedule
            # if lambda_schedule == 'linear':
            #     # Linear warmup from 0 to final_lambda
            #     lambda_ = min(epoch / warmup_epochs, 1.0) * final_lambda
            
            # elif lambda_schedule == 'cosine':
            #     # Cosine schedule - smoother transition
            #     if epoch < warmup_epochs:
            #         lambda_ = 0.5 * (1 - math.cos(math.pi * epoch / warmup_epochs)) * final_lambda
            #     else:
            #         lambda_ = final_lambda
            
            # elif lambda_schedule == 'step':
            #     # Step schedule - discrete jumps
            #     if epoch < warmup_epochs // 2:
            #         lambda_ = 0.1 * final_lambda
            #     elif epoch < warmup_epochs:
            #         lambda_ = 0.5 * final_lambda
            #     else:
            #         lambda_ = final_lambda
            
            for i, (images, labels,_) in enumerate(train_loader):
                images = images.to(hyperparams['device'])
                labels = labels.float().to(hyperparams['device'])
                
                optimizer.zero_grad()
                
                if hyperparams.get('use_mixed_precision', False):
                    with autocast(device_type='cuda'):  # Use autocast for mixed precision
                        outputs, attention_weights = model(images)
                        outputs = outputs.squeeze()
                        loss = bce_criterion(outputs, labels)
                
                    scaler.scale(loss).backward()  # Scale the loss for mixed precision
                    scaler.step(optimizer)  # Step the optimizer
                    scaler.update()  # Update the scaler
                else:
                    outputs, _ = model(images)
                    outputs = outputs.squeeze()
                    loss = bce_criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item()
                epoch_loss += loss.item()
                scores = torch.sigmoid(outputs)
                predictions = (scores > 0.5).float()
                
                all_train_labels.extend(labels.cpu().detach().numpy())
                all_train_predictions.extend(predictions.cpu().detach().numpy())
                all_train_scores.extend(scores.cpu().detach().numpy())
                
                if i % 10 == 9:
                    print(f'Fold [{fold_idx + 1}/5] Epoch [{epoch+1}/{hyperparams["num_epochs"]}], '
                          f'Step [{i+1}], Loss: {running_loss/10:.4f}')
                    running_loss = 0.0

            
            # Calculate metrics
            train_metrics = calculate_metrics(
                np.array(all_train_labels),
                np.array(all_train_predictions),
                np.array(all_train_scores)
            )
            train_metrics['loss'] = epoch_loss / len(train_loader)
            val_metrics, best_threshold = validate(model, val_loader, bce_criterion, hyperparams, return_threshold=True)
            val_metrics['best_threshold'] = best_threshold
            
            # Print progress with additional metrics
            print(f'Fold [{fold_idx + 1}/5] Epoch [{epoch+1}/{hyperparams["num_epochs"]}]:')
            print(f'Train - Loss: {train_metrics["loss"]:.4f}, '
                  f'AUC: {train_metrics["auc_roc"]:.4f}, '
                  f'PPV: {train_metrics["ppv"]:.4f}, '
                  f'Acc: {train_metrics["accuracy"]:.4f}')
            print(f'Val - Loss: {val_metrics["loss"]:.4f}, '
                  f'AUC: {val_metrics["auc_roc"]:.4f}, '
                  f'PPV: {val_metrics["ppv"]:.4f}, '
                  f'Acc: {val_metrics["accuracy"]:.4f}')
            
            # Modified scheduler step
            if hyperparams['scheduler_type'].lower() == 'plateau':
                scheduler.step(val_metrics['loss'])
            else:  # cosine
                scheduler.step()
            
            fold_metrics['train_metrics'].append(train_metrics)
            fold_metrics['val_metrics'].append(val_metrics)
            fold_metrics['epochs'].append(epoch)
            
            # Early stopping check
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                fold_metrics['best_val_metrics'] = val_metrics.copy()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': val_metrics
                }, os.path.join(fold_dir, 'best_model.pth'))
                print(f'Best model saved at epoch {epoch+1}')
            else:
                patience_counter += 1
                if hyperparams['use_early_stopping'] and patience_counter >= hyperparams['early_stopping_patience']:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
            
            fold_metrics['best_val_loss'] = best_val_loss
            save_fold_metrics(fold_metrics, fold_dir)
        
        # Test phase
        best_model = MRIClassifier(dropout_rate=hyperparams['dropout_rate']).to(hyperparams['device'])
        checkpoint = torch.load(os.path.join(fold_dir, 'best_model.pth'))
        best_model.load_state_dict(checkpoint['model_state_dict'])
        
        threshold = (fold_metrics['best_val_metrics']['best_threshold'] 
                    if hyperparams['use_best_threshold'] else 0.5)
        test_metrics, individual_results = validate(
            best_model, test_loader, bce_criterion, hyperparams, 
            fixed_threshold=threshold, return_individual_results=True
        )

        # Create performance labels (TP, FP, TN, FN)
        performance_labels = []
        for true_label, pred_label in zip(individual_results['labels'], individual_results['predictions']):
            if true_label == 1 and pred_label == 1:
                performance_labels.append('TP')
            elif true_label == 0 and pred_label == 1:
                performance_labels.append('FP')
            elif true_label == 1 and pred_label == 0:
                performance_labels.append('FN')
            else:
                performance_labels.append('TN')

        # Create DataFrame for individual results
        results_df = pd.DataFrame({
            'img_path': individual_results['paths'],
            'predicted_score': individual_results['scores'],
            'original_label': individual_results['labels'],
            'predicted_label': individual_results['predictions'],
            'performance_type': performance_labels
        })

        # Save to CSV
        results_df.to_csv(os.path.join(fold_dir, 'individual_test_performance.csv'), index=False)

        test_metrics['threshold_used'] = threshold
        
        # Save results
        fold_metrics['test_metrics'] = test_metrics
        save_fold_metrics(fold_metrics, fold_dir)
        all_folds_metrics['fold_metrics'].append(fold_metrics)
        
        # Print test results
        print(f"\nFold {fold_idx + 1} Test Results:")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # After test phase
        if hyperparams['generate_saliency_maps']==True:
            print("Generating saliency maps....")
            #generate_fold_saliency_maps(fold_dir, hyperparams)
    
    # Calculate and save overall results
    overall_results = calculate_overall_metrics(all_folds_metrics)
    all_folds_metrics_to_save = {}
    all_folds_metrics_to_save['overall_results'] = overall_results

    # Save all results
    with open(os.path.join(run_dir, 'all_folds_metrics.json'), 'w') as f:
        json.dump(all_folds_metrics_to_save, f, indent=4)

    # Print comprehensive summary
    print("\nOverall 5-Fold Cross-Validation Results:")
    print("-" * 50)
    for metric, values in overall_results.items():
        print(f"{metric.upper():12s}: {values['mean']:.4f} ± {values['std']:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    hyperparams = {
        'train_csv': "/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/data_csvs/train.csv",
        'test_csv': "/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/data_csvs/val.csv",
        'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        'batch_size': 128,
        'learning_rate': 1e-3,
        'num_epochs': 100,
        'dropout_rate': 0.3,
        'runs_dir': '/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/classifier/new_runs/runs_500',
        'val_split': 0.37,
        'use_early_stopping': True,
        'early_stopping_patience': 50,
        'scheduler_patience': 5,
        'scheduler_factor': 0.5,
        'num_workers': 4,
        'scheduler_type': 'cosine',  # or 'cosine'
        'cosine_t0': 25,  # Number of epochs for first restart
        'cosine_t_mult': 2,  # Multiply T_0 by this factor after each restart
        'cosine_eta_min': 1e-5,  # Minimum learning rate
        'use_best_threshold': False,
        'random_seed': 488,
        'sample_seed': 1000,
        'split_seed':7,
        'weight_decay': 0,
        'use_mixed_precision': False,
        'supcon_temperature': 0.1,
        'warmup_epochs': 15,          # Number of epochs for warmup
        'final_lambda': 2,          # Final weight for BCE loss
        'lambda_schedule': 'linear',   # 'linear', 'cosine', or 'step'
        'generate_saliency_maps':False,
        'synth_hc_folder_path':"/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/synthetic_data_pkls_HC",
        'synth_tle_folder_path':"/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/synthetic_data_pkls_TLE",
    }
    
    train_model(hyperparams,'efficientNetV2')
