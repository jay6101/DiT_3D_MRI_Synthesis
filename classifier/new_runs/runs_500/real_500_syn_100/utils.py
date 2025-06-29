import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve, 
    auc, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix,
    roc_curve,
    average_precision_score,
    accuracy_score
)
from sklearn.model_selection import train_test_split
import os
import json
import pandas as pd

def calculate_metrics(labels, predictions, scores, return_threshold=False):
    """Calculate various classification metrics"""
    # Calculate best threshold using validation data
    fpr, tpr, thresholds = roc_curve(labels, scores)
    optimal_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[optimal_idx]
    
    # Use the provided predictions if they exist, otherwise calculate using best_threshold
    if predictions is None:
        predictions = (scores >= best_threshold).astype(float)
    
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'auc_roc': roc_auc_score(labels, scores),
        'auc_pr': average_precision_score(labels, scores),
        'ppv': precision_score(labels, predictions),
        'sensitivity': recall_score(labels, predictions),
        'specificity': recall_score(labels, predictions, pos_label=0),
        'precision': precision_score(labels, predictions),
        'recall': recall_score(labels, predictions),
        'f1_score': f1_score(labels, predictions),
        'best_threshold': best_threshold
    }
    
    if return_threshold:
        return metrics, best_threshold
    return metrics

def validate(model, dataloader, criterion, hyperparams, fixed_threshold=None, return_threshold=False, return_individual_results=False):
    """Validate the model on the provided dataloader"""
    model.eval()
    val_loss = 0
    all_labels = []
    all_scores = []
    all_paths = []  # New: store image paths
    
    with torch.no_grad():
        for images, labels, paths in dataloader:  # Modified to unpack paths
            images = images.to(hyperparams['device'])
            labels = labels.float().to(hyperparams['device'])
            
            outputs, _ = model(images)
            outputs = outputs.view(-1)
            #print(f"Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            scores = torch.sigmoid(outputs)
            
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
            all_paths.extend(paths)  # New: store paths
    
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # If fixed_threshold is provided, use it to calculate predictions
    threshold = fixed_threshold if fixed_threshold is not None else 0.5
    all_predictions = (all_scores >= threshold).astype(float)
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_predictions, all_scores)
    metrics['loss'] = val_loss / len(dataloader)
    
    if return_individual_results:
        individual_results = {
            'paths': all_paths,
            'scores': all_scores,
            'labels': all_labels,
            'predictions': all_predictions
        }
        return metrics, individual_results
    
    if return_threshold:
        return metrics, metrics['best_threshold']
    
    return metrics

def prepare_fold_data(lr_train_df, lr_test_df, hc_train_df, hc_val_df, hc_test_df, hyperparams, fold_dir):
    """Prepare data for a specific fold"""
    # First split LR data into train and validation
    lr_train_df, lr_val_df = train_test_split(
        lr_train_df,
        test_size=hyperparams['val_split'],
        stratify=lr_train_df['HC_vs_LTLE_vs_RTLE_string'],
        random_state=hyperparams['split_seed']
    )
    
    # Combine LR and HC data
    train_df = pd.concat([lr_train_df, hc_train_df], ignore_index=True)
    val_df = pd.concat([lr_val_df, hc_val_df], ignore_index=True)
    test_df = pd.concat([lr_test_df, hc_test_df], ignore_index=True)
    
    # Print sizes for verification
    print("\nDataset sizes:")
    print(f"Train - LR: {len(lr_train_df)}, HC: {len(hc_train_df)}, Total: {len(train_df)}")
    print(f"Val   - LR: {len(lr_val_df)}, HC: {len(hc_val_df)}, Total: {len(val_df)}")
    print(f"Test  - LR: {len(lr_test_df)}, HC: {len(hc_test_df)}, Total: {len(test_df)}")
    
    # Save CSVs
    train_csv = os.path.join(fold_dir, 'train.csv')
    val_csv = os.path.join(fold_dir, 'val.csv')
    test_csv = os.path.join(fold_dir, 'test.csv')
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    return train_csv, val_csv, test_csv

def save_fold_metrics(metrics, fold_dir):
    """Save metrics to JSON file, converting numpy and torch types to native Python types"""
    def convert_to_python_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.device):
            return str(obj)  # Convert device to string representation
        elif isinstance(obj, dict):
            return {key: convert_to_python_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        return obj

    metrics_to_save = convert_to_python_types(metrics)
    
    with open(os.path.join(fold_dir, 'fold_metrics.json'), 'w') as f:
        json.dump(metrics_to_save, f, indent=4)

def calculate_overall_metrics(all_folds_metrics):
    """Calculate mean and std for all test metrics across folds"""
    metrics_to_calculate = [
        'accuracy', 'auc_roc', 'auc_pr', 'ppv', 
        'sensitivity', 'specificity', 'precision', 
        'recall', 'f1_score'
    ]
    
    overall_results = {}
    for metric in metrics_to_calculate:
        # Convert values to native Python float
        values = [float(fold['test_metrics'][metric]) 
                 for fold in all_folds_metrics['fold_metrics']]
        # Ensure mean and std are native Python floats
        mean_value = float(np.mean(values))
        std_value = float(np.std(values))
        overall_results[metric] = {
            'mean': mean_value,
            'std': std_value
        }
    
    return overall_results
