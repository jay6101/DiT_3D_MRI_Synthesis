#!/usr/bin/env python3
"""
Saliency Map Generation Script for MRI Classification Model

This script loads a trained model from the syn_2723 run and generates saliency maps
using Captum for the validation dataset.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import nibabel as nib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.append('/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/classifier')
sys.path.append('/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/classifier/new_runs/runs_syn/syn_2723')

# Import captum attribution methods
from captum.attr import (
    IntegratedGradients,
    GradientShap,
    DeepLift,
    DeepLiftShap,
    InputXGradient,
    Saliency,
    NoiseTunnel,
    Lime,
    KernelShap
)
from captum.attr import visualization as viz

# Import model and dataset classes
from model.efficientNetV2 import MRIClassifier
from dataset import MRIDataset

class SaliencyMapGenerator:
    def __init__(self, model_path, parameters_path, device='cuda:0'):
        """
        Initialize the saliency map generator
        
        Args:
            model_path: Path to the trained model (.pth file)
            parameters_path: Path to parameters.json file
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load parameters
        with open(parameters_path, 'r') as f:
            self.params = json.load(f)
        
        # Load model
        self.model = MRIClassifier(dropout_rate=self.params['dropout_rate'])
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize attribution methods with model wrapper
        self.attribution_methods = {
            'integrated_gradients': IntegratedGradients(self.model_wrapper),
            'gradient_shap': GradientShap(self.model_wrapper),
            'deep_lift': DeepLift(self.model_wrapper),
            'input_x_gradient': InputXGradient(self.model_wrapper),
            'saliency': Saliency(self.model_wrapper),
            'noise_tunnel_ig': NoiseTunnel(IntegratedGradients(self.model_wrapper)),
        }
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def prepare_dataloader(self, csv_path, batch_size=1):
        """
        Prepare dataloader for validation dataset
        
        Args:
            csv_path: Path to validation CSV file
            batch_size: Batch size for dataloader
        """
        df = pd.read_csv(csv_path)
        dataset = MRIDataset(df, self.params, train=False)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 for debugging
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"Validation dataset loaded: {len(dataset)} samples")
        return dataloader, dataset
    
    def model_wrapper(self, input_tensor):
        """
        Wrapper function for the model to return only logits for attribution
        """
        logits, _ = self.model(input_tensor)
        return logits
    
    def compute_attributions(self, input_tensor, target_class, method_name='integrated_gradients'):
        """
        Compute attributions using specified method
        
        Args:
            input_tensor: Input tensor of shape [batch_size, channels, height, width]
            target_class: Target class index (not used for binary classification with single output)
            method_name: Name of attribution method to use
        """
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        
        method = self.attribution_methods[method_name]
        
        # For binary classification with single output unit, target should be None or 0
        target = None  # This will use the single output unit
        
        try:
            if method_name == 'gradient_shap':
                # Generate random baseline for GradientShap
                baseline = torch.randn_like(input_tensor) * 0.1
                attributions = method.attribute(
                    input_tensor,
                    baselines=baseline,
                    target=target,
                    n_samples=10
                )
            elif method_name == 'noise_tunnel_ig':
                attributions = method.attribute(
                    input_tensor,
                    target=target,
                    nt_type='smoothgrad',
                    nt_samples=10,
                    stdevs=0.1
                )
            elif method_name in ['integrated_gradients', 'deep_lift']:
                # Use zero baseline for these methods
                baseline = torch.zeros_like(input_tensor)
                attributions = method.attribute(
                    input_tensor,
                    baselines=baseline,
                    target=target
                )
            else:
                # For saliency and input_x_gradient
                attributions = method.attribute(
                    input_tensor,
                    target=target
                )
            
            return attributions.detach().cpu()
        
        except Exception as e:
            print(f"Error computing {method_name}: {str(e)}")
            return None
    
    def create_saliency_visualization(self, original_image, attributions, slice_idx, 
                                    method_name, save_path, prediction, true_label):
        """
        Create and save saliency map visualization
        
        Args:
            original_image: Original image slice [H, W]
            attributions: Attribution values [H, W]
            slice_idx: Index of the slice
            method_name: Name of attribution method
            save_path: Path to save the visualization
            prediction: Model prediction
            true_label: Ground truth label
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        im1 = axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title(f'Original Slice {slice_idx}')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Attribution map
        im2 = axes[1].imshow(attributions, cmap='RdBu_r', vmin=-np.abs(attributions).max(), vmax=np.abs(attributions).max())
        axes[1].set_title(f'{method_name.replace("_", " ").title()}')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay
        axes[2].imshow(original_image, cmap='gray', alpha=0.7)
        im3 = axes[2].imshow(np.abs(attributions), cmap='hot', alpha=0.5)
        axes[2].set_title('Overlay (Absolute Attribution)')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        
        # Add prediction info
        pred_label = "TLE" if prediction > 0.5 else "HC"
        true_label_str = "TLE" if true_label == 1 else "HC"
        fig.suptitle(f'Prediction: {pred_label} (conf: {prediction:.3f}) | True: {true_label_str}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_summary_statistics(self, attributions, method_name):
        """
        Generate summary statistics for attributions
        
        Args:
            attributions: Attribution tensor [channels, height, width]
            method_name: Name of attribution method
        """
        stats = {
            'method': method_name,
            'shape': list(attributions.shape),
            'mean': float(attributions.mean()),
            'std': float(attributions.std()),
            'min': float(attributions.min()),
            'max': float(attributions.max()),
            'abs_mean': float(torch.abs(attributions).mean()),
            'positive_ratio': float((attributions > 0).float().mean()),
            'negative_ratio': float((attributions < 0).float().mean())
        }
        return stats
    
    def generate_saliency_maps(self, csv_path, output_dir, methods=None, 
                             max_samples=None, key_slices=None):
        """
        Generate saliency maps for validation dataset
        
        Args:
            csv_path: Path to validation CSV file
            output_dir: Directory to save saliency maps
            methods: List of attribution methods to use
            max_samples: Maximum number of samples to process
            key_slices: List of slice indices to focus on (e.g., [40, 50, 60, 70, 80])
        """
        if methods is None:
            methods = ['integrated_gradients', 'saliency', 'gradient_shap']
        
        if key_slices is None:
            key_slices = [30, 40, 50, 60, 70, 80, 90]  # Focus on middle slices
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        
        for method in methods:
            os.makedirs(os.path.join(output_dir, method), exist_ok=True)
        
        # Prepare dataloader
        dataloader, dataset = self.prepare_dataloader(csv_path, batch_size=1)
        
        all_stats = []
        sample_count = 0
        
        print(f"Generating saliency maps using methods: {methods}")
        print(f"Focusing on slices: {key_slices}")
        
        for batch_idx, (images, labels, img_paths) in enumerate(tqdm(dataloader, desc="Processing samples")):
            if max_samples and sample_count >= max_samples:
                break
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                logits, _ = self.model(images)
                probabilities = torch.sigmoid(logits)
                predictions = probabilities.cpu().numpy()
            
            # Process each sample in batch (typically batch_size=1)
            for sample_idx in range(images.shape[0]):
                sample_image = images[sample_idx:sample_idx+1]  # Keep batch dimension
                sample_label = labels[sample_idx].item()
                sample_pred = predictions[sample_idx, 0]
                img_path = img_paths[sample_idx]
                
                # Extract subject ID from path
                subject_id = os.path.basename(img_path).split('_')[0].replace('mwp1', '')
                
                print(f"\nProcessing sample {sample_count + 1}: {subject_id}")
                print(f"True label: {'TLE' if sample_label == 1 else 'HC'}, "
                      f"Prediction: {sample_pred:.3f} ({'TLE' if sample_pred > 0.5 else 'HC'})")
                
                # Generate attributions for each method
                for method_name in methods:
                    print(f"  Computing {method_name}...")
                    
                    attributions = self.compute_attributions(
                        sample_image, 
                        target_class=sample_label, 
                        method_name=method_name
                    )
                    
                    if attributions is None:
                        continue
                    
                    # Generate statistics
                    stats = self.generate_summary_statistics(attributions[0], method_name)
                    stats.update({
                        'subject_id': subject_id,
                        'true_label': sample_label,
                        'prediction': sample_pred,
                        'sample_idx': sample_count
                    })
                    all_stats.append(stats)
                    
                    # Create visualizations for key slices
                    method_dir = os.path.join(output_dir, method_name)
                    subject_dir = os.path.join(method_dir, subject_id)
                    os.makedirs(subject_dir, exist_ok=True)
                    
                    # Save summary visualization with multiple slices
                    fig, axes = plt.subplots(2, len(key_slices), figsize=(4*len(key_slices), 8))
                    if len(key_slices) == 1:
                        axes = axes.reshape(-1, 1)
                    
                    for i, slice_idx in enumerate(key_slices):
                        if slice_idx < sample_image.shape[1]:  # Check if slice exists
                            # Original image
                            original_slice = sample_image[0, slice_idx].detach().cpu().numpy()
                            attribution_slice = attributions[0, slice_idx].numpy()
                            
                            # Top row: original images
                            axes[0, i].imshow(original_slice, cmap='gray')
                            axes[0, i].set_title(f'Slice {slice_idx}')
                            axes[0, i].axis('off')
                            
                            # Bottom row: attribution maps
                            im = axes[1, i].imshow(attribution_slice, cmap='RdBu_r', 
                                                 vmin=-np.abs(attribution_slice).max(), 
                                                 vmax=np.abs(attribution_slice).max())
                            axes[1, i].set_title(f'Attribution {slice_idx}')
                            axes[1, i].axis('off')
                            
                            # Add colorbar to last subplot
                            if i == len(key_slices) - 1:
                                plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
                    
                    # Add overall title
                    pred_label = "TLE" if sample_pred > 0.5 else "HC"
                    true_label_str = "TLE" if sample_label == 1 else "HC"
                    fig.suptitle(f'{subject_id} - {method_name.replace("_", " ").title()}\n'
                               f'Prediction: {pred_label} (conf: {sample_pred:.3f}) | True: {true_label_str}', 
                               fontsize=14, fontweight='bold')
                    
                    plt.tight_layout()
                    summary_path = os.path.join(subject_dir, f'{subject_id}_{method_name}_summary.png')
                    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    # Save individual slice visualizations for key slices
                    for slice_idx in key_slices:
                        if slice_idx < sample_image.shape[1]:
                            original_slice = sample_image[0, slice_idx].detach().cpu().numpy()
                            attribution_slice = attributions[0, slice_idx].numpy()
                            
                            slice_path = os.path.join(subject_dir, 
                                                    f'{subject_id}_{method_name}_slice_{slice_idx}.png')
                            
                            self.create_saliency_visualization(
                                original_slice, attribution_slice, slice_idx,
                                method_name, slice_path, sample_pred, sample_label
                            )
                    
                    # Save raw attribution data
                    attr_data_path = os.path.join(subject_dir, f'{subject_id}_{method_name}_attributions.npy')
                    np.save(attr_data_path, attributions[0].numpy())
                
                sample_count += 1
        
        # Save statistics
        stats_df = pd.DataFrame(all_stats)
        stats_path = os.path.join(output_dir, 'attribution_statistics.csv')
        stats_df.to_csv(stats_path, index=False)
        
        # Create summary plots
        self.create_summary_plots(stats_df, output_dir)
        
        print(f"\nSaliency map generation completed!")
        print(f"Processed {sample_count} samples")
        print(f"Results saved to: {output_dir}")
        print(f"Statistics saved to: {stats_path}")
    
    def create_summary_plots(self, stats_df, output_dir):
        """
        Create summary plots of attribution statistics
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Mean attribution by method
        sns.boxplot(data=stats_df, x='method', y='abs_mean', ax=axes[0, 0])
        axes[0, 0].set_title('Absolute Mean Attribution by Method')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Standard deviation by method
        sns.boxplot(data=stats_df, x='method', y='std', ax=axes[0, 1])
        axes[0, 1].set_title('Attribution Standard Deviation by Method')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Positive/Negative ratio
        sns.boxplot(data=stats_df, x='method', y='positive_ratio', ax=axes[0, 2])
        axes[0, 2].set_title('Positive Attribution Ratio by Method')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Plot 4: Attribution vs prediction confidence
        for method in stats_df['method'].unique():
            method_data = stats_df[stats_df['method'] == method]
            axes[1, 0].scatter(method_data['prediction'], method_data['abs_mean'], 
                             label=method, alpha=0.6)
        axes[1, 0].set_xlabel('Prediction Confidence')
        axes[1, 0].set_ylabel('Absolute Mean Attribution')
        axes[1, 0].set_title('Attribution vs Prediction Confidence')
        axes[1, 0].legend()
        
        # Plot 5: Attribution by true label
        sns.boxplot(data=stats_df, x='true_label', y='abs_mean', hue='method', ax=axes[1, 1])
        axes[1, 1].set_title('Attribution by True Label')
        axes[1, 1].set_xlabel('True Label (0=HC, 1=TLE)')
        
        # Plot 6: Method comparison heatmap
        method_means = stats_df.groupby('method')['abs_mean'].mean()
        method_stds = stats_df.groupby('method')['std'].mean()
        comparison_data = pd.DataFrame({
            'Mean Attribution': method_means,
            'Std Attribution': method_stds
        }).T
        
        sns.heatmap(comparison_data, annot=True, fmt='.4f', ax=axes[1, 2])
        axes[1, 2].set_title('Method Comparison')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'attribution_summary.png'), dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate saliency maps for MRI classification')
    parser.add_argument('--model_path', 
                       default='/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/classifier/new_runs/runs_syn/syn_2723/fold_1/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--parameters_path',
                       default='/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/classifier/new_runs/runs_syn/syn_2723/parameters.json',
                       help='Path to parameters.json file')
    parser.add_argument('--csv_path',
                       default='/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/data_csvs/val.csv',
                       help='Path to validation CSV file')
    parser.add_argument('--output_dir',
                       default='/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/classifier/new_runs/runs_syn/syn_2723/fold_1/saliency_maps_detailed',
                       help='Output directory for saliency maps')
    parser.add_argument('--methods', nargs='+',
                       default=['integrated_gradients', 'saliency', 'gradient_shap', 'input_x_gradient'],
                       help='Attribution methods to use')
    parser.add_argument('--max_samples', type=int, default=50,
                       help='Maximum number of samples to process (None for all)')
    parser.add_argument('--key_slices', nargs='+', type=int,
                       default=[30, 40, 50, 60, 70, 80, 90],
                       help='Key slice indices to visualize')
    parser.add_argument('--device', default='cuda:0',
                       help='Device to use for computation')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MRI Saliency Map Generator")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Parameters: {args.parameters_path}")
    print(f"Validation data: {args.csv_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Methods: {args.methods}")
    print(f"Max samples: {args.max_samples}")
    print(f"Key slices: {args.key_slices}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Initialize generator
    generator = SaliencyMapGenerator(
        model_path=args.model_path,
        parameters_path=args.parameters_path,
        device=args.device
    )
    
    # Generate saliency maps
    generator.generate_saliency_maps(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        methods=args.methods,
        max_samples=args.max_samples,
        key_slices=args.key_slices
    )


if __name__ == "__main__":
    main() 