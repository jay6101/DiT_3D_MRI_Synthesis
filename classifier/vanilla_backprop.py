import torch
import numpy as np
import os
from model.efficientNetV2 import MRIClassifier
import pickle
from tqdm import tqdm

class VanillaBackprop:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        
    def save_gradient(self, grad):
        self.gradients = grad
        
    def generate_gradients(self, input_image, target_class=1):
        # Register hook for gradients
        input_image.requires_grad = True
        input_image.register_hook(self.save_gradient)
        
        # Forward pass
        model_output, _ = self.model(input_image)
        model_output = torch.sigmoid(model_output)
        
        # Backward pass
        self.model.zero_grad()
        model_output.backward()
        
        # Generate gradients
        gradients = self.gradients.cpu().numpy()
        return gradients

def compute_saliency_maps(model, test_loader, device, fold_dir):
    """
    Compute and save saliency maps for test samples
    
    Args:
        model: trained model
        test_loader: DataLoader for test data
        device: torch device
        fold_dir: directory to save results
    """
    vanilla_bp = VanillaBackprop(model)
    saliency_maps_dir = os.path.join(fold_dir, 'saliency_maps')
    os.makedirs(saliency_maps_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels, paths) in tqdm(enumerate(test_loader)):
            # For test data, images is not a list of augmentations
            images = images.to(device)
            batch_size = images.size(0)
            
            for i in range(batch_size):
                single_image = images[i:i+1]
                img_path = paths[i]
                true_label = labels[i].item()
                
                # Enable gradients for this specific computation
                with torch.enable_grad():
                    # Get model prediction
                    logits, _ = model(single_image)
                    pred_score = torch.sigmoid(logits).item()
                    pred_label = 1 if pred_score >= 0.5 else 0
                    
                    # Compute saliency map
                    saliency_map = vanilla_bp.generate_gradients(single_image)
                    
                    # Take absolute value and normalize
                    saliency_map = np.abs(saliency_map)
                    
                    # Create results dictionary
                    results = {
                        'original_path': img_path,
                        'predicted_score': pred_score,
                        'predicted_label': pred_label,
                        'true_label': true_label,
                        'saliency_map': saliency_map
                    }
                    
                    # Save results
                    # Use a filename based on the original image name
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    save_path = os.path.join(saliency_maps_dir, f'{base_name}_saliency.pkl')
                    
                    with open(save_path, 'wb') as f:
                        pickle.dump(results, f)

def generate_fold_saliency_maps(fold_dir, hyperparams):
    """
    Generate saliency maps for a specific fold
    
    Args:
        fold_dir: directory containing fold data
        hyperparams: hyperparameters dictionary
    """
    # Load the best model
    model = MRIClassifier(dropout_rate=hyperparams['dropout_rate']).to(hyperparams['device'])
    checkpoint = torch.load(os.path.join(fold_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create test dataset and loader
    from dataset import MRIDataset
    import pandas as pd
    from torch.utils.data import DataLoader
    
    test_df = pd.read_csv("/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/data_csvs/val.csv")
    test_dataset = MRIDataset(test_df, hyperparams, train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one image at a time for saliency maps
        shuffle=False,
        num_workers=hyperparams['num_workers'],
        pin_memory=True
    )
    
    # Compute and save saliency maps
    compute_saliency_maps(model, test_loader, hyperparams['device'], fold_dir)