import os
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from scipy import linalg
import pandas as pd
import nibabel as nib
from random import shuffle

def load_inception_model(device):
    """
    Load a pretrained Inception v3 model from torchvision, 
    configured to return features from the last pooling layer (2048-d).
    """
    inception = models.inception_v3(pretrained=True, transform_input=False)
    inception.eval()
    inception.to(device)

    # We'll register a forward hook to grab features from 'avgpool'
    outputs = []
    feature_dimension = 2048
    feature_layer = 'avgpool'

    def hook(module, input, output):
        # output should have shape [N, 2048, 1, 1] after the average pool
        outputs.append(output.flatten(start_dim=1))

    handle = getattr(inception, feature_layer).register_forward_hook(hook)
    return inception, outputs, handle, feature_dimension


def get_inception_features(inception, outputs, images, device):
    """
    Pass a batch of images (N, 3, H, W) through Inception; returns numpy array of features (N, 2048).
    """
    # Clear any stale outputs from a previous batch
    outputs.clear()

    with torch.no_grad():
        inception(images.to(device))

    # outputs[0] should now be shape (N, 2048)
    feat = outputs[0].cpu().numpy()
    outputs.clear()  # Clear for next usage
    return feat


def preprocess_slice_for_inception(slice_2d, target_height=112, target_width=112, minmax_for_synthetic=False):
    """
    Convert a 2D numpy array (e.g., shape [H, W]) into a (3, target_size, target_size) tensor for Inception:
      1) If minmax_for_synthetic=True, scale slice by (slice - min)/(max - min) then *255.
         Otherwise assume the slice is already in a suitable range (e.g., after standardizing the entire volume).
      2) Convert to PIL and resize to (target_size, target_size).
      3) Replicate 1 channel → 3 channels.
      4) Normalize by ImageNet means & stds.
    """
    # For synthetic data, you may want per-slice min-max. Adjust as needed.
    # if minmax_for_synthetic:
    #     smin, smax = slice_2d.min(), slice_2d.max()
    #     if smax - smin < 1e-10:
    #         # Degenerate slice
    #         slice_2d = np.zeros_like(slice_2d)
    #     else:
    #         slice_2d = (slice_2d - smin) / (smax - smin)
    #     slice_2d = (slice_2d * 255).astype(np.uint8)
    # else:
    #     # For real data, if the entire volume was already mean-std normalized, 
    #     # it might be okay to do a quick shift to a 0..255 range per slice 
    #     # or just cast to float. The exact strategy depends on your data.
    #     # Here, let's just do a min-max on the slice itself for demonstration.
    #     smin, smax = slice_2d.min(), slice_2d.max()
    #     if smax - smin < 1e-10:
    #         slice_2d = np.zeros_like(slice_2d)
    #     else:
    #         slice_2d = (slice_2d - smin) / (smax - smin)
    #     slice_2d = (slice_2d * 255).astype(np.uint8)

    pil_img = Image.fromarray(slice_2d)
    transform_steps = transforms.Compose([
        transforms.Resize((target_height, target_width)),
        transforms.Resize((299,299)),
        transforms.ToTensor(),  # shape (1, H, W)
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # replicate grayscale -> 3 channels
        # transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # ),
    ])
    return transform_steps(pil_img)


def gather_synthetic_features_by_slice(
    pkl_folder, pkl_folder_2, device, inception, outputs, batch_size=16, max_files=100
):
    """
    Gather features from synthetic data stored in .pkl files, organized by slice position.
    Returns a dictionary with 3 keys (axis0, axis1, axis2), each containing lists of feature arrays
    organized by slice position.
    """
    pkl_files = sorted(glob(os.path.join(pkl_folder, '*.pkl')))
    pkl_files_2 = sorted(glob(os.path.join(pkl_folder_2, '*.pkl')))
    pkl_files.extend(pkl_files_2)
    shuffle(pkl_files)
    
    pkl_files = pkl_files[:max_files]
    print(f"Found {len(pkl_files)} synthetic .pkl files in {pkl_folder}")

    # Dictionary to store features by axis and slice position
    features_by_position = {
        'axis0': {},  # Will contain {slice_idx: [features...]}
        'axis1': {},  # Will contain {slice_idx: [features...]}
        'axis2': {}   # Will contain {slice_idx: [features...]}
    }
    
    # Buffers for batching by axis and slice position
    buffer_by_position = {
        'axis0': {},  # Will contain {slice_idx: [images...]}
        'axis1': {},  # Will contain {slice_idx: [images...]}
        'axis2': {}   # Will contain {slice_idx: [images...]}
    }

    def flush_buffer_for_axis_position(axis_name, position):
        buffer = buffer_by_position[axis_name].get(position, [])
        if not buffer:
            return
            
        batch_tensor = torch.stack(buffer, dim=0)
        feats = get_inception_features(inception, outputs, batch_tensor, device)
        
        if position not in features_by_position[axis_name]:
            features_by_position[axis_name][position] = []
        
        features_by_position[axis_name][position].append(feats)
        buffer_by_position[axis_name][position] = []

    def flush_all_buffers():
        for axis_name in ['axis0', 'axis1', 'axis2']:
            for position in list(buffer_by_position[axis_name].keys()):
                flush_buffer_for_axis_position(axis_name, position)

    for pkl_f in tqdm(pkl_files, desc="Extracting synthetic features"):
        with open(pkl_f, 'rb') as f:
            data_dict = pickle.load(f)

        # The 3D volume is stored under the key "image"
        volume = data_dict["image"]  # shape [D,H,W]
        volume = np.asarray(volume, dtype=np.float32)
        D, H, W = volume.shape

        # axis 0: (sagittal)
        for i in range(D):
            slice_2d = volume[i, :, :]
            img_tensor = preprocess_slice_for_inception(slice_2d, target_height=136, target_width=112, minmax_for_synthetic=False)
            
            if i not in buffer_by_position['axis0']:
                buffer_by_position['axis0'][i] = []
                
            buffer_by_position['axis0'][i].append(img_tensor)
            
            if len(buffer_by_position['axis0'][i]) >= batch_size:
                flush_buffer_for_axis_position('axis0', i)

        # axis 1: (coronal)
        for i in range(H):
            slice_2d = volume[:, i, :]
            img_tensor = preprocess_slice_for_inception(slice_2d, target_height=112, target_width=112, minmax_for_synthetic=False)
            
            if i not in buffer_by_position['axis1']:
                buffer_by_position['axis1'][i] = []
                
            buffer_by_position['axis1'][i].append(img_tensor)
            
            if len(buffer_by_position['axis1'][i]) >= batch_size:
                flush_buffer_for_axis_position('axis1', i)

        # axis 2: (axial)
        for i in range(W):
            slice_2d = volume[:, :, i]
            img_tensor = preprocess_slice_for_inception(slice_2d, target_height=112, target_width=136, minmax_for_synthetic=False)
            
            if i not in buffer_by_position['axis2']:
                buffer_by_position['axis2'][i] = []
                
            buffer_by_position['axis2'][i].append(img_tensor)
            
            if len(buffer_by_position['axis2'][i]) >= batch_size:
                flush_buffer_for_axis_position('axis2', i)

    # Flush any remaining buffers
    flush_all_buffers()
    
    # Concatenate features for each position
    for axis_name in ['axis0', 'axis1', 'axis2']:
        for position in features_by_position[axis_name]:
            if features_by_position[axis_name][position]:
                features_by_position[axis_name][position] = np.concatenate(
                    features_by_position[axis_name][position], axis=0)
    
    return features_by_position


def gather_real_features_by_slice(
    csv_file, device, inception, outputs, batch_size=16, max_files=100
):
    """
    Gather features from real data specified in a CSV file, organized by slice position.
    Returns a dictionary with 3 keys (axis0, axis1, axis2), each containing lists of feature arrays
    organized by slice position.
    """
    df = pd.read_csv(csv_file)
    #df = df[~df['file'].str.contains("/space/mcdonald-syn01/1/BIDS//enigma_conglom//derivatives//cat12_copy/sub-upenn", na=False)]
    df = df.loc[df["HC_vs_LTLE_vs_RTLE_string"].isin(["HC","left","right"])]
    print(len(df))
    df = df.sample(max_files, random_state=42)
    nii_paths = df['file'].tolist()

    print(f"Found {len(nii_paths)} real .nii files in CSV: {csv_file}")

    # Dictionary to store features by axis and slice position
    features_by_position = {
        'axis0': {},  # Will contain {slice_idx: [features...]}
        'axis1': {},  # Will contain {slice_idx: [features...]}
        'axis2': {}   # Will contain {slice_idx: [features...]}
    }
    
    # Buffers for batching by axis and slice position
    buffer_by_position = {
        'axis0': {},  # Will contain {slice_idx: [images...]}
        'axis1': {},  # Will contain {slice_idx: [images...]}
        'axis2': {}   # Will contain {slice_idx: [images...]}
    }

    def flush_buffer_for_axis_position(axis_name, position):
        buffer = buffer_by_position[axis_name].get(position, [])
        if not buffer:
            return
            
        batch_tensor = torch.stack(buffer, dim=0)
        feats = get_inception_features(inception, outputs, batch_tensor, device)
        
        if position not in features_by_position[axis_name]:
            features_by_position[axis_name][position] = []
        
        features_by_position[axis_name][position].append(feats)
        buffer_by_position[axis_name][position] = []

    def flush_all_buffers():
        for axis_name in ['axis0', 'axis1', 'axis2']:
            for position in list(buffer_by_position[axis_name].keys()):
                flush_buffer_for_axis_position(axis_name, position)

    for nii_path in tqdm(nii_paths, desc="Extracting real features"):
        if not os.path.exists(nii_path):
            print(f"File not found: {nii_path}")
            continue
            
        # Load NIFTI
        img_nib = nib.load(nii_path)
        volume = img_nib.get_fdata()  # shape [D,H,W]
        #volume = np.asarray(volume, dtype=np.float32)

        # Subtract mean and divide by std for entire 3D volume
        # mean_val, std_val = volume.mean(), volume.std()
        # std_val = std_val if std_val > 1e-12 else 1.0
        volume = (volume - np.mean(volume)) / np.std(volume)

        volume = volume[:,1:137,:]
        
        # Resize each slice from 113x113 to 112x112 using PyTorch transforms
        current_D, current_H, current_W = volume.shape
        resized_slices = []
        
        resize_transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor()
        ])
        
        # Resize slices along axis 1 (coronal slices)
        for h_idx in range(current_H):
            slice_2d = Image.fromarray(volume[:, h_idx, :])  # Shape: (D, W) = (113, 113)
            slice_tensor = resize_transform(slice_2d)
            resized_slices.append(slice_tensor.numpy()[0])
            
        
        # Reconstruct volume with new dimensions: (112, 136, 112)
        volume = np.stack(resized_slices, axis=1)  # Stack along H dimension
        
        D, H, W = volume.shape

        # axis 0: (sagittal)
        for i in range(D):
            slice_2d = volume[i, :, :]
            img_tensor = preprocess_slice_for_inception(slice_2d, target_height=136, target_width=112, minmax_for_synthetic=False)
            
            if i not in buffer_by_position['axis0']:
                buffer_by_position['axis0'][i] = []
                
            buffer_by_position['axis0'][i].append(img_tensor)
            
            if len(buffer_by_position['axis0'][i]) >= batch_size:
                flush_buffer_for_axis_position('axis0', i)

        # axis 1: (coronal)
        for i in range(H):
            slice_2d = volume[:, i, :]
            img_tensor = preprocess_slice_for_inception(slice_2d, target_height=112, target_width=112, minmax_for_synthetic=False)
            
            if i not in buffer_by_position['axis1']:
                buffer_by_position['axis1'][i] = []
                
            buffer_by_position['axis1'][i].append(img_tensor)
            
            if len(buffer_by_position['axis1'][i]) >= batch_size:
                flush_buffer_for_axis_position('axis1', i)

        # axis 2: (axial)
        for i in range(W):
            slice_2d = volume[:, :, i]
            img_tensor = preprocess_slice_for_inception(slice_2d, target_height=112, target_width=136, minmax_for_synthetic=False)
            
            if i not in buffer_by_position['axis2']:
                buffer_by_position['axis2'][i] = []
                
            buffer_by_position['axis2'][i].append(img_tensor)
            
            if len(buffer_by_position['axis2'][i]) >= batch_size:
                flush_buffer_for_axis_position('axis2', i)

    # Flush any remaining buffers
    flush_all_buffers()
    
    # Concatenate features for each position
    for axis_name in ['axis0', 'axis1', 'axis2']:
        for position in features_by_position[axis_name]:
            if features_by_position[axis_name][position]:
                features_by_position[axis_name][position] = np.concatenate(
                    features_by_position[axis_name][position], axis=0)
    
    return features_by_position


def calculate_activation_statistics(features):
    """Compute mean (D,) and covariance (D,D) of a set of features."""
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Compute Fréchet Distance between two Gaussians defined by (mu1, sigma1) and (mu2, sigma2).
    """
    diff = mu1 - mu2

    # Add a bit of eps to the diagonal to avoid singularities
    covmean, _ = linalg.sqrtm(
        (sigma1 + np.eye(sigma1.shape[0]) * eps)
         .dot(sigma2 + np.eye(sigma2.shape[0]) * eps),
        disp=False
    )
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError("Imaginary component in covmean")
        covmean = covmean.real

    return (diff.dot(diff) +
            np.trace(sigma1) +
            np.trace(sigma2) -
            2 * np.trace(covmean))


def calculate_per_slice_fid(
    config
):
    """
    Calculate FID for each slice position along all three axes.
    
    Args:
        config: Dictionary containing parameters:
            - synthetic_folder: Path to folder containing synthetic .pkl files
            - real_csv: CSV file containing paths to real .nii images
            - device: Device to use (e.g. "cuda:0" or "cpu")
            - batch_size: Batch size for Inception inference
            - max_files: Maximum number of files to process
            
    Returns:
        Dictionary containing FID arrays for each axis:
            - axis0_fids: Array of FIDs for each slice along axis 0 (sagittal)
            - axis1_fids: Array of FIDs for each slice along axis 1 (coronal)
            - axis2_fids: Array of FIDs for each slice along axis 2 (axial)
    """
    # Set default parameters
    synthetic_folder = config.get('synthetic_folder')
    synthetic_folder_2 = config.get('synthetic_folder_2')
    real_csv = config.get('real_csv')
    device = config.get('device', 'cuda:0')
    batch_size = config.get('batch_size', 16)
    max_files = config.get('max_files', 100)
    
    # Load inception model
    inception, outputs, handle, feat_dim = load_inception_model(device)

    # Extract features for SYNTHETIC data
    print("Extracting features for SYNTHETIC volumes...")
    synth_features_by_position = gather_synthetic_features_by_slice(
        pkl_folder=synthetic_folder,
        pkl_folder_2=synthetic_folder_2,    
        device=device,
        inception=inception,
        outputs=outputs,
        batch_size=batch_size,
        max_files=max_files
    )

    # Extract features for REAL data
    print("Extracting features for REAL volumes...")
    real_features_by_position = gather_real_features_by_slice(
        csv_file=real_csv,
        device=device,
        inception=inception,
        outputs=outputs,
        batch_size=batch_size,
        max_files=max_files
    )

    # Done with the hook
    handle.remove()

    # Calculate FID for each slice position
    axis0_fids = {}
    axis1_fids = {}
    axis2_fids = {}

    # Process axis 0 (sagittal)
    for position in sorted(set(synth_features_by_position['axis0'].keys()) & 
                         set(real_features_by_position['axis0'].keys())):
        synth_feat = synth_features_by_position['axis0'][position]
        real_feat = real_features_by_position['axis0'][position]
        
        if len(synth_feat) < 2 or len(real_feat) < 2:
            print(f"Skipping axis0 position {position}: Not enough samples")
            continue
            
        mu_synth, sigma_synth = calculate_activation_statistics(synth_feat)
        mu_real, sigma_real = calculate_activation_statistics(real_feat)
        
        try:
            fid_value = calculate_frechet_distance(mu_synth, sigma_synth, mu_real, sigma_real)
            axis0_fids[position] = fid_value
            print(f"Axis 0, Position {position}: FID = {fid_value:.4f}")
        except Exception as e:
            print(f"Error calculating FID for axis0 position {position}: {e}")

    # Process axis 1 (coronal)
    for position in sorted(set(synth_features_by_position['axis1'].keys()) & 
                         set(real_features_by_position['axis1'].keys())):
        synth_feat = synth_features_by_position['axis1'][position]
        real_feat = real_features_by_position['axis1'][position]
        
        if len(synth_feat) < 2 or len(real_feat) < 2:
            print(f"Skipping axis1 position {position}: Not enough samples")
            continue
            
        mu_synth, sigma_synth = calculate_activation_statistics(synth_feat)
        mu_real, sigma_real = calculate_activation_statistics(real_feat)
        
        try:
            fid_value = calculate_frechet_distance(mu_synth, sigma_synth, mu_real, sigma_real)
            axis1_fids[position] = fid_value
            print(f"Axis 1, Position {position}: FID = {fid_value:.4f}")
        except Exception as e:
            print(f"Error calculating FID for axis1 position {position}: {e}")

    # Process axis 2 (axial)
    for position in sorted(set(synth_features_by_position['axis2'].keys()) & 
                         set(real_features_by_position['axis2'].keys())):
        synth_feat = synth_features_by_position['axis2'][position]
        real_feat = real_features_by_position['axis2'][position]
        
        if len(synth_feat) < 2 or len(real_feat) < 2:
            print(f"Skipping axis2 position {position}: Not enough samples")
            continue
            
        mu_synth, sigma_synth = calculate_activation_statistics(synth_feat)
        mu_real, sigma_real = calculate_activation_statistics(real_feat)
        
        try:
            fid_value = calculate_frechet_distance(mu_synth, sigma_synth, mu_real, sigma_real)
            axis2_fids[position] = fid_value
            print(f"Axis 2, Position {position}: FID = {fid_value:.4f}")
        except Exception as e:
            print(f"Error calculating FID for axis2 position {position}: {e}")

    # Convert dictionaries to sorted arrays
    axis0_fid_array = np.array([axis0_fids[k] for k in sorted(axis0_fids.keys())])
    axis1_fid_array = np.array([axis1_fids[k] for k in sorted(axis1_fids.keys())])
    axis2_fid_array = np.array([axis2_fids[k] for k in sorted(axis2_fids.keys())])
    
    # Calculate average FIDs for each axis
    avg_fid_axis0 = np.mean(axis0_fid_array) if len(axis0_fid_array) > 0 else float('nan')
    avg_fid_axis1 = np.mean(axis1_fid_array) if len(axis1_fid_array) > 0 else float('nan')
    avg_fid_axis2 = np.mean(axis2_fid_array) if len(axis2_fid_array) > 0 else float('nan')
    
    print(f"Average FID for axis 0 (sagittal): {avg_fid_axis0:.4f}")
    print(f"Average FID for axis 1 (coronal): {avg_fid_axis1:.4f}")
    print(f"Average FID for axis 2 (axial): {avg_fid_axis2:.4f}")
    
    return {
        'axis0_fids': axis0_fid_array,
        'axis1_fids': axis1_fid_array,
        'axis2_fids': axis2_fid_array,
        'axis0_fids_by_position': axis0_fids,
        'axis1_fids_by_position': axis1_fids,
        'axis2_fids_by_position': axis2_fids,
        'avg_fid_axis0': avg_fid_axis0,
        'avg_fid_axis1': avg_fid_axis1,
        'avg_fid_axis2': avg_fid_axis2
    }


if __name__ == "__main__":
    # Example usage with configuration dictionary
    config = {
        'synthetic_folder': '/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/synthetic_data_pkls_HC',
        'synthetic_folder_2': '/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/synthetic_data_pkls_TLE',
        'real_csv': '/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/data_csvs/val.csv',
        'device': 'cuda:0',
        'batch_size': 16,
        'max_files': 477
    }
    
    results = calculate_per_slice_fid(config)
    
    # Save numpy arrays
    save_dir = os.path.join(os.path.dirname(config['synthetic_folder']), 'fid_results')
    os.makedirs(save_dir, exist_ok=True)
    
    # Save individual arrays
    np.save(os.path.join(save_dir, 'axis0_fids.npy'), results['axis0_fids'])
    np.save(os.path.join(save_dir, 'axis1_fids.npy'), results['axis1_fids'])
    np.save(os.path.join(save_dir, 'axis2_fids.npy'), results['axis2_fids'])
    
    # Save positions and values as numpy array
    for axis in ['axis0', 'axis1', 'axis2']:
        positions = np.array(sorted(results[f'{axis}_fids_by_position'].keys()))
        values = np.array([results[f'{axis}_fids_by_position'][k] for k in positions])
        np.savez(os.path.join(save_dir, f'{axis}_fids_by_position.npz'), positions=positions, values=values)
    
    # Save full results dictionary as pickle
    with open(os.path.join(save_dir, 'fid_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Saved FID results to {save_dir}")
    
    # Generate and save plots
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(sorted(results['axis0_fids_by_position'].keys()), 
             [results['axis0_fids_by_position'][k] for k in sorted(results['axis0_fids_by_position'].keys())])
    plt.title('Axis 0 (Sagittal) FIDs')
    plt.xlabel('Slice Position')
    plt.ylabel('FID Score')
    
    plt.subplot(1, 3, 2)
    plt.plot(sorted(results['axis1_fids_by_position'].keys()), 
             [results['axis1_fids_by_position'][k] for k in sorted(results['axis1_fids_by_position'].keys())])
    plt.title('Axis 1 (Coronal) FIDs')
    plt.xlabel('Slice Position')
    plt.ylabel('FID Score')
    
    plt.subplot(1, 3, 3)
    plt.plot(sorted(results['axis2_fids_by_position'].keys()), 
             [results['axis2_fids_by_position'][k] for k in sorted(results['axis2_fids_by_position'].keys())])
    plt.title('Axis 2 (Axial) FIDs')
    plt.xlabel('Slice Position')
    plt.ylabel('FID Score')
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'per_slice_fids.png')
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}") 