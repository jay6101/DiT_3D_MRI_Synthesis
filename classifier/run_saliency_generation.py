#!/usr/bin/env python3
"""
Simple runner script for saliency map generation
"""

import subprocess
import sys
import os

def main():
    # Default parameters
    script_path = "/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/classifier/generate_saliency_maps.py"
    
    # Command to run
    cmd = [
        sys.executable, script_path,
        "--model_path", "/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/classifier/new_runs/runs_2723/real_2723_syn_50/fold_1/best_model.pth",
        "--parameters_path", "/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/classifier/new_runs/runs_2723/real_2723_syn_50/parameters.json",
        "--csv_path", "/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/data_csvs/val.csv",
        "--output_dir", "/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/classifier/new_runs/runs_2723/real_2723_syn_50/fold_1/saliency_maps_detailed",
        "--methods", "integrated_gradients", "saliency", "gradient_shap", "input_x_gradient",
        "--max_samples", "477",  # Start with 20 samples to test
        "--key_slices", "40", "50", "60", "70", "80",
        "--device", "cuda:1"
    ]
    
    print("Running saliency map generation...")
    print("Command:", " ".join(cmd))
    print("-" * 60)
    
    # Execute the command
    try:
        subprocess.run(cmd, check=True)
        print("-" * 60)
        print("Saliency map generation completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error running saliency generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 