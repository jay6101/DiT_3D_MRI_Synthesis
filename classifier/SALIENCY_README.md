# MRI Saliency Map Generation

This repository contains scripts for generating saliency maps from trained MRI classification models using Captum attribution methods.

## Overview

The saliency map generator (`generate_saliency_maps.py`) loads a trained EfficientNetV2 model from the syn_2723 run and generates various types of attribution maps to visualize which regions of the MRI scans the model focuses on when making predictions.

## Features

- **Multiple Attribution Methods**: Supports Integrated Gradients, Saliency Maps, Gradient SHAP, Input×Gradient, and more
- **Comprehensive Visualization**: Creates individual slice visualizations and multi-slice summaries
- **Statistical Analysis**: Generates attribution statistics and comparison plots
- **Flexible Configuration**: Customizable slice selection, sample limits, and output directories
- **Raw Data Export**: Saves attribution arrays for further analysis

## Files

- `generate_saliency_maps.py` - Main saliency map generation script
- `run_saliency_generation.py` - Simple runner script with default parameters
- `saliency_requirements.txt` - Python package requirements
- `SALIENCY_README.md` - This documentation file

## Installation

1. Install required packages:
```bash
pip install -r saliency_requirements.txt
```

2. Ensure you have access to:
   - Trained model: `new_runs/runs_syn/syn_2723/fold_1/best_model.pth`
   - Model parameters: `new_runs/runs_syn/syn_2723/parameters.json`
   - Validation data: `/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/data_csvs/val.csv`

## Quick Start

### Option 1: Use the Runner Script (Recommended for Testing)
```bash
python run_saliency_generation.py
```

This will process 20 samples with default settings.

### Option 2: Direct Script Usage
```bash
python generate_saliency_maps.py \
    --model_path new_runs/runs_syn/syn_2723/fold_1/best_model.pth \
    --parameters_path new_runs/runs_syn/syn_2723/parameters.json \
    --csv_path /space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/data_csvs/val.csv \
    --output_dir new_runs/runs_syn/syn_2723/fold_1/saliency_maps_detailed \
    --max_samples 50 \
    --methods integrated_gradients saliency gradient_shap \
    --key_slices 40 50 60 70 80
```

## Configuration Options

### Attribution Methods
- `integrated_gradients` - Integrated Gradients (recommended)
- `saliency` - Simple gradient-based saliency
- `gradient_shap` - Gradient SHAP with random baselines
- `input_x_gradient` - Input × Gradient
- `deep_lift` - DeepLift attribution
- `noise_tunnel_ig` - Noise Tunnel with Integrated Gradients

### Key Parameters
- `--max_samples`: Number of samples to process (default: 50)
- `--key_slices`: Slice indices to visualize (default: 30,40,50,60,70,80,90)
- `--methods`: Attribution methods to use
- `--device`: GPU device (default: cuda:0)
- `--output_dir`: Directory for saving results

### Slice Selection
The script focuses on key slices in the middle range of the MRI volume where anatomical structures are most relevant for classification. Default slices (30-90) correspond to regions containing hippocampus and surrounding temporal lobe structures.

## Output Structure

```
output_dir/
├── integrated_gradients/
│   ├── subject-001/
│   │   ├── subject-001_integrated_gradients_summary.png
│   │   ├── subject-001_integrated_gradients_slice_40.png
│   │   ├── subject-001_integrated_gradients_slice_50.png
│   │   ├── ...
│   │   └── subject-001_integrated_gradients_attributions.npy
│   └── ...
├── saliency/
├── gradient_shap/
├── attribution_statistics.csv
└── attribution_summary.png
```

### Output Files
- **Summary Images**: Multi-slice overview for each subject and method
- **Individual Slice Images**: Detailed visualization for each key slice
- **Raw Attribution Data**: NumPy arrays with attribution values (.npy files)
- **Statistics**: CSV file with attribution statistics for all samples
- **Summary Plots**: Comparative analysis across methods and samples

## Understanding the Visualizations

### Individual Slice Visualizations
Each slice visualization contains three panels:
1. **Original Slice**: Grayscale MRI slice
2. **Attribution Map**: Red-blue heatmap showing positive/negative attributions
3. **Overlay**: Attribution magnitude overlaid on original image

### Summary Visualizations
- Top row: Original MRI slices across key slice indices
- Bottom row: Corresponding attribution maps
- Title shows prediction confidence and true label

### Statistical Analysis
The summary plots include:
- Attribution magnitude distributions by method
- Prediction confidence vs. attribution strength
- Comparison across different attribution methods
- Analysis by true label (HC vs TLE)

## Attribution Method Details

### Integrated Gradients
- **Best for**: Overall understanding of feature importance
- **Characteristics**: Satisfies axioms of attribution, stable results
- **Use case**: Primary method for clinical interpretation

### Saliency Maps
- **Best for**: Quick gradient-based insights
- **Characteristics**: Fast computation, may be noisy
- **Use case**: Initial exploration and comparison baseline

### Gradient SHAP
- **Best for**: Game theory-based attributions
- **Characteristics**: Uses random baselines, good for complex interactions
- **Use case**: Complementary analysis to Integrated Gradients

### Input × Gradient
- **Best for**: Simple multiplication-based attribution
- **Characteristics**: Fast, highlights regions with high gradient and input
- **Use case**: Lightweight alternative to more complex methods

## Clinical Interpretation

### Expected Patterns
- **TLE cases**: High attribution in hippocampal and temporal lobe regions
- **HC cases**: More distributed or different regional patterns
- **Slice-wise**: Peak attribution in middle slices (40-80) where hippocampus is visible

### Quality Indicators
- Consistent attribution patterns across similar cases
- Anatomically plausible attribution locations
- Correlation between prediction confidence and attribution strength

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Missing dependencies**: Install from requirements.txt
3. **File path errors**: Check absolute paths to model and data files
4. **Attribution computation errors**: Some methods may fail on specific inputs

### Performance Tips
- Start with small sample size (20-50) for testing
- Use fewer attribution methods initially
- Consider CPU execution for debugging
- Monitor GPU memory usage

## Advanced Usage

### Custom Slice Selection
For specific anatomical focus:
```bash
--key_slices 45 55 65 75  # Focus on hippocampal region
```

### Method Comparison
For research analysis:
```bash
--methods integrated_gradients gradient_shap deep_lift saliency input_x_gradient
```

### Full Dataset Processing
For production analysis:
```bash
--max_samples 0  # Process all samples (remove limit)
```

## Research Applications

- **Feature Importance Analysis**: Identify critical brain regions for classification
- **Model Interpretability**: Understand decision-making process
- **Clinical Validation**: Verify model focuses on relevant anatomy
- **Method Comparison**: Evaluate different attribution techniques
- **Quality Assessment**: Detect potential model biases or artifacts

## Citation

If you use this saliency generation tool in your research, please cite the relevant papers for:
- Captum library for attribution methods
- EfficientNetV2 for the underlying model architecture
- The specific attribution methods you use (Integrated Gradients, etc.)

## Support

For questions or issues:
1. Check the troubleshooting section above
2. Verify all file paths and dependencies
3. Test with smaller sample sizes first
4. Check CUDA/GPU availability if using GPU acceleration 