#!/usr/bin/env python3
"""
Generate plots of FID scores vs slice position for all three axes.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_fid_data():
    """Load FID data for all three axes."""
    data = {}
    for axis in [0, 1, 2]:
        npz_data = np.load(f'axis{axis}_fids_by_position.npz')
        data[f'axis{axis}'] = {
            'positions': npz_data['positions'],
            'fids': npz_data['values']
        }
    return data

def create_fid_plots():
    """Create FID vs position plots for all three axes."""
    
    # Load data
    data = load_fid_data()
    
    # Calculate global y-range for consistent scaling
    all_fids = []
    for axis_data in data.values():
        all_fids.extend(axis_data['fids'])
    
    y_min = min(all_fids) * 0.95  # Add some padding
    y_max = max(all_fids) * 1.05
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('FID Scores vs Slice Position Across All Axes', fontsize=16, fontweight='bold')
    
    axis_names = ['Sagittal (X)', 'Coronal (Y)', 'Axial (Z)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    for i, (axis_key, axis_data) in enumerate(data.items()):
        ax = axes[i]
        
        positions = axis_data['positions']
        fids = axis_data['fids']
        
        # Plot with nice styling
        ax.plot(positions, fids, 'o-', color=colors[i], linewidth=2, 
                markersize=4, alpha=0.8, label=f'Axis {i}')
        
        # Customize subplot
        ax.set_title(f'{axis_names[i]} (Axis {i})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Slice Position', fontsize=12)
        ax.set_ylabel('FID Score', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(y_min, y_max)
        
        # Add statistics text box
        mean_fid = np.mean(fids)
        std_fid = np.std(fids)
        min_fid = np.min(fids)
        max_fid = np.max(fids)
        
        stats_text = f'Mean: {mean_fid:.2f}\nStd: {std_fid:.2f}\nMin: {min_fid:.2f}\nMax: {max_fid:.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = 'fid_by_position_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_path}")
    
    return fig

def create_overlay_plot():
    """Create an overlay plot with all three axes on the same graph."""
    
    # Load data
    data = load_fid_data()
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    axis_names = ['Sagittal (X)', 'Coronal (Y)', 'Axial (Z)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    
    for i, (axis_key, axis_data) in enumerate(data.items()):
        positions = axis_data['positions']
        fids = axis_data['fids']
        
        ax.plot(positions, fids, marker=markers[i], color=colors[i], 
                linewidth=2, markersize=5, alpha=0.8, 
                label=f'{axis_names[i]} (Axis {i})')
    
    ax.set_title('FID Scores vs Slice Position - All Axes Comparison', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Slice Position', fontsize=12)
    ax.set_ylabel('FID Score', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save the overlay plot
    output_path = 'fid_by_position_overlay.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Overlay plot saved as: {output_path}")
    
    return fig

def print_summary_stats():
    """Print summary statistics for all axes."""
    data = load_fid_data()
    
    print("\n" + "="*60)
    print("FID SCORE SUMMARY STATISTICS")
    print("="*60)
    
    axis_names = ['Sagittal (X)', 'Coronal (Y)', 'Axial (Z)']
    
    for i, (axis_key, axis_data) in enumerate(data.items()):
        fids = axis_data['fids']
        positions = axis_data['positions']
        
        print(f"\n{axis_names[i]} (Axis {i}):")
        print(f"  Number of slices: {len(fids)}")
        print(f"  Position range: {positions.min():.0f} - {positions.max():.0f}")
        print(f"  Mean FID: {np.mean(fids):.3f}")
        print(f"  Std FID: {np.std(fids):.3f}")
        print(f"  Min FID: {np.min(fids):.3f} (position {positions[np.argmin(fids)]:.0f})")
        print(f"  Max FID: {np.max(fids):.3f} (position {positions[np.argmax(fids)]:.0f})")

if __name__ == "__main__":
    # Create the plots
    print("Generating FID vs position plots...")
    
    # Create side-by-side comparison
    fig1 = create_fid_plots()
    
    # Create overlay plot
    fig2 = create_overlay_plot()
    
    # Print summary statistics
    print_summary_stats()
    
    print("\nPlots generated successfully!")
    plt.show() 