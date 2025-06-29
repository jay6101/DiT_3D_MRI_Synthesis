#!/usr/bin/env python3
"""
Enhanced FID visualization with trend analysis and statistical insights.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter1d
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

def create_enhanced_comparison_plot():
    """Create enhanced side-by-side comparison with trend lines and statistics."""
    
    data = load_fid_data()
    
    # Calculate global y-range
    all_fids = []
    for axis_data in data.values():
        all_fids.extend(axis_data['fids'])
    y_min, y_max = min(all_fids) * 0.92, max(all_fids) * 1.08
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Main plots in top row
    axis_names = ['Sagittal (X)', 'Coronal (Y)', 'Axial (Z)']
    colors = ['#e74c3c', '#3498db', '#2ecc71']  # Red, Blue, Green
    
    for i, (axis_key, axis_data) in enumerate(data.items()):
        ax = axes[0, i]
        
        positions = axis_data['positions']
        fids = axis_data['fids']
        
        # Raw data points
        ax.scatter(positions, fids, color=colors[i], alpha=0.6, s=20, label='Raw FID scores')
        
        # Smoothed trend line
        smoothed_fids = gaussian_filter1d(fids, sigma=2)
        ax.plot(positions, smoothed_fids, color=colors[i], linewidth=3, alpha=0.8, label='Smoothed trend')
        
        # Rolling statistics
        window = max(5, len(fids) // 10)
        rolling_mean = np.convolve(fids, np.ones(window)/window, mode='same')
        rolling_std = np.array([np.std(fids[max(0, j-window//2):min(len(fids), j+window//2+1)]) 
                               for j in range(len(fids))])
        
        # Confidence interval
        ax.fill_between(positions, rolling_mean - rolling_std, rolling_mean + rolling_std, 
                       color=colors[i], alpha=0.2, label='±1 std')
        
        # Customize
        ax.set_title(f'{axis_names[i]} (Axis {i})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Slice Position', fontsize=12)
        ax.set_ylabel('FID Score', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(y_min, y_max)
        ax.legend(fontsize=10)
        
        # Mark extreme values
        min_idx, max_idx = np.argmin(fids), np.argmax(fids)
        ax.annotate(f'Min: {fids[min_idx]:.2f}', 
                   xy=(positions[min_idx], fids[min_idx]), 
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', color='green'))
        ax.annotate(f'Max: {fids[max_idx]:.2f}', 
                   xy=(positions[max_idx], fids[max_idx]), 
                   xytext=(-10, -20), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', color='red'))
    
    # Distribution plots in bottom row
    for i, (axis_key, axis_data) in enumerate(data.items()):
        ax = axes[1, i]
        
        fids = axis_data['fids']
        
        # Histogram with KDE
        ax.hist(fids, bins=20, density=True, alpha=0.7, color=colors[i], label='Histogram')
        
        # KDE curve
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(fids)
        x_range = np.linspace(fids.min(), fids.max(), 100)
        ax.plot(x_range, kde(x_range), color=colors[i], linewidth=2, label='KDE')
        
        # Normal distribution overlay
        mu, sigma = stats.norm.fit(fids)
        x_norm = np.linspace(fids.min(), fids.max(), 100)
        ax.plot(x_norm, stats.norm.pdf(x_norm, mu, sigma), 'k--', alpha=0.6, label='Normal fit')
        
        ax.set_title(f'Distribution - {axis_names[i]}', fontsize=12, fontweight='bold')
        ax.set_xlabel('FID Score', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Enhanced FID Analysis: Trends and Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = 'enhanced_fid_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Enhanced analysis saved as: {output_path}")
    
    return fig

def create_statistical_summary():
    """Create a comprehensive statistical summary plot."""
    
    data = load_fid_data()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    axis_names = ['Sagittal (X)', 'Coronal (Y)', 'Axial (Z)']
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    # Prepare data for summary statistics
    all_fids = [data[f'axis{i}']['fids'] for i in range(3)]
    all_positions = [data[f'axis{i}']['positions'] for i in range(3)]
    
    # 1. Box plot comparison
    box_data = [fids for fids in all_fids]
    bp = ax1.boxplot(box_data, labels=axis_names, patch_artist=True, notch=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_title('FID Score Distribution Comparison', fontweight='bold')
    ax1.set_ylabel('FID Score')
    ax1.grid(True, alpha=0.3)
    
    # 2. Violin plot
    parts = ax2.violinplot(box_data, positions=[1, 2, 3], showmeans=True, showmedians=True)
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(axis_names)
    ax2.set_title('FID Score Density Distributions', fontweight='bold')
    ax2.set_ylabel('FID Score')
    ax2.grid(True, alpha=0.3)
    
    # 3. Correlation between slice position and FID
    for i, (fids, positions, color, name) in enumerate(zip(all_fids, all_positions, colors, axis_names)):
        # Normalize positions to 0-1 for comparison
        norm_positions = (positions - positions.min()) / (positions.max() - positions.min())
        correlation, p_value = stats.pearsonr(norm_positions, fids)
        
        ax3.scatter(norm_positions, fids, alpha=0.6, color=color, label=f'{name} (r={correlation:.3f})')
        
        # Add trend line
        z = np.polyfit(norm_positions, fids, 1)
        p = np.poly1d(z)
        ax3.plot(norm_positions, p(norm_positions), color=color, linestyle='--', alpha=0.8)
    
    ax3.set_xlabel('Normalized Slice Position')
    ax3.set_ylabel('FID Score')
    ax3.set_title('FID vs Normalized Position Correlation', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary statistics table as a bar chart
    stats_data = []
    stats_labels = ['Mean', 'Std', 'Min', 'Max', 'Range']
    
    for fids in all_fids:
        stats_data.append([
            np.mean(fids),
            np.std(fids),
            np.min(fids),
            np.max(fids),
            np.max(fids) - np.min(fids)
        ])
    
    x = np.arange(len(stats_labels))
    width = 0.25
    
    for i, (stats_row, color, name) in enumerate(zip(stats_data, colors, axis_names)):
        ax4.bar(x + i*width, stats_row, width, label=name, color=color, alpha=0.7)
    
    ax4.set_xlabel('Statistics')
    ax4.set_ylabel('FID Score')
    ax4.set_title('Summary Statistics Comparison', fontweight='bold')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(stats_labels)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive FID Statistical Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = 'fid_statistical_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Statistical summary saved as: {output_path}")
    
    return fig

def print_detailed_analysis():
    """Print detailed statistical analysis."""
    data = load_fid_data()
    
    print("\n" + "="*80)
    print("DETAILED FID STATISTICAL ANALYSIS")
    print("="*80)
    
    axis_names = ['Sagittal (X)', 'Coronal (Y)', 'Axial (Z)']
    
    for i, (axis_key, axis_data) in enumerate(data.items()):
        fids = axis_data['fids']
        positions = axis_data['positions']
        
        print(f"\n{axis_names[i]} (Axis {i}):")
        print("-" * 40)
        
        # Basic statistics
        print(f"Sample size: {len(fids)}")
        print(f"Position range: {positions.min():.0f} - {positions.max():.0f}")
        print(f"Mean FID: {np.mean(fids):.3f} ± {np.std(fids):.3f}")
        print(f"Median FID: {np.median(fids):.3f}")
        print(f"Range: {np.min(fids):.3f} - {np.max(fids):.3f}")
        print(f"IQR: {np.percentile(fids, 25):.3f} - {np.percentile(fids, 75):.3f}")
        
        # Normality test
        _, p_normal = stats.shapiro(fids[:min(len(fids), 5000)])  # Shapiro-Wilk limited to 5000 samples
        print(f"Normality test (p-value): {p_normal:.6f} {'(Normal)' if p_normal > 0.05 else '(Non-normal)'}")
        
        # Correlation with position
        norm_positions = (positions - positions.min()) / (positions.max() - positions.min())
        correlation, p_corr = stats.pearsonr(norm_positions, fids)
        print(f"Position correlation: r = {correlation:.3f} (p = {p_corr:.6f})")
        
        # Outliers (using IQR method)
        Q1, Q3 = np.percentile(fids, [25, 75])
        IQR = Q3 - Q1
        outlier_threshold_low = Q1 - 1.5 * IQR
        outlier_threshold_high = Q3 + 1.5 * IQR
        outliers = fids[(fids < outlier_threshold_low) | (fids > outlier_threshold_high)]
        print(f"Outliers (IQR method): {len(outliers)} ({len(outliers)/len(fids)*100:.1f}%)")

if __name__ == "__main__":
    print("Generating enhanced FID analysis...")
    
    # Create enhanced comparison plot
    fig1 = create_enhanced_comparison_plot()
    
    # Create statistical summary
    fig2 = create_statistical_summary()
    
    # Print detailed analysis
    print_detailed_analysis()
    
    print("\nEnhanced analysis complete!")
    plt.show() 