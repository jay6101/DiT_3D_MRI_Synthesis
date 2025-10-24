import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
import os

# Set the style for publication-quality plots
plt.style.use('seaborn-v0_8')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 18,
    'font.family': 'serif'
})

# Define method paths and names
methods = {
    'Ours': '/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/fid_results_ignite',
    'α-GAN': '/space/mcdonald-syn01/1/projects/jsawant/3dbraingen/fid_results_alpha_gan_ignite',
    'WGAN': '/space/mcdonald-syn01/1/projects/jsawant/3dbraingen/fid_results_wgan_ignite',
    'α-WGAN': '/space/mcdonald-syn01/1/projects/jsawant/3dbraingen/fid_results_alpha_wgan_ignite',
    'Med-DDPM': '/space/mcdonald-syn01/1/projects/jsawant/med-ddpm/fid_results_med_ddpm'
}

# Define consistent colors for each method
method_colors = {
    'Ours': '#2E8B57',        # Sea Green
    'α-GAN': '#FF6347',       # Tomato
    'WGAN': '#4682B4',        # Steel Blue
    'α-WGAN': '#DAA520',      # Goldenrod
    'Med-DDPM': '#9370DB'     # Medium Purple
}

# Load data from all methods
all_data = {}
for method_name, path in methods.items():
    pkl_file = os.path.join(path, 'fid_results.pkl')
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
        all_data[method_name] = data

# Calculate overall means for ordering
overall_means = {}
for method_name in methods.keys():
    overall_means[method_name] = all_data[method_name]['overall_avg_fid']

# Sort methods by overall performance (ascending order)
sorted_methods = sorted(methods.keys(), key=lambda x: overall_means[x])

# 1. Create beautiful violin plot comparing all methods for each axis
fig, axes = plt.subplots(1, 3, figsize=(20, 8))
axis_names = ['Sagittal View', 'Coronal View', 'Axial View']
axis_keys = ['axis0_fids', 'axis1_fids', 'axis2_fids']

# Prepare data for violin plots
for i, (axis_name, axis_key) in enumerate(zip(axis_names, axis_keys)):
    violin_data = []
    
    for method_name in sorted_methods:  # Use sorted order
        fids = all_data[method_name][axis_key]
        violin_data.extend([(method_name, fid) for fid in fids])
    
    df = pd.DataFrame(violin_data, columns=['Method', 'FID'])
    
    # Create violin plot with beautiful styling
    violin_parts = axes[i].violinplot([df[df['Method'] == method]['FID'] for method in sorted_methods], 
                                     positions=range(len(sorted_methods)), 
                                     showmeans=True, showmedians=True)
    
    # Style the violin plots with consistent colors
    for pc, method in zip(violin_parts['bodies'], sorted_methods):
        pc.set_facecolor(method_colors[method])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
    
    # Style the statistical lines
    violin_parts['cmeans'].set_color('red')
    violin_parts['cmeans'].set_linewidth(3)
    violin_parts['cmedians'].set_color('black')
    violin_parts['cmedians'].set_linewidth(2)
    
    axes[i].set_title(f'{axis_name}', fontsize=16, fontweight='bold', pad=20)
    axes[i].set_xlabel('Method', fontsize=14, fontweight='bold')
    axes[i].set_ylabel('FID Score', fontsize=14, fontweight='bold')
    axes[i].grid(True, alpha=0.3, linestyle='--')
    axes[i].set_xticks(range(len(sorted_methods)))
    axes[i].set_xticklabels(sorted_methods, rotation=45, ha='right')
    
    # Add background color
    axes[i].set_facecolor('#f8f9fa')
    
    # Add subtle border
    for spine in axes[i].spines.values():
        spine.set_edgecolor('gray')
        spine.set_linewidth(1.5)
    
    # Create individual legend for this axis with FID values
    legend_elements = []
    for method in sorted_methods:
        # Get the mean FID for this specific axis
        avg_fid_key = axis_key.replace('_fids', '').replace('axis', 'avg_fid_axis')
        mean_val = all_data[method][avg_fid_key]
        legend_elements.append(plt.Line2D([0], [0], color=method_colors[method], lw=4, 
                                        label=f'{method}: {mean_val:.1f}'))
    
    axes[i].legend(handles=legend_elements, loc='upper left', fontsize=10, 
                  title='Mean FID', title_fontsize=11, framealpha=0.9)

plt.suptitle('FID Score Distribution Comparison Across Methods and Anatomical Views', 
             fontsize=20, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('clean_violin_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 2. Create clean performance ranking plot
fig, axes = plt.subplots(1, 3, figsize=(20, 8))

for i, (axis_name, axis_key) in enumerate(zip(axis_names, axis_keys)):
    # Get average FID for each method for this axis
    method_fids = []
    method_names_list = []
    
    for method_name in methods.keys():
        avg_fid_key = axis_key.replace('_fids', '').replace('axis', 'avg_fid_axis')
        avg_fid = all_data[method_name][avg_fid_key]
        method_fids.append(avg_fid)
        method_names_list.append(method_name)
    
    # Sort by FID score (lower is better)
    sorted_data = sorted(zip(method_fids, method_names_list))
    sorted_fids, sorted_names = zip(*sorted_data)
    
    # Use consistent colors
    colors = [method_colors[method] for method in sorted_names]
    
    bars = axes[i].bar(sorted_names, sorted_fids, color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=2, width=0.7)
    
    # Add FID values on bars
    for j, (bar, fid) in enumerate(zip(bars, sorted_fids)):
        height = bar.get_height()
        axes[i].text(bar.get_x() + bar.get_width()/2., height + max(sorted_fids) * 0.02,
                     f'{fid:.1f}', ha='center', va='bottom', 
                     fontweight='bold', fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
    
    axes[i].set_title(f'{axis_name}', fontsize=16, fontweight='bold', pad=20)
    axes[i].set_xlabel('Method', fontsize=14, fontweight='bold')
    axes[i].set_ylabel('Average FID Score', fontsize=14, fontweight='bold')
    axes[i].grid(True, alpha=0.3, axis='y', linestyle='--')
    axes[i].tick_params(axis='x', rotation=45)
    
    # Add background color
    axes[i].set_facecolor('#f8f9fa')
    
    # Add subtle border
    for spine in axes[i].spines.values():
        spine.set_edgecolor('gray')
        spine.set_linewidth(1.5)

plt.suptitle('Performance Comparison by Average FID Score Across Anatomical Views\n(Lower = Better)', 
          fontsize=20, fontweight='bold', y=0.95)
plt.tight_layout()
plt.savefig('clean_ranking_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 3. Create clean overall performance ranking plot
overall_scores = []
for method_name in methods.keys():
    overall_avg = all_data[method_name]['overall_avg_fid']
    overall_scores.append((overall_avg, method_name))

# Sort by FID score (lower is better)
overall_scores.sort()
sorted_fids, sorted_names = zip(*overall_scores)

fig, ax = plt.subplots(figsize=(14, 10))

# Use consistent colors
colors = [method_colors[method] for method in sorted_names]

bars = ax.bar(sorted_names, sorted_fids, color=colors, alpha=0.8, 
              edgecolor='black', linewidth=2, width=0.6)

# Add FID values on bars
for i, (bar, fid) in enumerate(zip(bars, sorted_fids)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + max(sorted_fids) * 0.02,
             f'{fid:.1f}', ha='center', va='bottom', 
             fontweight='bold', fontsize=14,
             bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.9))

ax.set_title('Overall Performance Comparison Across All Methods\n(Lower = Better)', 
          fontsize=20, fontweight='bold', pad=30)
ax.set_xlabel('Method', fontsize=16, fontweight='bold')
ax.set_ylabel('Overall Average FID Score', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

# Add background color
ax.set_facecolor('#f8f9fa')

# Style the spines
for spine in ax.spines.values():
    spine.set_edgecolor('gray')
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig('clean_overall_ranking.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("🎨 Clean plots created successfully!")
print("📊 Generated files:")
print("   1. clean_violin_comparison.png - Violin plots with individual axis legends")
print("   2. clean_ranking_comparison.png - Simple ranking comparison")
print("   3. clean_overall_ranking.png - Clean overall ranking")
print(f"\n{'='*50}")
print("📈 PERFORMANCE SUMMARY")
print(f"{'='*50}")
for i, (score, method) in enumerate(overall_scores, 1):
    print(f"{i}. {method}: {score:.1f}")
print(f"\nConsistent colors used across all plots:")
for method, color in method_colors.items():
    print(f"  {method}: {color}") 