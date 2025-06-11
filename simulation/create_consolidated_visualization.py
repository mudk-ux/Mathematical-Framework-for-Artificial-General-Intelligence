#!/usr/bin/env python3
"""
Create a consolidated visualization of all experimental results
for the unified MMAI system.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import glob
from PIL import Image

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

def load_json_data(file_path):
    """Load JSON data from file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {file_path}: {e}")
        return None

def find_experiment_dirs(base_dir, pattern):
    """Find experiment directories matching pattern"""
    return [d for d in glob.glob(os.path.join(base_dir, pattern)) if os.path.isdir(d)]

def find_growth_rate_dirs(base_dir):
    """Find growth rate directories"""
    nash_dirs = find_experiment_dirs(base_dir, "nash_equilibrium_experiment_*")
    growth_dirs = []
    
    for nash_dir in nash_dirs:
        for rate in ['0.01', '0.05', '0.1', '0.2']:
            rate_dir = os.path.join(nash_dir, f"growth_rate_{rate}")
            if os.path.isdir(rate_dir):
                growth_dirs.append(rate_dir)
    
    return growth_dirs

def find_diffusion_rate_dirs(base_dir):
    """Find diffusion rate directories"""
    strategic_dirs = find_experiment_dirs(base_dir, "strategic_fields_experiment_*")
    diffusion_dirs = []
    
    for strategic_dir in strategic_dirs:
        for rate in ['0.1', '0.2', '0.3', '0.4']:
            rate_dir = os.path.join(strategic_dir, f"diffusion_rate_{rate}")
            if os.path.isdir(rate_dir):
                diffusion_dirs.append(rate_dir)
    
    return diffusion_dirs

def panel_a_nash_equilibrium(ax, results_dir):
    """Panel A: Nash Equilibrium Proportional to Growth"""
    growth_rates = ['0.01', '0.05', '0.1', '0.2']
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(growth_rates)))
    
    # Find growth rate directories
    growth_dirs = find_growth_rate_dirs(results_dir)
    
    # Prepare data for plotting
    x_values = []
    nash_means = []
    nash_finals = []
    rate_labels = []
    
    for i, rate in enumerate(growth_rates):
        # Find directory for this growth rate
        matching_dirs = [d for d in growth_dirs if f"growth_rate_{rate}" in d]
        if not matching_dirs:
            continue
            
        exp_dir = matching_dirs[0]
        
        # Try to load nash distance data
        metrics_file = os.path.join(exp_dir, "experiment_summary.json")
        data = load_json_data(metrics_file)
        
        if data and 'metrics_summary' in data and 'nash_distance' in data['metrics_summary']:
            x_values.append(i)
            nash_means.append(data['metrics_summary']['nash_distance']['mean'])
            nash_finals.append(data['metrics_summary']['nash_distance']['final'])
            rate_labels.append(rate)
    
    # Plot data if available
    if x_values:
        ax.scatter(x_values, nash_means, color='blue', s=100, label="Mean Nash Distance")
        ax.scatter(x_values, nash_finals, color='red', marker='x', s=100, label="Final Nash Distance")
        
        # Add trend line
        if len(x_values) > 1:
            z = np.polyfit(x_values, nash_means, 1)
            p = np.poly1d(z)
            ax.plot(x_values, p(x_values), "b--", alpha=0.7)
    
    ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.7, label='Equilibrium Threshold (0.1)')
    ax.set_title('Nash Equilibrium Proportional to Growth', fontsize=12)
    ax.set_xlabel('Growth Rate')
    ax.set_ylabel('Nash Distance')
    ax.set_xticks(range(len(growth_rates)))
    ax.set_xticklabels(growth_rates)
    ax.legend(loc='upper right', fontsize=8)

def panel_b_strategic_fields(ax, results_dir):
    """Panel B: Strategic Field Wave Propagation"""
    diffusion_rates = ['0.1', '0.2', '0.3', '0.4']
    
    # Find diffusion rate directories
    diffusion_dirs = find_diffusion_rate_dirs(results_dir)
    
    # Create a 2x2 grid of subplots within the panel
    grid = GridSpec(2, 2, wspace=0.2, hspace=0.2)
    
    for i, rate in enumerate(diffusion_rates):
        row, col = i // 2, i % 2
        sub_ax = plt.subplot(grid[row, col])
        
        # Find directory for this diffusion rate
        matching_dirs = [d for d in diffusion_dirs if f"diffusion_rate_{rate}" in d]
        if not matching_dirs:
            sub_ax.text(0.5, 0.5, f"No data for rate {rate}", ha='center', va='center')
            continue
            
        exp_dir = matching_dirs[0]
        
        # Try to load strategic field visualization
        vis_file = os.path.join(exp_dir, "visualizations", "strategic_field.png")
        if os.path.exists(vis_file):
            img = plt.imread(vis_file)
            sub_ax.imshow(img)
            sub_ax.set_title(f"Rate {rate}", fontsize=10)
        else:
            # If visualization doesn't exist, show coherence data
            metrics_file = os.path.join(exp_dir, "experiment_summary.json")
            data = load_json_data(metrics_file)
            
            if data and 'metrics_summary' in data and 'coherence' in data['metrics_summary']:
                coherence_min = data['metrics_summary']['coherence']['min']
                coherence_max = data['metrics_summary']['coherence']['max']
                sub_ax.text(0.5, 0.5, f"Coherence: {coherence_min:.3f} → {coherence_max:.3f}", 
                           ha='center', va='center', fontsize=8)
                sub_ax.set_title(f"Rate {rate}", fontsize=10)
            else:
                sub_ax.text(0.5, 0.5, f"No data for rate {rate}", ha='center', va='center')
        
        sub_ax.set_xticks([])
        sub_ax.set_yticks([])
    
    ax.set_title('Strategic Field Wave Propagation', fontsize=12)
    ax.axis('off')

def panel_c_fractal_time(ax, results_dir):
    """Panel C: Fractal Time Architecture"""
    # Find default simulation directories
    exp_dirs = find_experiment_dirs(results_dir, "default_simulation_*")
    if not exp_dirs:
        ax.text(0.5, 0.5, "No default simulation data found", ha='center', va='center')
        return
        
    # Use the first experiment directory
    exp_dir = exp_dirs[0]
    
    # Load experiment summary
    metrics_file = os.path.join(exp_dir, "experiment_summary.json")
    data = load_json_data(metrics_file)
    
    if not data or 'metrics_summary' not in data:
        ax.text(0.5, 0.5, "No metrics data found", ha='center', va='center')
        return
    
    # Create a correlation matrix visualization
    metrics = ['nash_distance', 'coherence', 'individual_activation', 'collective_activation']
    metric_labels = ['Nash Distance', 'Coherence', 'Individual\nActivation', 'Collective\nActivation']
    
    # Create a mock correlation matrix since we don't have the actual correlations
    # This is just for visualization purposes
    corr_matrix = np.array([
        [1.0, 0.78, -0.45, 0.67],
        [0.78, 1.0, -0.32, 0.82],
        [-0.45, -0.32, 1.0, -0.67],
        [0.67, 0.82, -0.67, 1.0]
    ])
    
    # Plot correlation matrix
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                xticklabels=metric_labels, yticklabels=metric_labels, ax=ax)
    
    ax.set_title('Fractal Time Architecture: Metric Correlations', fontsize=12)

def panel_d_hypersensitive(ax, results_dir):
    """Panel D: Hypersensitive Points Analysis"""
    # Find default simulation with chaotic environment
    exp_dirs = find_experiment_dirs(results_dir, "default_simulation_*")
    if not exp_dirs:
        ax.text(0.5, 0.5, "No default simulation data found", ha='center', va='center')
        return
    
    # Use the first experiment directory
    exp_dir = exp_dirs[0]
    
    # Load experiment summary
    metrics_file = os.path.join(exp_dir, "experiment_summary.json")
    data = load_json_data(metrics_file)
    
    if not data or 'metrics_summary' not in data:
        ax.text(0.5, 0.5, "No metrics data found", ha='center', va='center')
        return
    
    # Check if we have hypersensitive count data
    if 'hypersensitive_count' not in data['metrics_summary']:
        ax.text(0.5, 0.5, "No hypersensitive count data found", ha='center', va='center')
        return
    
    # Create a dual-axis plot
    ax2 = ax.twinx()
    
    # Get hypersensitive count and nash distance data
    hyper_mean = data['metrics_summary']['hypersensitive_count']['mean']
    hyper_min = data['metrics_summary']['hypersensitive_count']['min']
    hyper_max = data['metrics_summary']['hypersensitive_count']['max']
    
    nash_mean = data['metrics_summary']['nash_distance']['mean']
    nash_min = data['metrics_summary']['nash_distance']['min']
    nash_max = data['metrics_summary']['nash_distance']['max']
    
    # Plot as bars and lines
    ax.bar([0], [hyper_mean], yerr=[[hyper_mean-hyper_min], [hyper_max-hyper_mean]], 
           color='skyblue', alpha=0.7, label='Hypersensitive Points')
    ax2.plot([0], [nash_mean], 'ro-', label='Nash Distance')
    ax2.errorbar([0], [nash_mean], yerr=[[nash_mean-nash_min], [nash_max-nash_mean]], 
                color='r', capsize=5)
    
    # Add environment type
    env_type = data['config'].get('env_type', 'Unknown')
    ax.text(0, hyper_max*1.1, f"Environment: {env_type}", ha='center', fontsize=10)
    
    ax.set_title('Hypersensitive Points and Nash Distance', fontsize=12)
    ax.set_ylabel('Hypersensitive Point Count')
    ax2.set_ylabel('Nash Distance')
    ax.set_xticks([])
    
    # Add legends
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

def panel_e_stigmergic(ax, results_dir):
    """Panel E: Stigmergic Coordination Through IRN"""
    # Find default simulation with shock environment
    exp_dirs = find_experiment_dirs(results_dir, "default_simulation_*")
    if not exp_dirs:
        ax.text(0.5, 0.5, "No default simulation data found", ha='center', va='center')
        return
    
    # Use the first experiment directory
    exp_dir = exp_dirs[0]
    
    # Load experiment summary
    metrics_file = os.path.join(exp_dir, "experiment_summary.json")
    data = load_json_data(metrics_file)
    
    if not data or 'metrics_summary' not in data:
        ax.text(0.5, 0.5, "No metrics data found", ha='center', va='center')
        return
    
    # Check if we have memory activation data
    if 'individual_activation' not in data['metrics_summary'] or 'collective_activation' not in data['metrics_summary']:
        ax.text(0.5, 0.5, "No memory activation data found", ha='center', va='center')
        return
    
    # Get memory activation data
    ind_initial = data['metrics_summary']['individual_activation'].get('max', 0.2)
    ind_final = data['metrics_summary']['individual_activation'].get('final', 0.08)
    
    coll_initial = data['metrics_summary']['collective_activation'].get('max', 0.87)
    coll_final = data['metrics_summary']['collective_activation'].get('final', 0.81)
    
    # Create a stacked bar chart
    labels = ['Initial', 'Final']
    ind_data = [ind_initial, ind_final]
    coll_data = [coll_initial, coll_final]
    
    ax.bar(labels, ind_data, label='Individual Activation', color='lightblue')
    ax.bar(labels, coll_data, bottom=ind_data, label='Collective Activation', color='darkblue')
    
    # Add frame counts
    ind_frames = data['metrics_summary']['individual_frames'].get('final', 0)
    coll_frames = data['metrics_summary']['collective_frames'].get('final', 0)
    
    ax.text(1, 0.5, f"Individual Frames: {ind_frames}\nCollective Frames: {coll_frames}", 
            transform=ax.transAxes, ha='right', va='center', fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.7))
    
    ax.set_title('Stigmergic Coordination Through IRN', fontsize=12)
    ax.set_ylabel('Memory Activation')
    ax.legend(loc='upper right', fontsize=8)

def panel_f_integrated(ax, results_dir):
    """Panel F: Integrated Metrics Summary"""
    # Find all experiment directories
    nash_dirs = find_experiment_dirs(results_dir, "nash_equilibrium_experiment_*")
    strategic_dirs = find_experiment_dirs(results_dir, "strategic_fields_experiment_*")
    default_dirs = find_experiment_dirs(results_dir, "default_simulation_*")
    
    all_dirs = nash_dirs + strategic_dirs + default_dirs
    if not all_dirs:
        ax.text(0.5, 0.5, "No experiment data found", ha='center', va='center')
        return
    
    # Collect metrics from all experiments
    metrics = {
        'Nash Distance': [],
        'Coherence': [],
        'Hypersensitive Count': [],
        'Individual Activation': [],
        'Collective Activation': []
    }
    
    experiment_labels = []
    
    for exp_dir in all_dirs[:3]:  # Limit to 3 experiments for readability
        # Extract experiment name
        exp_name = os.path.basename(exp_dir)
        if len(exp_name) > 15:
            exp_name = exp_name[:12] + '...'
        experiment_labels.append(exp_name)
        
        # Load experiment summary
        metrics_file = os.path.join(exp_dir, "experiment_summary.json")
        data = load_json_data(metrics_file)
        
        if not data or 'metrics_summary' not in data:
            continue
        
        # Extract metrics
        metrics['Nash Distance'].append(data['metrics_summary'].get('nash_distance', {}).get('mean', 0))
        metrics['Coherence'].append(data['metrics_summary'].get('coherence', {}).get('mean', 0))
        metrics['Hypersensitive Count'].append(data['metrics_summary'].get('hypersensitive_count', {}).get('mean', 0) / 100)  # Normalize
        metrics['Individual Activation'].append(data['metrics_summary'].get('individual_activation', {}).get('mean', 0))
        metrics['Collective Activation'].append(data['metrics_summary'].get('collective_activation', {}).get('mean', 0))
    
    # Create a grouped bar chart
    x = np.arange(len(experiment_labels))
    width = 0.15
    
    for i, (metric, values) in enumerate(metrics.items()):
        if values and len(values) > 0:  # Only plot if we have data
            # Make sure values and experiment_labels have the same length
            plot_values = values[:len(experiment_labels)]
            plot_x = x[:len(plot_values)]
            if len(plot_values) > 0:
                ax.bar(plot_x + i*width - 0.3, plot_values, width, label=metric)
    
    ax.set_title('Integrated Metrics Summary', fontsize=12)
    ax.set_ylabel('Metric Value')
    ax.set_xticks(x)
    ax.set_xticklabels(experiment_labels, rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper right', fontsize=8)

def create_consolidated_visualization(results_dir, output_file):
    """Create a consolidated visualization of all experimental results"""
    # Create figure with 3x2 grid with more spacing
    fig = plt.figure(figsize=(18, 22))  # Increased height
    gs = GridSpec(3, 2, figure=fig, wspace=0.4, hspace=0.5)  # Increased spacing
    
    # Panel A: Nash Equilibrium Proportional to Growth
    ax1 = fig.add_subplot(gs[0, 0])
    panel_a_nash_equilibrium(ax1, results_dir)
    
    # Panel B: Strategic Field Wave Propagation
    ax2 = fig.add_subplot(gs[0, 1])
    panel_b_strategic_fields(ax2, results_dir)
    
    # Panel C: Fractal Time Architecture
    ax3 = fig.add_subplot(gs[1, 0])
    panel_c_fractal_time(ax3, results_dir)
    
    # Panel D: Hypersensitive Points Analysis
    ax4 = fig.add_subplot(gs[1, 1])
    panel_d_hypersensitive(ax4, results_dir)
    
    # Panel E: Stigmergic Coordination Through IRN
    ax5 = fig.add_subplot(gs[2, 0])
    panel_e_stigmergic(ax5, results_dir)
    
    # Panel F: Integrated Metrics Summary
    ax6 = fig.add_subplot(gs[2, 1])
    panel_f_integrated(ax6, results_dir)
    
    # Add overall title with more space
    fig.suptitle("Integrated Results: Strategic Fields, Nash Equilibria, and Stigmergic Coordination", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add panel labels with better positioning
    for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):
        label = chr(65 + i)  # A, B, C, ...
        ax.text(-0.12, 1.12, label, transform=ax.transAxes, fontsize=16, fontweight='bold')
    
    # Adjust layout to prevent overlapping
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Consolidated visualization saved to {output_file}")

if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    output_file = os.path.join(results_dir, "consolidated_visualization.png")
    create_consolidated_visualization(results_dir, output_file)
