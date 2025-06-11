#!/usr/bin/env python3
"""
Create a combined visualization of all experiment results.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

def create_combined_visualization(results_dir, output_file):
    """Create a combined visualization of all experiment results"""
    # Define paths to individual visualizations
    nash_viz = os.path.join(results_dir, "nash_equilibrium_visualization.png")
    strategic_viz = os.path.join(results_dir, "strategic_fields_visualization.png")
    fractal_viz = os.path.join(results_dir, "fractal_time_visualization.png")
    hypersensitive_viz = os.path.join(results_dir, "hypersensitive_visualization.png")
    stigmergic_viz = os.path.join(results_dir, "stigmergic_visualization.png")
    
    # Check if all visualizations exist
    for viz_path in [nash_viz, strategic_viz, fractal_viz, hypersensitive_viz, stigmergic_viz]:
        if not os.path.exists(viz_path):
            print(f"Missing visualization: {viz_path}")
            return
    
    # Create figure
    fig = plt.figure(figsize=(16, 20))
    
    # Create a 5x1 grid for the visualizations
    gs = GridSpec(5, 1, figure=fig, hspace=0.3)
    
    # Add each visualization as a subplot
    for i, (viz_path, title) in enumerate([
        (nash_viz, "1. Nash Equilibrium Proportional to Growth"),
        (strategic_viz, "2. Strategic Field Wave Propagation"),
        (fractal_viz, "3. Fractal Time Architecture"),
        (hypersensitive_viz, "4. Hypersensitive Points and Strategic Decision"),
        (stigmergic_viz, "5. Stigmergic Coordination Through IRN")
    ]):
        ax = fig.add_subplot(gs[i, 0])
        
        # Load and display the image
        img = plt.imread(viz_path)
        ax.imshow(img)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
    
    # Add overall title
    fig.suptitle('Experimental Validation of Theoretical Framework', fontsize=18, fontweight='bold', y=0.995)
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined visualization saved to {output_file}")

if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    output_file = os.path.join(results_dir, "combined_visualization.png")
    create_combined_visualization(results_dir, output_file)
