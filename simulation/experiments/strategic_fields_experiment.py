#!/usr/bin/env python3
"""
Strategic Fields Experiment for the unified MMAI system

This experiment focuses on analyzing the formation and propagation of strategic fields.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.unified_simulation import UnifiedSimulation, run_unified_simulation
from core.environment_system import EnvironmentType

def run_strategic_fields_experiment(base_dir="./results", log_level=logging.INFO):
    """
    Run the strategic fields experiment
    
    Parameters:
    - base_dir: Base directory for storing results
    - log_level: Logging level
    
    Returns:
    - results: Dictionary of experiment results
    """
    # Setup logging
    logger = logging.getLogger("strategic_fields_experiment")
    logger.setLevel(log_level)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(ch)
    
    logger.info("Starting Strategic Fields Experiment")
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(base_dir, f"strategic_fields_experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Define diffusion rates to test
    diffusion_rates = [0.1, 0.2, 0.3, 0.4]
    
    # Store results for each diffusion rate
    all_results = {}
    
    for diffusion_rate in diffusion_rates:
        logger.info(f"Testing diffusion rate: {diffusion_rate}")
        
        # Create configuration
        config = {
            'grid_size': 50,
            'n_agents': 50,
            'max_agents': 100,
            'n_strategies': 3,
            'dt': 0.01,
            't_scale': 50,
            'T_scale': 20,
            'env_type': EnvironmentType.STATIC,
            'enable_reproduction': True,
            'enable_resources': True,
            'enable_dynamic_population': True,
            'max_steps': 1000,
            'equilibrium_threshold': 0.1,
            'data_dir': experiment_dir,
            'experiment_name': f"diffusion_rate_{diffusion_rate}",
            'log_level': log_level
        }
        
        # Run simulation
        results, sim = run_unified_simulation(config)
        
        # Store results
        all_results[diffusion_rate] = results
        
        # Create additional visualizations specific to this experiment
        create_strategic_field_visualizations(sim, diffusion_rate, experiment_dir)
    
    # Create comparative visualizations
    create_comparative_visualizations(all_results, diffusion_rates, experiment_dir)
    
    logger.info(f"Strategic Fields Experiment completed. Results saved to {experiment_dir}")
    
    return {
        'experiment_dir': experiment_dir,
        'all_results': all_results,
        'diffusion_rates': diffusion_rates
    }

def create_strategic_field_visualizations(sim, diffusion_rate, experiment_dir):
    """
    Create visualizations specific to the strategic fields experiment
    
    Parameters:
    - sim: Simulation object
    - diffusion_rate: Current diffusion rate
    - experiment_dir: Directory to save visualizations
    """
    # Create directory for this diffusion rate
    viz_dir = os.path.join(experiment_dir, f"diffusion_rate_{diffusion_rate}", "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create strategic field visualization for each strategy
    for i in range(sim.n_strategies):
        fig = sim.field.visualize(strategy_idx=i, title=f"Strategy {i+1} Field (Diffusion Rate: {diffusion_rate})")
        fig_path = os.path.join(viz_dir, f"strategy_{i+1}_field.png")
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Create wave propagation visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim.field.wave_field, cmap='plasma', vmin=0)
    plt.colorbar(im, ax=ax, label='Wave Intensity')
    ax.set_title(f'Strategic Wave Propagation (Diffusion Rate: {diffusion_rate})', fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Plot agents
    agent_x = [agent.position[0] for agent in sim.agents]
    agent_y = [agent.position[1] for agent in sim.agents]
    ax.scatter(agent_y, agent_x, c='white', s=10, alpha=0.7)
    
    fig_path = os.path.join(viz_dir, "wave_propagation.png")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create coherence over time visualization
    if sim.field.coherence_history:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sim.field.coherence_history, linewidth=2)
        ax.set_title(f'Field Coherence Over Time (Diffusion Rate: {diffusion_rate})', fontsize=14)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Coherence', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        fig_path = os.path.join(viz_dir, "coherence_over_time.png")
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

def create_comparative_visualizations(all_results, diffusion_rates, experiment_dir):
    """
    Create comparative visualizations across different diffusion rates
    
    Parameters:
    - all_results: Dictionary of results for each diffusion rate
    - diffusion_rates: List of diffusion rates tested
    - experiment_dir: Directory to save visualizations
    """
    # Create directory for comparative visualizations
    comparative_dir = os.path.join(experiment_dir, "comparative_visualizations")
    os.makedirs(comparative_dir, exist_ok=True)
    
    # Compare coherence across diffusion rates
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for rate in diffusion_rates:
        results = all_results[rate]
        sim_dir = os.path.join(experiment_dir, f"diffusion_rate_{rate}")
        
        # Load metrics
        metrics_path = os.path.join(sim_dir, "metrics", "final_metrics.json")
        try:
            import json
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            # Extract coherence values
            coherence_values = [m.get('coherence', None) for m in metrics if 'coherence' in m]
            coherence_values = [v for v in coherence_values if v is not None]
            
            if coherence_values:
                ax.plot(coherence_values, label=f'Diffusion Rate: {rate}', linewidth=2)
        except Exception as e:
            print(f"Error loading metrics for diffusion rate {rate}: {e}")
    
    ax.set_title('Field Coherence Comparison Across Diffusion Rates', fontsize=14)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Coherence', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig_path = os.path.join(comparative_dir, "coherence_comparison.png")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Compare final coherence vs diffusion rate
    fig, ax = plt.subplots(figsize=(10, 6))
    
    final_coherence = []
    for rate in diffusion_rates:
        results = all_results[rate]
        final_coherence.append(results.get('coherence', 0))
    
    ax.plot(diffusion_rates, final_coherence, 'o-', linewidth=2)
    ax.set_title('Final Coherence vs Diffusion Rate', fontsize=14)
    ax.set_xlabel('Diffusion Rate', fontsize=12)
    ax.set_ylabel('Final Coherence', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    fig_path = os.path.join(comparative_dir, "final_coherence_vs_diffusion.png")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Compare Nash distance vs diffusion rate
    fig, ax = plt.subplots(figsize=(10, 6))
    
    final_nash = []
    for rate in diffusion_rates:
        results = all_results[rate]
        final_nash.append(results.get('nash_distance', 1.0))
    
    ax.plot(diffusion_rates, final_nash, 'o-', linewidth=2)
    ax.set_title('Final Nash Distance vs Diffusion Rate', fontsize=14)
    ax.set_xlabel('Diffusion Rate', fontsize=12)
    ax.set_ylabel('Final Nash Distance', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    fig_path = os.path.join(comparative_dir, "final_nash_vs_diffusion.png")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create manuscript-quality figure
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Coherence over time for each diffusion rate
    ax1 = axs[0, 0]
    for rate in diffusion_rates:
        results = all_results[rate]
        sim_dir = os.path.join(experiment_dir, f"diffusion_rate_{rate}")
        
        # Load metrics
        metrics_path = os.path.join(sim_dir, "metrics", "final_metrics.json")
        try:
            import json
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            # Extract coherence values
            coherence_values = [m.get('coherence', None) for m in metrics if 'coherence' in m]
            coherence_values = [v for v in coherence_values if v is not None]
            
            if coherence_values:
                ax1.plot(coherence_values, label=f'Rate: {rate}', linewidth=2)
        except Exception as e:
            print(f"Error loading metrics for diffusion rate {rate}: {e}")
    
    ax1.set_title('Field Coherence Over Time', fontsize=12)
    ax1.set_xlabel('Time Step', fontsize=10)
    ax1.set_ylabel('Coherence', fontsize=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final coherence vs diffusion rate
    ax2 = axs[0, 1]
    ax2.plot(diffusion_rates, final_coherence, 'o-', linewidth=2)
    ax2.set_title('Final Coherence vs Diffusion Rate', fontsize=12)
    ax2.set_xlabel('Diffusion Rate', fontsize=10)
    ax2.set_ylabel('Final Coherence', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Nash distance over time for each diffusion rate
    ax3 = axs[1, 0]
    for rate in diffusion_rates:
        results = all_results[rate]
        sim_dir = os.path.join(experiment_dir, f"diffusion_rate_{rate}")
        
        # Load metrics
        metrics_path = os.path.join(sim_dir, "metrics", "final_metrics.json")
        try:
            import json
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            # Extract Nash distance values
            nash_values = [m.get('nash_distance', None) for m in metrics if 'nash_distance' in m]
            nash_values = [v for v in nash_values if v is not None]
            
            if nash_values:
                ax3.plot(nash_values, label=f'Rate: {rate}', linewidth=2)
        except Exception as e:
            print(f"Error loading metrics for diffusion rate {rate}: {e}")
    
    ax3.set_title('Nash Distance Over Time', fontsize=12)
    ax3.set_xlabel('Time Step', fontsize=10)
    ax3.set_ylabel('Nash Distance', fontsize=10)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Final Nash distance vs diffusion rate
    ax4 = axs[1, 1]
    ax4.plot(diffusion_rates, final_nash, 'o-', linewidth=2)
    ax4.set_title('Final Nash Distance vs Diffusion Rate', fontsize=12)
    ax4.set_xlabel('Diffusion Rate', fontsize=10)
    ax4.set_ylabel('Final Nash Distance', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig_path = os.path.join(comparative_dir, "strategic_fields_manuscript.png")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    run_strategic_fields_experiment()
