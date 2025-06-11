#!/usr/bin/env python3
"""
Nash Equilibrium Experiment for the unified MMAI system

This experiment focuses on analyzing the emergence of Nash equilibria across temporal scales.
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

def run_nash_equilibrium_experiment(base_dir="./results", log_level=logging.INFO):
    """
    Run the Nash equilibrium experiment
    
    Parameters:
    - base_dir: Base directory for storing results
    - log_level: Logging level
    
    Returns:
    - results: Dictionary of experiment results
    """
    # Setup logging
    logger = logging.getLogger("nash_equilibrium_experiment")
    logger.setLevel(log_level)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(ch)
    
    logger.info("Starting Nash Equilibrium Experiment")
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(base_dir, f"nash_equilibrium_experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Define growth rates to test
    growth_rates = [0.01, 0.05, 0.1, 0.2]
    
    # Store results for each growth rate
    all_results = {}
    
    for growth_rate in growth_rates:
        logger.info(f"Testing growth rate: {growth_rate}")
        
        # Create configuration
        config = {
            'grid_size': 50,
            'n_agents': 30,
            'max_agents': 150,
            'n_strategies': 3,
            'dt': 0.01,
            't_scale': 50,
            'T_scale': 20,
            'env_type': EnvironmentType.STATIC,
            'enable_reproduction': True,
            'enable_resources': True,
            'enable_dynamic_population': True,
            'max_steps': 1500,
            'equilibrium_threshold': 0.1,
            'data_dir': experiment_dir,
            'experiment_name': f"growth_rate_{growth_rate}",
            'log_level': log_level
        }
        
        # Run simulation
        results, sim = run_unified_simulation(config)
        
        # Modify Nash validator growth rate
        sim.nash_validator.growth_rate = growth_rate
        
        # Store results
        all_results[growth_rate] = results
        
        # Create additional visualizations specific to this experiment
        create_nash_equilibrium_visualizations(sim, growth_rate, experiment_dir)
    
    # Create comparative visualizations
    create_comparative_visualizations(all_results, growth_rates, experiment_dir)
    
    logger.info(f"Nash Equilibrium Experiment completed. Results saved to {experiment_dir}")
    
    return {
        'experiment_dir': experiment_dir,
        'all_results': all_results,
        'growth_rates': growth_rates
    }

def create_nash_equilibrium_visualizations(sim, growth_rate, experiment_dir):
    """
    Create visualizations specific to the Nash equilibrium experiment
    
    Parameters:
    - sim: Simulation object
    - growth_rate: Current growth rate
    - experiment_dir: Directory to save visualizations
    """
    # Create directory for this growth rate
    viz_dir = os.path.join(experiment_dir, f"growth_rate_{growth_rate}", "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create payoff matrix visualization
    fig = sim.nash_validator.visualize_payoff_matrix()
    if fig:
        fig_path = os.path.join(viz_dir, "payoff_matrix.png")
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Create strategy distribution visualization
    fig = sim.nash_validator.visualize_strategy_distribution()
    if fig:
        fig_path = os.path.join(viz_dir, "strategy_distribution.png")
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Create growth-proportional equilibrium visualization
    fig = sim.nash_validator.visualize_growth_proportional_equilibrium()
    if fig:
        fig_path = os.path.join(viz_dir, "growth_proportional_equilibrium.png")
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Create Nash distance over time visualization
    if sim.nash_validator.nash_distance_history:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sim.nash_validator.nash_distance_history, linewidth=2)
        ax.axhline(y=sim.equilibrium_threshold, color='r', linestyle='--', alpha=0.7, 
                  label=f'Equilibrium Threshold ({sim.equilibrium_threshold})')
        
        ax.set_title(f'Nash Distance Over Time (Growth Rate: {growth_rate})', fontsize=14)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Nash Distance', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig_path = os.path.join(viz_dir, "nash_distance_over_time.png")
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Create population growth visualization
    if sim.nash_validator.population_history:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sim.nash_validator.population_history, linewidth=2)
        
        ax.set_title(f'Population Growth Over Time (Growth Rate: {growth_rate})', fontsize=14)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Population Size', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        fig_path = os.path.join(viz_dir, "population_growth.png")
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

def create_comparative_visualizations(all_results, growth_rates, experiment_dir):
    """
    Create comparative visualizations across different growth rates
    
    Parameters:
    - all_results: Dictionary of results for each growth rate
    - growth_rates: List of growth rates tested
    - experiment_dir: Directory to save visualizations
    """
    # Create directory for comparative visualizations
    comparative_dir = os.path.join(experiment_dir, "comparative_visualizations")
    os.makedirs(comparative_dir, exist_ok=True)
    
    # Compare Nash distance across growth rates
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for rate in growth_rates:
        results = all_results[rate]
        sim_dir = os.path.join(experiment_dir, f"growth_rate_{rate}")
        
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
                ax.plot(nash_values, label=f'Growth Rate: {rate}', linewidth=2)
        except Exception as e:
            print(f"Error loading metrics for growth rate {rate}: {e}")
    
    ax.set_title('Nash Distance Comparison Across Growth Rates', fontsize=14)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Nash Distance', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig_path = os.path.join(comparative_dir, "nash_distance_comparison.png")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Compare proportionality vs growth rate
    fig, ax = plt.subplots(figsize=(10, 6))
    
    proportionality_values = []
    for rate in growth_rates:
        results = all_results[rate]
        proportionality = results.get('summary', {}).get('metrics_summary', {}).get('proportionality', {}).get('final', 0)
        proportionality_values.append(proportionality)
    
    ax.plot(growth_rates, proportionality_values, 'o-', linewidth=2)
    ax.set_title('Equilibrium Proportionality vs Growth Rate', fontsize=14)
    ax.set_xlabel('Growth Rate', fontsize=12)
    ax.set_ylabel('Proportionality', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    fig_path = os.path.join(comparative_dir, "proportionality_vs_growth.png")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Compare time to equilibrium vs growth rate
    fig, ax = plt.subplots(figsize=(10, 6))
    
    equilibrium_times = []
    for rate in growth_rates:
        results = all_results[rate]
        sim_dir = os.path.join(experiment_dir, f"growth_rate_{rate}")
        
        # Load metrics
        metrics_path = os.path.join(sim_dir, "metrics", "final_metrics.json")
        try:
            import json
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            # Find first equilibrium
            equilibrium_time = None
            for i, m in enumerate(metrics):
                if m.get('is_equilibrium', False):
                    equilibrium_time = i
                    break
            
            equilibrium_times.append(equilibrium_time if equilibrium_time is not None else len(metrics))
        except Exception as e:
            print(f"Error loading metrics for growth rate {rate}: {e}")
            equilibrium_times.append(None)
    
    # Filter out None values
    valid_rates = [rate for i, rate in enumerate(growth_rates) if equilibrium_times[i] is not None]
    valid_times = [time for time in equilibrium_times if time is not None]
    
    if valid_rates and valid_times:
        ax.plot(valid_rates, valid_times, 'o-', linewidth=2)
        ax.set_title('Time to Equilibrium vs Growth Rate', fontsize=14)
        ax.set_xlabel('Growth Rate', fontsize=12)
        ax.set_ylabel('Time Steps to Equilibrium', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        fig_path = os.path.join(comparative_dir, "equilibrium_time_vs_growth.png")
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Create manuscript-quality figure
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Nash distance over time for each growth rate
    ax1 = axs[0, 0]
    for rate in growth_rates:
        results = all_results[rate]
        sim_dir = os.path.join(experiment_dir, f"growth_rate_{rate}")
        
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
                ax1.plot(nash_values, label=f'Rate: {rate}', linewidth=2)
        except Exception as e:
            print(f"Error loading metrics for growth rate {rate}: {e}")
    
    ax1.set_title('Nash Distance Over Time', fontsize=12)
    ax1.set_xlabel('Time Step', fontsize=10)
    ax1.set_ylabel('Nash Distance', fontsize=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Proportionality vs growth rate
    ax2 = axs[0, 1]
    ax2.plot(growth_rates, proportionality_values, 'o-', linewidth=2)
    ax2.set_title('Equilibrium Proportionality vs Growth Rate', fontsize=12)
    ax2.set_xlabel('Growth Rate', fontsize=10)
    ax2.set_ylabel('Proportionality', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Population growth for each growth rate
    ax3 = axs[1, 0]
    for rate in growth_rates:
        results = all_results[rate]
        sim_dir = os.path.join(experiment_dir, f"growth_rate_{rate}")
        
        # Load metrics
        metrics_path = os.path.join(sim_dir, "metrics", "final_metrics.json")
        try:
            import json
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            # Extract population values
            pop_values = [m.get('population', None) for m in metrics if 'population' in m]
            pop_values = [v for v in pop_values if v is not None]
            
            if pop_values:
                ax3.plot(pop_values, label=f'Rate: {rate}', linewidth=2)
        except Exception as e:
            print(f"Error loading metrics for growth rate {rate}: {e}")
    
    ax3.set_title('Population Growth Over Time', fontsize=12)
    ax3.set_xlabel('Time Step', fontsize=10)
    ax3.set_ylabel('Population Size', fontsize=10)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Time to equilibrium vs growth rate
    ax4 = axs[1, 1]
    if valid_rates and valid_times:
        ax4.plot(valid_rates, valid_times, 'o-', linewidth=2)
        ax4.set_title('Time to Equilibrium vs Growth Rate', fontsize=12)
        ax4.set_xlabel('Growth Rate', fontsize=10)
        ax4.set_ylabel('Time Steps to Equilibrium', fontsize=10)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig_path = os.path.join(comparative_dir, "nash_equilibrium_manuscript.png")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    run_nash_equilibrium_experiment()
