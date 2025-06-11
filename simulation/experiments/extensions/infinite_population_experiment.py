#!/usr/bin/env python3
"""
Infinite Population Experiment

This experiment compares finite agent simulations with infinite population
approximations, analyzing how results scale with population size and
extrapolating to the infinite limit.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from extensions.infinite_population import MeanFieldApproximation, ScalingAnalysis
from simulation.simulation import Simulation
from core.nash_validator import NashValidator

def run_experiment(output_dir=None, n_strategies=3, growth_rate=0.05, max_steps=1000):
    """
    Run the infinite population experiment
    
    Parameters:
    - output_dir: Directory for output files
    - n_strategies: Number of strategies
    - growth_rate: System growth rate
    - max_steps: Maximum simulation steps
    
    Returns:
    - results: Dictionary of experiment results
    """
    # Set up logging
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./results/infinite_population_experiment_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, "experiment.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting infinite population experiment with {n_strategies} strategies")
    
    # Create fixed payoff matrix for comparison
    payoff_matrix = np.array([
        [0.3, 0.7, 0.1],
        [0.6, 0.2, 0.5],
        [0.2, 0.4, 0.8]
    ])
    
    if n_strategies != 3:
        # Generate random payoff matrix for different strategy counts
        payoff_matrix = np.random.rand(n_strategies, n_strategies)
    
    logger.info(f"Using payoff matrix:\n{payoff_matrix}")
    
    # Part 1: Compare discrete-time mean field with finite simulation
    logger.info("Part 1: Comparing discrete-time mean field with finite simulation")
    
    # Run mean field simulation
    mean_field = MeanFieldApproximation(
        n_strategies=n_strategies,
        growth_rate=growth_rate
    )
    
    mean_field_results = {
        'nash_distance': [],
        'strategy_history': []
    }
    
    for step in range(max_steps):
        nash_distance = mean_field.update(payoff_matrix)
        mean_field_results['nash_distance'].append(nash_distance)
        mean_field_results['strategy_history'].append(mean_field.strategy_distribution.copy())
    
    # Run finite simulation with 100 agents
    logger.info("Running finite simulation with 100 agents")
    
    # Create strategic field
    from core.strategic_field import StrategicField
    strategic_field = StrategicField(grid_size=50, n_strategies=n_strategies)
    
    # Create fractal time manager
    from core.fractal_time_manager import FractalTimeManager
    fractal_time_manager = FractalTimeManager(dt=0.01, t_scale=50, T_scale=20)
    
    # Create Nash validator
    nash_validator = NashValidator(n_strategies=n_strategies)
    
    # Create payoff matrix
    payoff_matrix = np.array([
        [0.3, 0.7, 0.1],
        [0.6, 0.2, 0.5],
        [0.2, 0.4, 0.8]
    ])
    
    if n_strategies != 3:
        # Generate random payoff matrix if not using default
        payoff_matrix = np.random.rand(n_strategies, n_strategies)
    
    # Create simulation
    finite_sim = Simulation(
        strategic_field=strategic_field,
        n_agents=100,
        n_strategies=n_strategies,
        payoff_matrix=payoff_matrix,
        fractal_time_manager=fractal_time_manager,
        nash_validator=nash_validator,
        growth_rate=growth_rate
    )
    
    finite_results = finite_sim.run(max_steps=max_steps)
    
    # Compare results
    fig = mean_field.compare_with_finite(
        finite_results['nash_distance_history']
    )
    
    fig.savefig(os.path.join(output_dir, "mean_field_vs_finite.png"))
    
    # Part 2: Run continuous-time mean field simulation
    logger.info("Part 2: Running continuous-time mean field simulation")
    
    continuous_mean_field = MeanFieldApproximation(
        n_strategies=n_strategies,
        growth_rate=growth_rate
    )
    
    continuous_results = continuous_mean_field.simulate_continuous(
        payoff_matrix,
        t_span=(0, max_steps * 0.1),  # Scale time to match discrete steps
        t_points=max_steps
    )
    
    # Visualize continuous results
    fig = continuous_mean_field.visualize_nash_distance()
    fig.savefig(os.path.join(output_dir, "continuous_mean_field_nash.png"))
    
    fig = continuous_mean_field.visualize_strategy_distribution()
    fig.savefig(os.path.join(output_dir, "continuous_mean_field_strategies.png"))
    
    # Part 3: Run scaling analysis
    logger.info("Part 3: Running scaling analysis")
    
    scaling = ScalingAnalysis(
        n_strategies=n_strategies,
        growth_rate=growth_rate,
        logger=logger
    )
    
    scaling_results = scaling.run_scaling_analysis(
        simulation_class=Simulation,
        payoff_matrix=payoff_matrix,
        min_agents=10,
        max_agents=500,
        steps=5,
        max_steps=max_steps
    )
    
    # Visualize scaling results
    fig = scaling.visualize_scaling()
    fig.savefig(os.path.join(output_dir, "scaling_analysis.png"))
    
    fig = scaling.visualize_nash_distance_comparison()
    fig.savefig(os.path.join(output_dir, "nash_distance_comparison.png"))
    
    # Save combined figure for manuscript
    logger.info("Creating manuscript figure")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Mean field vs finite
    ax = axes[0, 0]
    ax.plot(mean_field_results['nash_distance'], linewidth=2, label='Mean Field')
    ax.plot(finite_results['nash_distance_history'], linewidth=2, linestyle='--', label='Finite (N=100)')
    ax.set_title('Nash Distance: Mean Field vs Finite', fontsize=14)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Nash Distance', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Strategy distribution
    ax = axes[0, 1]
    strategy_history = np.array(mean_field_results['strategy_history'])
    for i in range(n_strategies):
        ax.plot(strategy_history[:, i], linewidth=2, label=f'Strategy {i+1}')
    ax.set_title('Strategy Distribution (Mean Field)', fontsize=14)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Strategy Probability', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Scaling analysis - convergence time
    ax = axes[1, 0]
    sizes = np.array(scaling.population_sizes)
    convergence_times = np.array([
        scaling.finite_results[size]['convergence_time'] 
        for size in sizes
    ])
    ax.loglog(sizes, convergence_times, 'o-', linewidth=2)
    ax.set_title('Convergence Time vs Population Size', fontsize=14)
    ax.set_xlabel('Population Size (log scale)', fontsize=12)
    ax.set_ylabel('Convergence Time (log scale)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Nash distance comparison across population sizes
    ax = axes[1, 1]
    ax.plot(
        continuous_results['time'],
        continuous_results['nash_distance_history'],
        'r-', linewidth=2, label='Infinite (Continuous)'
    )
    for size in [sizes[0], sizes[-1]]:  # Smallest and largest
        nash_distance = scaling.finite_results[size]['nash_distance']
        ax.plot(
            range(len(nash_distance)),
            nash_distance,
            '--', linewidth=1.5,
            label=f'N = {size}'
        )
    ax.set_title('Nash Distance Comparison', fontsize=14)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Nash Distance', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "manuscript_figure.png"))
    
    logger.info(f"Experiment completed. Results saved to {output_dir}")
    
    return {
        'mean_field': mean_field_results,
        'finite': finite_results,
        'continuous': continuous_results,
        'scaling': scaling_results,
        'output_dir': output_dir
    }
    
    # Export raw data for analysis
    data_export = {
        'nash_distance': {
            'mean_field': mean_field_results.get('nash_distance', []),
            'finite': finite_results.get('nash_distance_history', [])
        },
        'strategy_history': {
            'mean_field': [s.tolist() if isinstance(s, np.ndarray) else s for s in mean_field_results.get('strategy_history', [])]
        },
        'continuous': {
            'time': continuous_results.get('time', []).tolist() if isinstance(continuous_results.get('time', []), np.ndarray) else continuous_results.get('time', []),
            'nash_distance': continuous_results.get('nash_distance_history', [])
        },
        'scaling': {
            'population_sizes': scaling_results.get('population_sizes', []).tolist() if isinstance(scaling_results.get('population_sizes', []), np.ndarray) else scaling_results.get('population_sizes', []),
            'convergence_times': [],
            'final_nash_distances': []
        },
        'proportionality': scaling.calculate_growth_proportional_equilibrium()[1] if hasattr(scaling, 'calculate_growth_proportional_equilibrium') else None
    }
    
    # Extract scaling data
    if hasattr(scaling_results, 'get'):
        for size in data_export['scaling']['population_sizes']:
            if size in scaling_results.get('finite_results', {}):
                data_export['scaling']['convergence_times'].append(
                    scaling_results['finite_results'][size].get('convergence_time', 0)
                )
                data_export['scaling']['final_nash_distances'].append(
                    scaling_results['finite_results'][size].get('final_nash_distance', 0)
                )
    
    # Save raw data using the data_collection module
    from experiments.extensions.data_collection import save_raw_data
    save_raw_data(data_export, output_dir)
    
    return {
        'mean_field': mean_field_results,
        'finite': finite_results,
        'continuous': continuous_results,
        'scaling': scaling_results,
        'output_dir': output_dir,
        'data_export': data_export
    }

if __name__ == "__main__":
    run_experiment()
