#!/usr/bin/env python3
"""
Consciousness Experiment

This experiment explores consciousness-like properties in multi-agent systems.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from extensions.consciousness.consciousness_metrics import ConsciousnessMetrics
from extensions.consciousness.self_reference import SelfReferenceFrame
from core.strategic_field import StrategicField
from core.agent import Agent
from utils.visualization import create_figure, plot_field, plot_time_series

def run_experiment(output_dir=None, n_agents=100, grid_size=50, n_strategies=3, steps=500):
    """
    Run the consciousness experiment
    
    Parameters:
    - output_dir: Directory for output files
    - n_agents: Number of agents
    - grid_size: Size of the grid
    - n_strategies: Number of strategies
    - steps: Number of simulation steps
    
    Returns:
    - results: Dictionary of results
    """
    # Set up output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./results/consciousness_experiment_{timestamp}"
    
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
    logger.info(f"Starting consciousness experiment with {n_agents} agents, {n_strategies} strategies")
    
    # Create payoff matrix
    payoff_matrix = np.array([
        [0.3, 0.7, 0.1],
        [0.6, 0.2, 0.5],
        [0.2, 0.4, 0.8]
    ])
    
    if n_strategies != 3:
        # Generate random payoff matrix if not using default
        payoff_matrix = np.random.rand(n_strategies, n_strategies)
    
    logger.info(f"Using payoff matrix:\n{payoff_matrix}")
    
    # Initialize systems
    strategic_field = StrategicField(grid_size, n_strategies)
    consciousness_metrics = ConsciousnessMetrics()
    self_reference_frame = SelfReferenceFrame()
    
    # Initialize agents
    agents = []
    for i in range(n_agents):
        position = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
        strategy = np.zeros(n_strategies)
        strategy[np.random.randint(0, n_strategies)] = 1.0
        agents.append(Agent(i, position, strategy, payoff_matrix))
    
    logger.info(f"Initialized {n_agents} agents")
    
    # Run simulation
    integration_metrics = []
    coherence_history = []
    
    logger.info(f"Running simulation for {steps} steps")
    
    for step in range(steps):
        if step % 50 == 0:
            logger.info(f"Simulation step {step}/{steps}")
        
        # Update field
        strategic_field.update(agents)
        
        # Calculate coherence
        coherence = strategic_field.calculate_coherence()
        coherence_history.append(coherence)
        
        # Calculate consciousness metrics
        metrics = consciousness_metrics.calculate_metrics(strategic_field.field)
        
        # Update self-reference frame
        self_reference_frame.update(strategic_field.field, agents)
        
        # Calculate overall integration
        metrics = {
            'temporal_integration': consciousness_metrics.calculate_temporal_integration(),
            'spatial_integration': consciousness_metrics.calculate_spatial_integration(),
            'phi': consciousness_metrics.calculate_phi_measure(),
            'self_reference': self_reference_frame.calculate_self_reference()
        }
        
        # Calculate overall integration
        metrics['overall_integration'] = (
            metrics.get('temporal_integration', 0) * 0.3 +
            metrics.get('spatial_integration', 0) * 0.3 +
            metrics.get('phi', 0) * 0.2 +
            metrics.get('self_reference', 0) * 0.2
        )
        
        integration_metrics.append(metrics)
        
        # Agents make decisions
        for agent in agents:
            # Get field information at current position
            x, y = agent.position.astype(int)
            field_info = strategic_field.field[x, y]
            
            # Make decision
            new_strategy = np.zeros(n_strategies)
            new_strategy[np.argmax(field_info)] = 1.0
            agent.strategy = new_strategy
    
    logger.info(f"Simulation completed in {steps} steps")
    
    # Create figures
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Phi values
    ax = axes[0, 0]
    if integration_metrics:
        phi_values = [metrics.get('phi', 0) for metrics in integration_metrics]
        ax.plot(phi_values, linewidth=2)
        ax.set_title('Integrated Information (Phi)', fontsize=14)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Phi Value', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Overall integration
    ax = axes[0, 1]
    if integration_metrics:
        integration_values = [metrics.get('overall_integration', 0) for metrics in integration_metrics]
        ax.plot(integration_values, linewidth=2, color='green')
        ax.set_title('Overall Integration', fontsize=14)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Integration Value', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Self-reference
    ax = axes[1, 0]
    if integration_metrics:
        self_ref_values = [metrics.get('self_reference', 0) for metrics in integration_metrics]
        ax.plot(self_ref_values, linewidth=2, color='purple')
        ax.set_title('Self-Reference', fontsize=14)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Self-Reference Value', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Field coherence
    ax = axes[1, 1]
    if coherence_history:
        ax.plot(coherence_history, linewidth=2, color='orange')
        ax.set_title('Field Coherence', fontsize=14)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Coherence Value', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    # Save figure
    fig.tight_layout()
    fig_path = os.path.join(output_dir, 'consciousness_metrics.png')
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    
    # Create field visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    field_data = np.argmax(strategic_field.field, axis=2)
    im = ax.imshow(field_data, cmap='viridis')
    plt.colorbar(im, ax=ax)
    ax.set_title('Final Strategic Field (Dominant Strategy)', fontsize=14)
    
    # Save field visualization
    field_fig_path = os.path.join(output_dir, 'final_field.png')
    fig.savefig(field_fig_path, dpi=300)
    plt.close(fig)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'phi_values': [float(metrics.get('phi', 0)) for metrics in integration_metrics],
            'integration_values': [float(metrics.get('overall_integration', 0)) for metrics in integration_metrics],
            'self_reference_values': [float(metrics.get('self_reference', 0)) for metrics in integration_metrics],
            'coherence_values': [float(c) for c in coherence_history]
        }, f, indent=2)
    
    logger.info(f"Experiment completed. Results saved to {output_dir}")
    
    # Return results
    results = {
        'output_dir': output_dir,
        'metrics': integration_metrics,
        'coherence': coherence_history,
        'final_field': strategic_field.field.tolist()
    }
    
    return results

if __name__ == "__main__":
    run_experiment()
