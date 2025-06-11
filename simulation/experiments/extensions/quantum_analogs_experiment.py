#!/usr/bin/env python3
"""
Quantum Analogs Experiment

This experiment explores quantum analogs in multi-agent systems.
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

from extensions.quantum_analogs.quantum_strategic_field import QuantumStrategicField
from extensions.quantum_analogs.quantum_agent import QuantumAgent
from core.strategic_field import StrategicField
from core.agent import Agent
from utils.visualization import create_figure, plot_field, plot_time_series

def run_experiment(output_dir=None, n_agents=100, grid_size=50, n_strategies=3, steps=500):
    """
    Run the quantum analogs experiment
    
    Parameters:
    - output_dir: Directory for output files
    - n_agents: Number of agents
    - grid_size: Size of the grid
    - n_strategies: Number of strategies
    - steps: Number of simulation steps
    
    Returns:
    - Dictionary of results
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./results/quantum_analogs_experiment_{timestamp}"
    
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
    logger.info(f"Starting quantum analogs experiment with {n_agents} agents, {n_strategies} strategies")
    
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
    
    # Initialize fields
    classical_field = StrategicField(grid_size, n_strategies)
    quantum_field = QuantumStrategicField(grid_size, n_strategies)
    
    # Initialize agents
    agents = []
    for i in range(n_agents):
        position = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
        strategy = np.zeros(n_strategies)
        strategy[np.random.randint(0, n_strategies)] = 1.0
        agents.append(Agent(i, position, strategy, payoff_matrix))
    
    logger.info(f"Initialized {n_agents} agents")
    
    # Run simulation
    classical_coherence = []
    quantum_coherence = []
    decision_history = [[] for _ in range(n_agents)]
    entanglement_effects = []
    
    logger.info(f"Running simulation for {steps} steps")
    
    for step in range(steps):
        if step % 50 == 0:
            logger.info(f"Simulation step {step}/{steps}")
        
        # Update fields
        classical_field.update(agents)
        quantum_field.update(agents)
        
        # Calculate coherence
        classical_coherence.append(classical_field.calculate_coherence())
        quantum_coherence.append(quantum_field.calculate_coherence())
        
        # Agents make decisions
        for i, agent in enumerate(agents):
            # Get field information at current position
            x, y = agent.position.astype(int)
            
            # Get classical and quantum probabilities
            classical_probs = classical_field.field[x, y]
            quantum_probs = quantum_field.get_probability_distribution(agent.position)
            
            # Make decisions
            classical_strategy = np.zeros(n_strategies)
            classical_strategy[np.argmax(classical_probs)] = 1.0
            
            quantum_strategy = np.zeros(n_strategies)
            quantum_strategy[np.argmax(quantum_probs)] = 1.0
            
            # Record decision
            decision_history[i].append(quantum_strategy)
            
            # Apply quantum decision
            agent.strategy = quantum_strategy
        
        # Calculate entanglement effects
        if hasattr(quantum_field, 'calculate_entanglement'):
            entanglement_effects.append(quantum_field.calculate_entanglement())
    
    logger.info(f"Simulation completed in {steps} steps")
    
    # Create figures
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Coherence comparison
    ax = axes[0, 0]
    ax.plot(classical_coherence, label='Classical', linewidth=2)
    ax.plot(quantum_coherence, label='Quantum', linewidth=2)
    ax.set_title('Coherence Comparison', fontsize=14)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Coherence', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Quantum advantage
    ax = axes[0, 1]
    advantage = np.array(quantum_coherence) - np.array(classical_coherence)
    ax.plot(advantage, color='purple', linewidth=2)
    ax.set_title('Quantum Advantage', fontsize=14)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Advantage (Q - C)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 3: Strategy distribution
    ax = axes[1, 0]
    
    # Calculate strategy frequencies
    strategy_counts = np.zeros((steps, n_strategies))
    for agent_history in decision_history:
        for t, strategy in enumerate(agent_history[:steps]):
            if isinstance(strategy, np.ndarray):
                # For mixed strategies, add fractional counts
                strategy_counts[t] += strategy
            else:
                # For pure strategies, add 1 to the corresponding strategy
                strategy_counts[t, strategy] += 1
    
    # Normalize
    strategy_frequencies = strategy_counts / n_agents
    
    # Plot stacked area chart
    x = range(steps)
    y_bottom = np.zeros(steps)
    
    for s in range(n_strategies):
        ax.fill_between(x, y_bottom, y_bottom + strategy_frequencies[:, s], 
                        alpha=0.7, label=f'Strategy {s+1}')
        y_bottom += strategy_frequencies[:, s]
    
    ax.set_title('Strategy Frequencies', fontsize=14)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend(loc='upper right')
    ax.set_title('Strategy Frequencies', fontsize=14)
    
    # Plot 4: Entanglement effects
    ax = axes[1, 1]
    if entanglement_effects:
        ax.plot(entanglement_effects, linewidth=2)
        ax.set_title('Entanglement Correlation', fontsize=14)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Average Correlation', fontsize=12)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No entanglement data available', ha='center', va='center', fontsize=12)
        ax.set_title('Entanglement Effects', fontsize=14)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "manuscript_figure.png"))
    
    logger.info(f"Experiment completed. Results saved to {output_dir}")
    
    # Export raw data for analysis
    data_export = {
        'coherence_comparison': {
            'classical_coherence': classical_coherence,
            'quantum_coherence': quantum_coherence
        },
        'decision_history': [
            [decision.tolist() if isinstance(decision, np.ndarray) else decision for decision in history]
            if isinstance(history, list) else []
            for history in decision_history
        ],
        'entanglement_effects': entanglement_effects,
        'interference_pattern': None  # Will be added if available
    }
    
    # Try to extract interference pattern data if available
    try:
        if hasattr(quantum_field, 'interference_history') and quantum_field.interference_history:
            # Take the last interference pattern
            interference = quantum_field.interference_history[-1]
            data_export['interference_pattern'] = interference.tolist() if isinstance(interference, np.ndarray) else interference
    except Exception as e:
        logger.warning(f"Could not extract interference pattern: {e}")
    
    # Save raw data directly
    import json
    try:
        raw_data_path = os.path.join(output_dir, "raw_data.json")
        logger.info(f"Attempting to save raw data to {raw_data_path}")
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {}
        for key, value in data_export.items():
            if isinstance(value, dict):
                serializable_data[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        serializable_data[key][k] = v.tolist()
                    else:
                        serializable_data[key][k] = v
            elif isinstance(value, np.ndarray):
                serializable_data[key] = value.tolist()
            else:
                serializable_data[key] = value
        
        with open(raw_data_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        # Verify the file was created
        if os.path.exists(raw_data_path):
            logger.info(f"Raw data successfully exported to {raw_data_path}")
        else:
            logger.error(f"Failed to create raw data file at {raw_data_path}")
    except Exception as e:
        logger.error(f"Error exporting raw data: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return {
        'classical_coherence': classical_coherence,
        'quantum_coherence': quantum_coherence,
        'decision_history': decision_history,
        'entanglement_effects': entanglement_effects,
        'output_dir': output_dir,
        'data_export': data_export
    }

if __name__ == "__main__":
    run_experiment()
