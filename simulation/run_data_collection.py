#!/usr/bin/env python3
"""
Data Collection Script for MMAI System Extensions

This script runs all extension experiments with data collection enabled,
saving both raw numerical data and visualizations for rigorous analysis.
"""

import os
import sys
import json
import logging
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_output_directory():
    """
    Set up output directory for data collection
    
    Returns:
    - output_dir: Path to output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./results/data_collection_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for each experiment
    for exp in ["infinite_population", "optimization", "quantum_analogs", "consciousness"]:
        os.makedirs(os.path.join(output_dir, exp), exist_ok=True)
    
    # Create analysis directory
    os.makedirs(os.path.join(output_dir, "analysis"), exist_ok=True)
    
    return output_dir

def run_infinite_population_experiment(output_dir):
    """
    Run infinite population experiment with data collection
    
    Parameters:
    - output_dir: Path to output directory
    
    Returns:
    - data: Dictionary of collected data
    """
    from experiments.extensions.infinite_population_experiment import run_experiment
    
    # Run experiment
    exp_output_dir = os.path.join(output_dir, "infinite_population")
    results = run_experiment(output_dir=exp_output_dir)
    
    # Extract and save raw data
    data = {
        'nash_distance': {
            'mean_field': results.get('mean_field', {}).get('nash_distance', []),
            'finite': results.get('finite', {}).get('nash_distance_history', [])
        },
        'strategy_history': {
            'mean_field': [s.tolist() if isinstance(s, np.ndarray) else s for s in results.get('mean_field', {}).get('strategy_history', [])]
        },
        'continuous': {
            'time': results.get('continuous', {}).get('time', []).tolist() if isinstance(results.get('continuous', {}).get('time', []), np.ndarray) else results.get('continuous', {}).get('time', []),
            'nash_distance': results.get('continuous', {}).get('nash_distance_history', [])
        },
        'scaling': {
            'population_sizes': results.get('scaling', {}).get('population_sizes', []).tolist() if isinstance(results.get('scaling', {}).get('population_sizes', []), np.ndarray) else results.get('scaling', {}).get('population_sizes', []),
            'convergence_times': [],
            'final_nash_distances': []
        }
    }
    
    # Extract scaling data
    for size in data['scaling']['population_sizes']:
        if size in results.get('scaling', {}).get('finite_results', {}):
            data['scaling']['convergence_times'].append(
                results['scaling']['finite_results'][size].get('convergence_time', 0)
            )
            data['scaling']['final_nash_distances'].append(
                results['scaling']['finite_results'][size].get('final_nash_distance', 0)
            )
    
    # Save raw data
    with open(os.path.join(exp_output_dir, "raw_data.json"), 'w') as f:
        json.dump(data, f, indent=2)
    
    return data

def run_optimization_experiment(output_dir):
    """
    Run optimization experiment with data collection
    
    Parameters:
    - output_dir: Path to output directory
    
    Returns:
    - data: Dictionary of collected data
    """
    from experiments.extensions.optimization_experiment import run_experiment
    
    # Run experiment
    exp_output_dir = os.path.join(output_dir, "optimization")
    results = run_experiment(output_dir=exp_output_dir)
    
    # Extract and save raw data
    data = {
        'field_comparison': {
            'original_times': results.get('original_times', []),
            'sparse_times': results.get('sparse_times', []),
            'dense_times': results.get('dense_times', []),
            'original_coherence': results.get('original_coherence', []),
            'sparse_coherence': results.get('sparse_coherence', []),
            'dense_coherence': results.get('dense_coherence', [])
        },
        'spatial_comparison': {
            'naive_times': results.get('naive_times', []),
            'spatial_times': results.get('spatial_times', [])
        },
        'performance_metrics': {
            'sparse_speedup': results.get('sparse_speedup', 0),
            'dense_speedup': results.get('dense_speedup', 0),
            'spatial_speedup': results.get('spatial_speedup', 0),
            'memory_reduction': results.get('memory_reduction', 0)
        }
    }
    
    # Save raw data
    with open(os.path.join(exp_output_dir, "raw_data.json"), 'w') as f:
        json.dump(data, f, indent=2)
    
    return data

def run_quantum_analogs_experiment(output_dir):
    """
    Run quantum analogs experiment with data collection
    
    Parameters:
    - output_dir: Path to output directory
    
    Returns:
    - data: Dictionary of collected data
    """
    from experiments.extensions.quantum_analogs_experiment import run_experiment
    
    # Run experiment
    exp_output_dir = os.path.join(output_dir, "quantum_analogs")
    results = run_experiment(output_dir=exp_output_dir)
    
    # Extract and save raw data
    data = {
        'coherence_comparison': {
            'classical_coherence': results.get('classical_coherence', []),
            'quantum_coherence': results.get('quantum_coherence', [])
        },
        'decision_history': [
            [decision.tolist() if isinstance(decision, np.ndarray) else decision for decision in history]
            if isinstance(history, list) else []
            for history in results.get('decision_history', [])
        ],
        'entanglement_effects': results.get('entanglement_effects', [])
    }
    
    # Save raw data
    with open(os.path.join(exp_output_dir, "raw_data.json"), 'w') as f:
        json.dump(data, f, indent=2)
    
    return data

def run_consciousness_experiment(output_dir):
    """
    Run consciousness experiment with data collection
    
    Parameters:
    - output_dir: Path to output directory
    
    Returns:
    - data: Dictionary of collected data
    """
    from experiments.extensions.consciousness_experiment import run_experiment
    
    # Run experiment
    exp_output_dir = os.path.join(output_dir, "consciousness")
    results = run_experiment(output_dir=exp_output_dir)
    
    # Extract and save raw data
    data = {
        'integration_metrics': [
            {
                'temporal_integration': metrics.get('temporal_integration', 0),
                'spatial_integration': metrics.get('spatial_integration', 0),
                'phi': metrics.get('phi', 0),
                'self_reference': metrics.get('self_reference', 0),
                'resonance_integration': metrics.get('resonance_integration', 0),
                'overall_integration': metrics.get('overall_integration', 0)
            }
            for metrics in results.get('integration_metrics', [])
        ],
        'coherence_history': results.get('coherence_history', []),
        'self_reference_metrics': {
            'self_reference_level': results.get('self_reference_metrics', {}).get('self_reference_level', 0),
            'n_meta_frames': results.get('self_reference_metrics', {}).get('n_meta_frames', 0),
            'n_agent_models': results.get('self_reference_metrics', {}).get('n_agent_models', 0),
            'history': results.get('self_reference_metrics', {}).get('history', [])
        }
    }
    
    # Save raw data
    with open(os.path.join(exp_output_dir, "raw_data.json"), 'w') as f:
        json.dump(data, f, indent=2)
    
    return data

def run_all_experiments():
    """
    Run all experiments with data collection
    
    Returns:
    - all_data: Dictionary of all collected data
    - output_dir: Path to output directory
    """
    # Set up output directory
    output_dir = setup_output_directory()
    
    # Set up logging
    log_file = os.path.join(output_dir, "data_collection.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting data collection for all experiments")
    
    # Run experiments
    all_data = {}
    
    try:
        logger.info("Running infinite population experiment")
        all_data['infinite_population'] = run_infinite_population_experiment(output_dir)
        logger.info("Infinite population experiment completed")
    except Exception as e:
        logger.error(f"Error in infinite population experiment: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    try:
        logger.info("Running optimization experiment")
        all_data['optimization'] = run_optimization_experiment(output_dir)
        logger.info("Optimization experiment completed")
    except Exception as e:
        logger.error(f"Error in optimization experiment: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    try:
        logger.info("Running quantum analogs experiment")
        all_data['quantum_analogs'] = run_quantum_analogs_experiment(output_dir)
        logger.info("Quantum analogs experiment completed")
    except Exception as e:
        logger.error(f"Error in quantum analogs experiment: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    try:
        logger.info("Running consciousness experiment")
        all_data['consciousness'] = run_consciousness_experiment(output_dir)
        logger.info("Consciousness experiment completed")
    except Exception as e:
        logger.error(f"Error in consciousness experiment: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Save all data
    with open(os.path.join(output_dir, "all_data.json"), 'w') as f:
        json.dump(all_data, f, indent=2)
    
    logger.info(f"Data collection completed. Results saved to {output_dir}")
    
    return all_data, output_dir

if __name__ == "__main__":
    run_all_experiments()
