#!/usr/bin/env python3
"""
Run All Extensions

This script runs all extension experiments to demonstrate the enhanced capabilities
of the MMAI system.
"""

import os
import sys
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_all_extensions():
    """
    Run all extension experiments
    """
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./results/all_extensions_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, "all_extensions.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Running all extension experiments")
    
    # Run infinite population experiment
    logger.info("Running infinite population experiment")
    sys.path.insert(0, os.path.abspath('.'))
    from experiments.extensions.infinite_population_experiment import run_experiment as run_infinite_population
    infinite_population_dir = os.path.join(output_dir, "infinite_population")
    infinite_population_results = run_infinite_population(output_dir=infinite_population_dir)
    logger.info(f"Infinite population experiment completed. Results saved to {infinite_population_dir}")
    
    # Run optimization experiment
    logger.info("Running optimization experiment")
    from experiments.extensions.optimization_experiment import run_experiment as run_optimization
    optimization_dir = os.path.join(output_dir, "optimization")
    optimization_results = run_optimization(output_dir=optimization_dir)
    logger.info(f"Optimization experiment completed. Results saved to {optimization_dir}")
    
    # Run quantum analogs experiment
    logger.info("Running quantum analogs experiment")
    from experiments.extensions.quantum_analogs_experiment import run_experiment as run_quantum_analogs
    quantum_analogs_dir = os.path.join(output_dir, "quantum_analogs")
    quantum_analogs_results = run_quantum_analogs(output_dir=quantum_analogs_dir)
    logger.info(f"Quantum analogs experiment completed. Results saved to {quantum_analogs_dir}")
    
    # Run consciousness experiment
    logger.info("Running consciousness experiment")
    from experiments.extensions.consciousness_experiment import run_experiment as run_consciousness
    consciousness_dir = os.path.join(output_dir, "consciousness")
    consciousness_results = run_consciousness(output_dir=consciousness_dir)
    logger.info(f"Consciousness experiment completed. Results saved to {consciousness_dir}")
    
    logger.info(f"All extension experiments completed. Results saved to {output_dir}")
    
    return {
        'output_dir': output_dir,
        'infinite_population_results': infinite_population_results,
        'optimization_results': optimization_results,
        'quantum_analogs_results': quantum_analogs_results,
        'consciousness_results': consciousness_results
    }

if __name__ == "__main__":
    run_all_extensions()
