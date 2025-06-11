#!/usr/bin/env python3
"""
Run All Experiments with Data Collection

This script runs all extension experiments and collects the data for analysis.
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
    output_dir = f"./results/all_experiments_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create analysis directory
    os.makedirs(os.path.join(output_dir, "analysis"), exist_ok=True)
    
    return output_dir

def run_all_experiments():
    """
    Run all experiments and collect data
    
    Returns:
    - output_dir: Path to output directory
    """
    # Set up output directory
    output_dir = setup_output_directory()
    
    # Set up logging
    log_file = os.path.join(output_dir, "all_experiments.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting all experiments with data collection")
    
    # Run infinite population experiment
    try:
        logger.info("Running infinite population experiment")
        from experiments.extensions.infinite_population_experiment import run_experiment as run_infinite_population
        infinite_population_results = run_infinite_population()
        logger.info(f"Infinite population experiment completed. Results saved to {infinite_population_results['output_dir']}")
        
        # Copy raw data to analysis directory
        import shutil
        src_file = os.path.join(infinite_population_results['output_dir'], "raw_data.json")
        dst_file = os.path.join(output_dir, "analysis", "infinite_population_data.json")
        shutil.copy(src_file, dst_file)
    except Exception as e:
        logger.error(f"Error in infinite population experiment: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Run optimization experiment
    try:
        logger.info("Running optimization experiment")
        from experiments.extensions.optimization_experiment import run_experiment as run_optimization
        optimization_results = run_optimization()
        logger.info(f"Optimization experiment completed. Results saved to {optimization_results['output_dir']}")
        
        # Copy raw data to analysis directory
        import shutil
        src_file = os.path.join(optimization_results['output_dir'], "raw_data.json")
        dst_file = os.path.join(output_dir, "analysis", "optimization_data.json")
        shutil.copy(src_file, dst_file)
    except Exception as e:
        logger.error(f"Error in optimization experiment: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Run quantum analogs experiment
    try:
        logger.info("Running quantum analogs experiment")
        from experiments.extensions.quantum_analogs_experiment import run_experiment as run_quantum_analogs
        quantum_analogs_results = run_quantum_analogs()
        logger.info(f"Quantum analogs experiment completed. Results saved to {quantum_analogs_results['output_dir']}")
        
        # Copy raw data to analysis directory
        import shutil
        src_file = os.path.join(quantum_analogs_results['output_dir'], "raw_data.json")
        dst_file = os.path.join(output_dir, "analysis", "quantum_analogs_data.json")
        shutil.copy(src_file, dst_file)
    except Exception as e:
        logger.error(f"Error in quantum analogs experiment: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Run consciousness experiment
    try:
        logger.info("Running consciousness experiment")
        from experiments.extensions.consciousness_experiment import run_experiment as run_consciousness
        consciousness_results = run_consciousness()
        logger.info(f"Consciousness experiment completed. Results saved to {consciousness_results['output_dir']}")
        
        # Copy raw data to analysis directory
        import shutil
        src_file = os.path.join(consciousness_results['output_dir'], "raw_data.json")
        dst_file = os.path.join(output_dir, "analysis", "consciousness_data.json")
        shutil.copy(src_file, dst_file)
    except Exception as e:
        logger.error(f"Error in consciousness experiment: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info(f"All experiments completed. Results saved to {output_dir}")
    
    return output_dir

if __name__ == "__main__":
    run_all_experiments()
