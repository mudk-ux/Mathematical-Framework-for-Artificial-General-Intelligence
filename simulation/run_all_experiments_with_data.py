#!/usr/bin/env python3
"""
Run All Experiments with Comprehensive Data Collection

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

def convert_to_serializable(data):
    """
    Convert data to JSON-serializable format
    
    Parameters:
    - data: Data to convert
    
    Returns:
    - serializable_data: JSON-serializable data
    """
    if isinstance(data, dict):
        return {k: convert_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    else:
        return data

def save_raw_data(data, output_path):
    """
    Save raw data to a JSON file
    
    Parameters:
    - data: Data to save
    - output_path: Path to save the data
    
    Returns:
    - success: True if the data was saved successfully, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert data to JSON-serializable format
        serializable_data = convert_to_serializable(data)
        
        # Save data
        with open(output_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        # Verify the file was created
        if os.path.exists(output_path):
            print(f"Raw data successfully exported to {output_path}")
            return True
        else:
            print(f"Failed to create raw data file at {output_path}")
            return False
    except Exception as e:
        print(f"Error exporting raw data: {e}")
        import traceback
        print(traceback.format_exc())
        return False

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
    logger.info("Starting all experiments with comprehensive data collection")
    
    # Run infinite population experiment
    try:
        logger.info("Running infinite population experiment")
        from experiments.extensions.infinite_population_experiment import run_experiment as run_infinite_population
        infinite_population_results = run_infinite_population()
        logger.info(f"Infinite population experiment completed. Results saved to {infinite_population_results['output_dir']}")
        
        # Extract and save raw data
        data_path = os.path.join(output_dir, "analysis", "infinite_population_data.json")
        save_raw_data(infinite_population_results, data_path)
        logger.info(f"Infinite population data saved to {data_path}")
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
        
        # Extract and save raw data
        data_path = os.path.join(output_dir, "analysis", "optimization_data.json")
        save_raw_data(optimization_results, data_path)
        logger.info(f"Optimization data saved to {data_path}")
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
        
        # Extract and save raw data
        data_path = os.path.join(output_dir, "analysis", "quantum_analogs_data.json")
        save_raw_data(quantum_analogs_results, data_path)
        logger.info(f"Quantum analogs data saved to {data_path}")
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
        
        # Extract and save raw data
        data_path = os.path.join(output_dir, "analysis", "consciousness_data.json")
        save_raw_data(consciousness_results, data_path)
        logger.info(f"Consciousness data saved to {data_path}")
    except Exception as e:
        logger.error(f"Error in consciousness experiment: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Create combined data file
    try:
        combined_data = {
            "infinite_population": infinite_population_results,
            "optimization": optimization_results,
            "quantum_analogs": quantum_analogs_results,
            "consciousness": consciousness_results
        }
        
        combined_data_path = os.path.join(output_dir, "analysis", "combined_data.json")
        save_raw_data(combined_data, combined_data_path)
        logger.info(f"Combined data saved to {combined_data_path}")
    except Exception as e:
        logger.error(f"Error creating combined data: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info(f"All experiments completed. Results saved to {output_dir}")
    
    return output_dir

if __name__ == "__main__":
    run_all_experiments()
