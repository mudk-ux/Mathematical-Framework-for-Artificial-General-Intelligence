#!/usr/bin/env python3
"""
Full test script for the LLM-enhanced MMAI system
"""

import os
import sys
import logging
import argparse
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import simulation components
from simulation.llm_unified_simulation import LLMUnifiedSimulation
from core.environment_system import EnvironmentType

def setup_logging(log_level=logging.INFO):
    """Setup logging for the simulation"""
    # Create logger
    logger = logging.getLogger("llm_full_test")
    logger.setLevel(log_level)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(ch)
    
    return logger

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run LLM-enhanced MMAI simulation')
    
    # Basic simulation parameters
    parser.add_argument('--grid-size', type=int, default=10, help='Size of the grid')
    parser.add_argument('--n-agents', type=int, default=5, help='Number of agents')
    parser.add_argument('--max-steps', type=int, default=10, help='Maximum number of steps')
    parser.add_argument('--env-type', type=str, default='static', choices=['static', 'dynamic', 'fractal'], 
                        help='Environment type')
    
    # LLM parameters
    parser.add_argument('--model-id', type=str, default='anthropic.claude-3-sonnet-20240229-v1:0', 
                        help='LLM model ID')
    parser.add_argument('--region', type=str, default='us-east-1', help='AWS region')
    parser.add_argument('--llm-agent-ratio', type=float, default=0.2, 
                        help='Ratio of LLM agents to total agents')
    
    # Output parameters
    parser.add_argument('--results-dir', type=str, default='./results', 
                        help='Directory to save results')
    parser.add_argument('--log-level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                        help='Logging level')
    
    return parser.parse_args()

def main():
    """Main entry point"""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(getattr(logging, args.log_level))
    logger.info("Starting full LLM simulation test")
    
    # Map environment type string to enum
    env_type_map = {
        'static': EnvironmentType.STATIC,
        'dynamic': EnvironmentType.PERIODIC,  # Use PERIODIC for dynamic
        'fractal': EnvironmentType.CHAOTIC    # Use CHAOTIC for fractal
    }
    
    # Create configuration
    config = {
        'grid_size': args.grid_size,
        'n_agents': args.n_agents,
        'max_agents': args.n_agents * 2,  # Allow for population growth
        'n_strategies': 3,  # Default number of strategies
        'dt': 0.01,  # Small time step for fine-grained simulation
        't_scale': 10,  # Scale for agent time
        'T_scale': 5,  # Scale for system time
        'env_type': env_type_map[args.env_type],
        'enable_reproduction': False,  # Disable reproduction for simplicity
        'enable_resources': True,  # Enable resource dynamics
        'enable_dynamic_population': False,  # Disable dynamic population for simplicity
        'max_steps': args.max_steps,
        'equilibrium_threshold': 0.1,  # Threshold for Nash equilibrium
        'data_dir': args.results_dir,
        'experiment_name': f"llm_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'model_id': args.model_id,
        'region': args.region,
        'llm_agent_ratio': args.llm_agent_ratio,
        'logger': logger
    }
    
    # Create and run simulation
    logger.info("Creating simulation")
    simulation = LLMUnifiedSimulation(**config)
    
    logger.info("Running simulation")
    simulation.run()
    
    logger.info("Collecting results")
    results = simulation.collect_results()
    
    logger.info(f"Simulation completed. Results saved to {results.get('experiment_dir', 'unknown')}")
    
    return results, simulation

if __name__ == "__main__":
    main()
