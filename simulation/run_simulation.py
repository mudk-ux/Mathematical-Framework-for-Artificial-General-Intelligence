#!/usr/bin/env python3
"""
Main entry point for running MMAI simulations

This script provides a command-line interface for running various experiments
and simulations with the unified MMAI system.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import simulation components
from simulation.unified_simulation import UnifiedSimulation, run_unified_simulation
from core.environment_system import EnvironmentType

# Import experiments
from experiments.strategic_fields_experiment import run_strategic_fields_experiment
from experiments.nash_equilibrium_experiment import run_nash_equilibrium_experiment

def setup_logging(log_level=logging.INFO):
    """Setup logging for the simulation"""
    # Create logger
    logger = logging.getLogger("mmai_simulation")
    logger.setLevel(log_level)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    
    # Create file handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs("logs", exist_ok=True)
    fh = logging.FileHandler(f"logs/simulation_{timestamp}.log")
    fh.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Run MMAI simulations')
    
    # General arguments
    parser.add_argument('--experiment', type=str, default='default',
                        choices=['default', 'strategic_fields', 'nash_equilibrium', 'hypersensitive', 'temporal'],
                        help='Experiment to run')
    parser.add_argument('--results-dir', type=str, default='./results',
                        help='Directory to store results')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    
    # Simulation parameters
    parser.add_argument('--grid-size', type=int, default=50,
                        help='Size of the environment grid')
    parser.add_argument('--n-agents', type=int, default=50,
                        help='Initial number of agents')
    parser.add_argument('--max-agents', type=int, default=200,
                        help='Maximum number of agents')
    parser.add_argument('--n-strategies', type=int, default=3,
                        help='Number of strategies')
    parser.add_argument('--dt', type=float, default=0.01,
                        help='Time step size')
    parser.add_argument('--t-scale', type=int, default=50,
                        help='Number of dt steps in one t step')
    parser.add_argument('--T-scale', type=int, default=20,
                        help='Number of t steps in one T step')
    parser.add_argument('--env-type', type=str, default='STATIC',
                        choices=['STATIC', 'PERIODIC', 'CHAOTIC', 'SHOCK'],
                        help='Type of environment dynamics')
    parser.add_argument('--max-steps', type=int, default=1000,
                        help='Maximum number of simulation steps')
    parser.add_argument('--no-reproduction', action='store_true',
                        help='Disable agent reproduction')
    parser.add_argument('--no-resources', action='store_true',
                        help='Disable resource dynamics')
    parser.add_argument('--no-dynamic-population', action='store_true',
                        help='Disable dynamic population growth')
    
    return parser.parse_args()

def run_default_simulation(args, logger):
    """Run a default simulation with the specified parameters"""
    logger.info("Running default simulation")
    
    # Map environment type string to enum
    env_type_map = {
        'STATIC': EnvironmentType.STATIC,
        'PERIODIC': EnvironmentType.PERIODIC,
        'CHAOTIC': EnvironmentType.CHAOTIC,
        'SHOCK': EnvironmentType.SHOCK
    }
    
    # Create configuration
    config = {
        'grid_size': args.grid_size,
        'n_agents': args.n_agents,
        'max_agents': args.max_agents,
        'n_strategies': args.n_strategies,
        'dt': args.dt,
        't_scale': args.t_scale,
        'T_scale': args.T_scale,
        'env_type': env_type_map[args.env_type],
        'enable_reproduction': not args.no_reproduction,
        'enable_resources': not args.no_resources,
        'enable_dynamic_population': not args.no_dynamic_population,
        'max_steps': args.max_steps,
        'equilibrium_threshold': 0.1,
        'data_dir': args.results_dir,
        'experiment_name': f"default_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'log_level': getattr(logging, args.log_level)
    }
    
    # Run simulation
    results, sim = run_unified_simulation(config)
    
    logger.info(f"Simulation completed. Results saved to {results['experiment_dir']}")
    
    return results, sim

def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(log_level)
    
    logger.info(f"Starting MMAI simulation with experiment: {args.experiment}")
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Run the specified experiment
    if args.experiment == 'default':
        results, sim = run_default_simulation(args, logger)
    elif args.experiment == 'strategic_fields':
        results = run_strategic_fields_experiment(args.results_dir, log_level)
    elif args.experiment == 'nash_equilibrium':
        results = run_nash_equilibrium_experiment(args.results_dir, log_level)
    elif args.experiment == 'hypersensitive':
        logger.error("Hypersensitive points experiment not yet implemented")
        sys.exit(1)
    elif args.experiment == 'temporal':
        logger.error("Temporal resonance experiment not yet implemented")
        sys.exit(1)
    else:
        logger.error(f"Unknown experiment: {args.experiment}")
        sys.exit(1)
    
    logger.info("Simulation completed successfully")

if __name__ == "__main__":
    main()
