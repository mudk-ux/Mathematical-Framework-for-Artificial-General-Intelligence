#!/usr/bin/env python3
"""
Unified Simulation for the MMAI system

This module implements the main simulation framework that integrates all components.
"""

import os
import sys
import time
import numpy as np
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from core.agent import Agent
from core.strategic_field import StrategicField
from core.fractal_time_manager import FractalTimeManager
from core.memory_system import MemorySystem
from core.nash_validator import NashValidator
from core.environment_system import EnvironmentSystem, EnvironmentType
from core.resource_system import ResourceSystem
from simulation.data_manager import DataManager

class UnifiedSimulation:
    """
    Unified simulation that integrates all components of the MMAI system
    
    This simulation framework combines:
    - Agents with memory and strategic decision-making
    - Strategic field for spatial coordination
    - Fractal time architecture for multi-scale temporal dynamics
    - Memory system (IRN) for individual and collective memory
    - Nash validator for equilibrium analysis
    - Environment system for environmental dynamics
    - Resource system for resource dynamics
    """
    def __init__(self, 
                 grid_size=50,
                 n_agents=50,
                 max_agents=200,
                 n_strategies=3,
                 dt=0.01,
                 t_scale=50,
                 T_scale=20,
                 env_type=EnvironmentType.STATIC,
                 enable_reproduction=True,
                 enable_resources=True,
                 enable_dynamic_population=True,
                 max_steps=3000,
                 equilibrium_threshold=0.1,
                 data_dir="./results",
                 experiment_name=None,
                 log_level=logging.INFO):
        """
        Initialize the unified simulation
        
        Parameters:
        - grid_size: Size of the environment grid
        - n_agents: Initial number of agents
        - max_agents: Maximum number of agents allowed
        - n_strategies: Number of strategies
        - dt: Time step size
        - t_scale: Number of dt steps in one t step
        - T_scale: Number of t steps in one T step
        - env_type: Type of environment dynamics
        - enable_reproduction: Whether to enable agent reproduction
        - enable_resources: Whether to enable resource dynamics
        - enable_dynamic_population: Whether to enable dynamic population growth
        - max_steps: Maximum number of simulation steps
        - equilibrium_threshold: Threshold for Nash equilibrium
        - data_dir: Directory for storing simulation data
        - experiment_name: Name of the experiment
        - log_level: Logging level
        """
        # Setup logging
        self.logger = self._setup_logging(log_level)
        self.logger.info("Initializing unified simulation")
        
        # Store parameters
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.max_agents = max_agents
        self.n_strategies = n_strategies
        self.dt = dt
        self.t_scale = t_scale
        self.T_scale = T_scale
        self.env_type = env_type
        self.enable_reproduction = enable_reproduction
        self.enable_resources = enable_resources
        self.enable_dynamic_population = enable_dynamic_population
        self.max_steps = max_steps
        self.equilibrium_threshold = equilibrium_threshold
        
        # Initialize components
        self.initialize_components()
        
        # Initialize data manager
        if experiment_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_name = f"unified_simulation_{timestamp}"
        
        self.data_manager = DataManager(
            base_dir=data_dir,
            experiment_name=experiment_name,
            logger=self.logger
        )
        
        # Save configuration
        self.data_manager.save_experiment_config({
            'grid_size': grid_size,
            'n_agents': n_agents,
            'max_agents': max_agents,
            'n_strategies': n_strategies,
            'dt': dt,
            't_scale': t_scale,
            'T_scale': T_scale,
            'env_type': env_type.name if hasattr(env_type, 'name') else str(env_type),
            'enable_reproduction': enable_reproduction,
            'enable_resources': enable_resources,
            'enable_dynamic_population': enable_dynamic_population,
            'max_steps': max_steps,
            'equilibrium_threshold': equilibrium_threshold
        })
        
        # Initialize tracking variables
        self.current_step = 0
    
    def _setup_logging(self, log_level):
        """Setup logging for the simulation"""
        logger = logging.getLogger("unified_simulation")
        logger.setLevel(log_level)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(ch)
        
        return logger
    
    def initialize_components(self):
        """Initialize all simulation components"""
        self.logger.info("Initializing simulation components")
        
        # Initialize fractal time manager
        self.time_manager = FractalTimeManager(
            dt=self.dt,
            t_scale=self.t_scale,
            T_scale=self.T_scale,
            logger=self.logger
        )
        
        # Initialize strategic field
        self.field = StrategicField(
            grid_size=self.grid_size,
            n_strategies=self.n_strategies,
            logger=self.logger
        )
        
        # Initialize memory system (IRN)
        self.memory_system = MemorySystem(
            update_frequency=1.0/self.dt,  # f = 1/dt
            logger=self.logger
        )
        
        # Initialize Nash validator
        self.nash_validator = NashValidator(
            n_strategies=self.n_strategies,
            equilibrium_threshold=self.equilibrium_threshold,
            logger=self.logger
        )
        
        # Initialize environment
        self.environment = EnvironmentSystem(
            grid_size=self.grid_size,
            env_type=self.env_type,
            logger=self.logger
        )
        
        # Initialize resource system if enabled
        if self.enable_resources:
            self.resource_system = ResourceSystem(
                grid_size=self.grid_size,
                initial_density=0.3,
                growth_rate=0.05,
                logger=self.logger
            )
        
        # Initialize agents
        self.agents = []
        for i in range(self.n_agents):
            # Random position
            position = np.random.randint(0, self.grid_size, size=2)
            
            # Create agent
            agent = Agent(
                agent_id=i,
                position=position,
                n_strategies=self.n_strategies,
                logger=self.logger
            )
            self.agents.append(agent)
            
            # Initialize agent memory in IRN
            self.memory_system.initialize_agent_memory(i, self.n_strategies)
    
    def step(self):
        """
        Run one simulation step
        
        Returns:
        - metrics: Dictionary of metrics for this step
        """
        self.current_step += 1
        
        # Update fractal time manager
        event_type = self.time_manager.step()
        
        # Update environment
        self.environment.update(dt=self.dt, current_time=self.current_step)
        
        # Update resources if enabled
        if self.enable_resources:
            self.resource_system.update(
                agents=self.agents,
                dt=self.dt,
                current_time=self.current_step,
                environment=self.environment
            )
        
        # Update strategic field
        if self.enable_resources:
            coherence = self.field.update(
                agents=self.agents,
                resources=self.resource_system,
                dt=self.dt
            )
        else:
            coherence = self.field.update(
                agents=self.agents,
                dt=self.dt
            )
        
        # Update memory system (IRN)
        self.memory_system.update(dt=self.dt)
        
        # Update agents
        self._update_agents()
        
        # Calculate Nash equilibrium
        nash_distance, best_responses = self.nash_validator.calculate_nash_distance(
            agents=self.agents,
            population_size=len(self.agents)
        )
        
        # Check for growth-proportional equilibrium
        is_proportional, proportionality = self.nash_validator.calculate_growth_proportional_equilibrium()
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            nash_distance=nash_distance,
            coherence=coherence,
            is_proportional=is_proportional,
            proportionality=proportionality
        )
        
        # Record metrics
        self.data_manager.record_metrics(metrics, self.current_step)
        
        # Handle dynamic population if enabled
        if self.enable_dynamic_population:
            self._update_population(nash_distance)
        
        # Save checkpoint periodically
        if self.current_step % 100 == 0 or self.current_step == self.max_steps:
            self.data_manager.save_checkpoint(self, self.current_step)
        
        # Log progress
        if self.current_step % 100 == 0:
            self.logger.info(f"Step {self.current_step}/{self.max_steps} - Nash distance: {nash_distance:.4f}, Agents: {len(self.agents)}")
        
        return metrics
    
    def _update_agents(self):
        """Update all agents"""
        # List to track agents that need to be removed
        agents_to_remove = []
        
        # List to track new agents from reproduction
        new_agents = []
        
        for agent in self.agents:
            # Get agent state for memory
            agent_state = agent.get_state()
            
            # Move agent
            if self.enable_resources:
                agent.move(
                    grid_size=self.grid_size,
                    strategic_field=self.field,
                    resource_field=self.resource_system
                )
            else:
                agent.move(
                    grid_size=self.grid_size,
                    strategic_field=self.field
                )
            
            # Get field influence at agent's position
            field_influence = self.field.get_strategy_at_position(agent.position)
            
            # Get IRN influence from memory
            irn_influence = self.memory_system.get_strategy_influence(agent.agent_id, agent_state)
            
            # Get best response from Nash validator
            best_response = self.nash_validator.best_response_history[-1].get(agent.agent_id) if self.nash_validator.best_response_history else None
            
            # Get resource level at agent's position
            resource_level = 0.0
            if self.enable_resources:
                resource_level = self.resource_system.get_resources_at(agent.position)
            
            # Update agent strategy
            agent.update_strategy(
                field_influence=field_influence,
                irn_influence=irn_influence,
                best_response=best_response,
                resource_level=resource_level
            )
            
            # Make strategic decision
            decision = agent.make_decision()
            
            # Calculate outcome based on resources and environment
            outcome = resource_level
            if self.enable_resources:
                outcome = resource_level
            else:
                # If no resources, use environment state
                outcome = self.environment.get_state_at(agent.position)
            
            # Store experience in agent memory
            agent.add_to_memory(agent_state, outcome)
            
            # Store experience in IRN
            self.memory_system.store_experience(agent.agent_id, agent_state, outcome)
            
            # Handle reproduction if enabled
            if self.enable_reproduction and agent.can_reproduce():
                child, success = agent.reproduce()
                if success and len(self.agents) + len(new_agents) < self.max_agents:
                    # Assign proper ID to child
                    child.agent_id = len(self.agents) + len(new_agents)
                    new_agents.append(child)
                    
                    # Initialize child memory in IRN
                    self.memory_system.initialize_agent_memory(child.agent_id, self.n_strategies)
            
            # Check if agent is still alive
            if hasattr(agent, 'is_alive') and not agent.is_alive():
                agents_to_remove.append(agent)
        
        # Remove dead agents
        for agent in agents_to_remove:
            self.agents.remove(agent)
        
        # Add new agents
        self.agents.extend(new_agents)
    
    def _update_population(self, nash_distance):
        """
        Update population based on strategic stability
        
        Parameters:
        - nash_distance: Current Nash distance
        """
        # Check if we're at equilibrium
        is_equilibrium = nash_distance < self.equilibrium_threshold
        
        # If not at equilibrium and below max population, add new agents
        if not is_equilibrium and len(self.agents) < self.max_agents:
            # Add new agents proportional to how far we are from equilibrium
            distance_factor = min(1.0, nash_distance)
            num_new_agents = int(distance_factor * 5)  # Up to 5 new agents per step
            
            for i in range(num_new_agents):
                if len(self.agents) >= self.max_agents:
                    break
                
                # Random position
                position = np.random.randint(0, self.grid_size, size=2)
                
                # Create agent
                agent_id = len(self.agents)
                agent = Agent(
                    agent_id=agent_id,
                    position=position,
                    n_strategies=self.n_strategies,
                    logger=self.logger
                )
                self.agents.append(agent)
                
                # Initialize agent memory in IRN
                self.memory_system.initialize_agent_memory(agent_id, self.n_strategies)
    
    def _calculate_metrics(self, nash_distance, coherence, is_proportional, proportionality):
        """
        Calculate metrics for the current step
        
        Parameters:
        - nash_distance: Current Nash distance
        - coherence: Strategic field coherence
        - is_proportional: Whether equilibrium is proportional to growth
        - proportionality: Measure of proportionality
        
        Returns:
        - metrics: Dictionary of metrics
        """
        metrics = {
            'step': self.current_step,
            'population': len(self.agents),
            'nash_distance': nash_distance,
            'coherence': coherence,
            'is_equilibrium': nash_distance < self.equilibrium_threshold,
            'is_proportional': is_proportional,
            'proportionality': proportionality,
            'avg_energy': np.mean([agent.energy for agent in self.agents]) if hasattr(self.agents[0], 'energy') else 0,
            'hypersensitive_count': sum(1 for agent in self.agents if hasattr(agent, 'check_hypersensitivity') and agent.check_hypersensitivity())
        }
        
        # Add resource metrics if enabled
        if self.enable_resources:
            metrics['total_resources'] = self.resource_system.get_total_resources()
            metrics['consumption'] = self.resource_system.consumption_history[-1] if self.resource_system.consumption_history else 0
        
        # Add environment metrics
        metrics['environment_state'] = self.environment.get_global_state()
        
        # Add memory system metrics
        memory_stats = self.memory_system.get_memory_stats()
        metrics.update({
            'individual_frames': memory_stats['individual_frames'],
            'collective_frames': memory_stats['collective_frames'],
            'individual_activation': memory_stats['individual_activation'],
            'collective_activation': memory_stats['collective_activation']
        })
        
        # Add temporal metrics
        current_time = self.time_manager.get_current_time()
        metrics.update({
            'dt_step': current_time['dt'],
            't_step': current_time['t'],
            'T_step': current_time['T']
        })
        
        # Record metrics in time manager
        for key, value in metrics.items():
            self.time_manager.record_metric(key, value)
        
        return metrics
    
    def run(self, steps=None):
        """
        Run the simulation for a specified number of steps
        
        Parameters:
        - steps: Number of steps to run (defaults to max_steps)
        
        Returns:
        - results: Dictionary of simulation results
        """
        if steps is None:
            steps = self.max_steps
        
        self.logger.info(f"Running unified simulation for {steps} steps...")
        start_time = time.time()
        
        for i in range(steps):
            metrics = self.step()
            
            # Check for early termination conditions
            if metrics['is_equilibrium'] and metrics['is_proportional'] and self.current_step > steps // 2:
                self.logger.info(f"Growth-proportional equilibrium reached at step {self.current_step}. Terminating early.")
                break
        
        end_time = time.time()
        self.logger.info(f"Simulation completed in {end_time - start_time:.2f} seconds")
        
        # Save final metrics
        self.data_manager.save_metrics("final_metrics.json")
        
        # Create experiment summary
        summary = self.data_manager.create_experiment_summary()
        
        # Create manuscript figure
        self.data_manager.create_manuscript_figure(self)
        
        # Create visualizations
        self.create_visualizations()
        
        # Return results
        results = {
            'steps_completed': self.current_step,
            'final_population': len(self.agents),
            'nash_distance': self.nash_validator.nash_distance_history[-1] if self.nash_validator.nash_distance_history else None,
            'coherence': self.field.coherence_history[-1] if self.field.coherence_history else None,
            'is_equilibrium': metrics['is_equilibrium'],
            'is_proportional': metrics['is_proportional'],
            'experiment_dir': self.data_manager.experiment_dir,
            'summary': summary
        }
        
        return results
    
    def create_visualizations(self):
        """Create and save visualizations of simulation results"""
        self.logger.info("Creating visualizations...")
        
        # Strategic field visualization
        fig = self.field.visualize(include_wave=True, agents=self.agents)
        self.data_manager.save_visualization(fig, "strategic_field.png")
        
        # Field coherence visualization
        fig = self.field.visualize_coherence()
        self.data_manager.save_visualization(fig, "field_coherence.png")
        
        # Nash distance visualization
        fig = self.nash_validator.visualize_nash_distance()
        self.data_manager.save_visualization(fig, "nash_distance.png")
        
        # Strategy distribution visualization
        fig = self.nash_validator.visualize_strategy_distribution()
        self.data_manager.save_visualization(fig, "strategy_distribution.png")
        
        # Payoff matrix visualization
        fig = self.nash_validator.visualize_payoff_matrix()
        self.data_manager.save_visualization(fig, "payoff_matrix.png")
        
        # Growth-proportional equilibrium visualization
        fig = self.nash_validator.visualize_growth_proportional_equilibrium()
        self.data_manager.save_visualization(fig, "growth_proportional_equilibrium.png")
        
        # Environment visualization
        fig = self.environment.visualize()
        self.data_manager.save_visualization(fig, "environment.png")
        
        # Environment history visualization
        fig = self.environment.visualize_history()
        self.data_manager.save_visualization(fig, "environment_history.png")
        
        # Resource visualization if enabled
        if self.enable_resources:
            fig = self.resource_system.visualize(agents=self.agents)
            self.data_manager.save_visualization(fig, "resources.png")
            
            fig = self.resource_system.visualize_history()
            self.data_manager.save_visualization(fig, "resource_history.png")
        
        # Temporal resonance visualization
        fig = self.time_manager.visualize_resonance()
        self.data_manager.save_visualization(fig, "temporal_resonance.png")


def run_unified_simulation(config):
    """
    Run a unified simulation with the specified configuration
    
    Parameters:
    - config: Dictionary of configuration parameters
    
    Returns:
    - results: Dictionary of simulation results
    """
    # Create simulation
    sim = UnifiedSimulation(**config)
    
    # Run simulation
    results = sim.run()
    
    return results, sim
