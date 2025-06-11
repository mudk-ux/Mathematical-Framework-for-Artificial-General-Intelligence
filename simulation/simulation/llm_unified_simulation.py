#!/usr/bin/env python3
"""
LLM-Enhanced Unified Simulation for the MMAI system

This module implements the main simulation framework that integrates all components
with LLM-enhanced capabilities via Amazon Bedrock.
"""

import os
import sys
import time
import numpy as np
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from core.agent import Agent
from core.llm_agent import LLMAgent
from core.strategic_field import StrategicField
from core.fractal_time_manager import FractalTimeManager
from core.memory_system import MemorySystem
from core.llm_enhanced_memory import LLMEnhancedMemorySystem
from core.nash_validator import NashValidator
from core.llm_enhanced_nash_validator import LLMEnhancedNashValidator
from core.environment_system import EnvironmentSystem, EnvironmentType
from core.resource_system import ResourceSystem
from simulation.data_manager import DataManager

class LLMUnifiedSimulation:
    """
    LLM-enhanced unified simulation that integrates all components of the MMAI system
    
    This simulation framework combines:
    - Standard and LLM-powered agents
    - Strategic field for spatial coordination
    - Fractal time architecture for multi-scale temporal dynamics
    - LLM-enhanced memory system for individual and collective memory
    - LLM-enhanced Nash validator for equilibrium analysis
    - Environment system for environmental dynamics
    - Resource system for resource dynamics
    """
    def __init__(self, 
                 grid_size=20,
                 n_agents=10,
                 max_agents=20,
                 n_strategies=3,
                 dt=0.01,
                 t_scale=10,
                 T_scale=5,
                 env_type=EnvironmentType.STATIC,
                 enable_reproduction=True,
                 enable_resources=True,
                 enable_dynamic_population=True,
                 max_steps=20,
                 equilibrium_threshold=0.1,
                 data_dir="./results",
                 experiment_name=None,
                 model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                 region="us-east-1",
                 llm_agent_ratio=0.5,
                 logger=None):
        """
        Initialize the LLM-enhanced unified simulation
        
        Parameters:
        - grid_size: Size of the environment grid
        - n_agents: Initial number of agents
        - max_agents: Maximum number of agents
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
        - data_dir: Directory to store results
        - experiment_name: Name of the experiment
        - model_id: Amazon Bedrock model ID
        - region: AWS region for Bedrock
        - llm_agent_ratio: Ratio of LLM agents to total agents (0.0 to 1.0)
        - logger: Optional logger instance
        """
        # Set up logger
        self.logger = logger or logging.getLogger(__name__)
        
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
        
        # LLM parameters
        self.model_id = model_id
        self.region = region
        self.llm_agent_ratio = llm_agent_ratio
        
        # Set up data management
        self.data_manager = DataManager(base_dir=data_dir, experiment_name=experiment_name)
        self.experiment_name = experiment_name or f"llm_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize components
        self.initialize_components()
        
        # Tracking variables
        self.current_step = 0
        self.is_running = False
        self.results = {}
        
        self.logger.info(f"Initialized LLM-enhanced unified simulation with {n_agents} agents ({int(n_agents * llm_agent_ratio)} LLM agents)")
    
    def initialize_components(self):
        """Initialize all simulation components"""
        # Initialize strategic field
        self.strategic_field = StrategicField(
            grid_size=int(self.grid_size),
            n_strategies=int(self.n_strategies),
            logger=self.logger
        )
        
        # Initialize fractal time manager
        self.time_manager = FractalTimeManager(
            dt=self.dt,
            t_scale=self.t_scale,
            T_scale=self.T_scale,
            logger=self.logger
        )
        
        # Initialize LLM-enhanced memory system
        self.memory_system = LLMEnhancedMemorySystem(
            capacity=1000,
            n_frames=10,
            model_id=self.model_id,
            region=self.region,
            logger=self.logger
        )
        
        # Initialize LLM-enhanced Nash validator
        self.nash_validator = LLMEnhancedNashValidator(
            n_strategies=self.n_strategies,
            equilibrium_threshold=self.equilibrium_threshold,
            model_id=self.model_id,
            region=self.region,
            logger=self.logger
        )
        
        # Initialize environment system
        self.environment = EnvironmentSystem(
            grid_size=self.grid_size,
            env_type=self.env_type,
            logger=self.logger
        )
        
        # Initialize resource system if enabled
        if self.enable_resources:
            self.resource_system = ResourceSystem(
                grid_size=self.grid_size,
                logger=self.logger
            )
        else:
            self.resource_system = None
        
        # Initialize agents
        self.agents = []
        self.initialize_agents()
        
        # Initialize metrics tracking
        self.metrics = {
            'nash_distance': [],
            'coherence': [],
            'population': [],
            'resources': [],
            'llm_agent_count': [],
            'standard_agent_count': [],
            'llm_response_times': [],
            'hypersensitive_points': []
        }
    
    def initialize_agents(self):
        """Initialize agents with a mix of standard and LLM agents"""
        self.agents = []
        
        # Calculate number of LLM agents
        n_llm_agents = int(self.n_agents * self.llm_agent_ratio)
        n_standard_agents = self.n_agents - n_llm_agents
        
        self.logger.info(f"Initializing {n_llm_agents} LLM agents and {n_standard_agents} standard agents")
        
        # Create LLM agents
        for i in range(n_llm_agents):
            position = np.random.randint(0, self.grid_size, size=2)
            agent = LLMAgent(
                agent_id=i,
                position=position,
                model_id=self.model_id,
                region=self.region,
                n_strategies=self.n_strategies,
                logger=self.logger
            )
            self.agents.append(agent)
        
        # Create standard agents
        for i in range(n_llm_agents, self.n_agents):
            position = np.random.randint(0, self.grid_size, size=2)
            agent = Agent(
                agent_id=i,
                position=position,
                n_strategies=self.n_strategies,
                logger=self.logger
            )
            self.agents.append(agent)
    
    def run(self):
        """Run the simulation for the specified number of steps"""
        self.is_running = True
        self.current_step = 0
        self.log_interval = 5  # Default log interval
        
        self.logger.info(f"Starting LLM-enhanced simulation for {self.max_steps} steps")
        
        start_time = time.time()
        
        try:
            while self.current_step < self.max_steps and self.is_running:
                self.step()
                
                # Log progress periodically
                if self.current_step % self.log_interval == 0 or self.current_step == self.max_steps - 1:
                    elapsed = time.time() - start_time
                    self.logger.info(f"Step {self.current_step}/{self.max_steps} completed ({elapsed:.2f}s elapsed)")
                    
                    # Calculate and log metrics
                    n_llm_agents = sum(1 for agent in self.agents if isinstance(agent, LLMAgent))
                    n_standard_agents = len(self.agents) - n_llm_agents
                    
                    self.logger.info(f"  Agents: {len(self.agents)} ({n_llm_agents} LLM, {n_standard_agents} standard)")
                    
                    # Handle case where nash_distance is a tuple
                    nash_dist = self.metrics['nash_distance'][-1]
                    if isinstance(nash_dist, tuple):
                        nash_dist = nash_dist[0]
                    self.logger.info(f"  Nash distance: {nash_dist:.4f}")
                    
                    self.logger.info(f"  Field coherence: {self.metrics['coherence'][-1]:.4f}")
                    
                    # Get LLM performance stats
                    llm_agent_stats = [agent.get_performance_stats() for agent in self.agents if isinstance(agent, LLMAgent)]
                    if llm_agent_stats:
                        avg_response_time = np.mean([stats['avg_response_time'] for stats in llm_agent_stats])
                        avg_cache_hit_rate = np.mean([stats['cache_hit_rate'] for stats in llm_agent_stats])
                        self.logger.info(f"  LLM avg response time: {avg_response_time:.2f}s, cache hit rate: {avg_cache_hit_rate:.2f}")
        
        except Exception as e:
            self.logger.error(f"Error during simulation: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        self.is_running = False
        
        # Collect final results
        self.collect_results()
        
        return self.results
    
    def step(self):
        """Execute one simulation step"""
        # Update time manager
        event_type = self.time_manager.step()
        time_scales = self.time_manager.get_current_time()
        dt_step, t_step, T_step = time_scales['dt'], time_scales['t'], time_scales['T']
        
        # Update environment
        env_state = self.environment.update(dt=self.dt, current_time=dt_step)
        
        # Update resources if enabled
        if self.resource_system:
            self.resource_system.update(self.agents, dt=self.dt, current_time=dt_step, environment=self.environment)
        
        # Update agents
        self.update_agents(env_state, dt_step, t_step, T_step)
        
        # Update strategic field
        coherence = self.strategic_field.update(self.agents, self.resource_system, self.dt)
        
        # Update Nash validator
        payoff_matrix = self.nash_validator.calculate_payoff_matrix(self.agents)
        nash_distance = self.nash_validator.calculate_nash_distance(self.agents)
        is_equilibrium = self.nash_validator.check_equilibrium(nash_distance)
        
        # Perform LLM-enhanced Nash analysis periodically
        if self.current_step % 5 == 0 or is_equilibrium:
            self.logger.info("Performing LLM-enhanced Nash analysis...")
            nash_analysis = self.nash_validator.analyze_equilibrium_dynamics()
            
            # Store analysis in memory system
            self.memory_system.add_memory({
                'type': 'nash_analysis',
                'time_step': self.current_step,
                'nash_distance': nash_distance,
                'is_equilibrium': is_equilibrium,
                'analysis': nash_analysis
            })
        
        # Update metrics
        self.metrics['nash_distance'].append(nash_distance)
        self.metrics['coherence'].append(coherence)
        self.metrics['population'].append(len(self.agents))
        
        if self.resource_system:
            self.metrics['resources'].append(self.resource_system.get_total_resources())
        else:
            self.metrics['resources'].append(0)
        
        # Track agent types
        n_llm_agents = sum(1 for agent in self.agents if isinstance(agent, LLMAgent))
        n_standard_agents = len(self.agents) - n_llm_agents
        self.metrics['llm_agent_count'].append(n_llm_agents)
        self.metrics['standard_agent_count'].append(n_standard_agents)
        
        # Track hypersensitive points
        n_hypersensitive = sum(1 for agent in self.agents if agent.check_hypersensitivity())
        self.metrics['hypersensitive_points'].append(n_hypersensitive)
        
        # Track LLM response times
        llm_agent_stats = [agent.get_performance_stats() for agent in self.agents if isinstance(agent, LLMAgent)]
        if llm_agent_stats:
            avg_response_time = np.mean([stats['avg_response_time'] for stats in llm_agent_stats])
            self.metrics['llm_response_times'].append(avg_response_time)
        else:
            self.metrics['llm_response_times'].append(0)
        
        # Increment step counter
        self.current_step += 1
    
    def update_agents(self, env_state, dt_step, t_step, T_step):
        """Update all agents"""
        # Track agents to remove (dead) and add (newborn)
        agents_to_remove = []
        agents_to_add = []
        
        # Update each agent
        for agent in self.agents:
            # Move agent
            resource_field = self.resource_system if self.enable_resources else None
            agent.move(self.grid_size, self.strategic_field, resource_field)
            
            # Get strategy influence from field
            field_influence = self.strategic_field.get_strategy_at_position(agent.position)
            
            # Get memory influence if agent has memory
            memory_influence = None
            if hasattr(agent, 'get_memory_influence'):
                memory_influence = agent.get_memory_influence()
            
            # Get best response from Nash validator
            best_response = self.nash_validator.get_best_response(agent.strategy)
            
            # Get resource level at agent's position
            resource_level = 0.0
            if self.resource_system:
                resource_level = self.resource_system.get_resources_at(agent.position)
                
                # Consume resources
                if resource_level > 0:
                    consumed = min(resource_level, 0.1)  # Limit consumption
                    self.resource_system.consume_at(agent.position, consumed)
                    agent.consume_resources(consumed)
            
            # Update agent strategy
            if isinstance(agent, LLMAgent):
                # LLM agents make decisions using LLM reasoning
                decision, reasoning = agent.make_decision(env_state, self.strategic_field, self.memory_system)
                
                # Store decision in memory
                self.memory_system.add_memory({
                    'type': 'llm_decision',
                    'agent_id': agent.agent_id,
                    'position': agent.position.tolist(),
                    'decision': int(decision),
                    'reasoning': reasoning,
                    'time_step': self.current_step
                })
            else:
                # Standard agents update strategy using standard method
                agent.update_strategy(field_influence, memory_influence, best_response, resource_level)
            
            # Check reproduction if enabled
            if self.enable_reproduction and agent.can_reproduce():
                child, success = agent.reproduce()
                if success:
                    # Assign new ID to child
                    child.agent_id = len(self.agents) + len(agents_to_add)
                    agents_to_add.append(child)
            
            # Check if agent is still alive
            if not agent.is_alive():
                agents_to_remove.append(agent)
        
        # Remove dead agents
        for agent in agents_to_remove:
            self.agents.remove(agent)
            self.logger.debug(f"Agent {agent.agent_id} died")
        
        # Add new agents if under max population
        if len(self.agents) + len(agents_to_add) <= self.max_agents:
            self.agents.extend(agents_to_add)
            for agent in agents_to_add:
                self.logger.debug(f"New agent {agent.agent_id} born")
        else:
            # Only add agents up to max population
            available_slots = self.max_agents - len(self.agents)
            self.agents.extend(agents_to_add[:available_slots])
            for agent in agents_to_add[:available_slots]:
                self.logger.debug(f"New agent {agent.agent_id} born")
    
    def collect_results(self):
        """Collect and organize simulation results"""
        # Basic metrics
        self.results = {
            'experiment_name': self.experiment_name,
            'parameters': {
                'grid_size': int(self.grid_size),
                'n_agents': int(self.n_agents),
                'max_agents': int(self.max_agents),
                'n_strategies': int(self.n_strategies),
                'dt': float(self.dt),
                't_scale': int(self.t_scale),
                'T_scale': int(self.T_scale),
                'env_type': str(self.env_type),
                'enable_reproduction': bool(self.enable_reproduction),
                'enable_resources': bool(self.enable_resources),
                'enable_dynamic_population': bool(self.enable_dynamic_population),
                'max_steps': int(self.max_steps),
                'equilibrium_threshold': float(self.equilibrium_threshold),
                'model_id': str(self.model_id),
                'region': str(self.region),
                'llm_agent_ratio': float(self.llm_agent_ratio)
            },
            'metrics': {
                "nash_distance": [float(x) if isinstance(x, (int, float)) else float(x[0]) if isinstance(x, tuple) else 0.0 for x in self.metrics["nash_distance"]],
                "coherence": [float(x) for x in self.metrics["coherence"]],
                "population": [int(x) for x in self.metrics["population"]],
                "resources": [float(x) for x in self.metrics["resources"]],
                "llm_agent_count": [int(x) for x in self.metrics["llm_agent_count"]],
                "standard_agent_count": [int(x) for x in self.metrics["standard_agent_count"]]
            },
            'final_state': {
                'n_agents': int(len(self.agents)),
                'n_llm_agents': int(sum(1 for agent in self.agents if isinstance(agent, LLMAgent))),
                'n_standard_agents': int(sum(1 for agent in self.agents if not isinstance(agent, LLMAgent))),
                'nash_distance': float(self.metrics['nash_distance'][-1][0]) if isinstance(self.metrics['nash_distance'][-1], tuple) else float(self.metrics['nash_distance'][-1]) if self.metrics['nash_distance'] else 0.0,
                'coherence': float(self.metrics['coherence'][-1]) if self.metrics['coherence'] else 0.0
            }
        }
        
        # Get LLM performance stats
        llm_agent_stats = [agent.get_performance_stats() for agent in self.agents if isinstance(agent, LLMAgent)]
        if llm_agent_stats:
            avg_response_time = np.mean([stats['avg_response_time'] for stats in llm_agent_stats])
            avg_cache_hit_rate = np.mean([stats['cache_hit_rate'] for stats in llm_agent_stats])
            avg_cache_size = np.mean([stats['cache_size'] for stats in llm_agent_stats])
            
            self.results['llm_performance'] = {
                'avg_response_time': float(avg_response_time),
                'avg_cache_hit_rate': float(avg_cache_hit_rate),
                'avg_cache_size': float(avg_cache_size),
                'total_llm_calls': int(sum(stats['llm_call_count'] for stats in llm_agent_stats))
            }
        else:
            self.results['llm_performance'] = {
                'avg_response_time': 0.0,
                'avg_cache_hit_rate': 0.0,
                'avg_cache_size': 0.0,
                'total_llm_calls': 0
            }
        
        # Get memory system stats
        memory_stats = self.memory_system.get_performance_stats()
        # Convert any numpy types to Python native types
        memory_stats_converted = {}
        for k, v in memory_stats.items():
            if isinstance(v, (np.integer, np.int64, np.int32)):
                memory_stats_converted[k] = int(v)
            elif isinstance(v, (np.floating, np.float64, np.float32)):
                memory_stats_converted[k] = float(v)
            elif isinstance(v, np.ndarray):
                memory_stats_converted[k] = v.tolist()
            else:
                memory_stats_converted[k] = v
        self.results['memory_performance'] = memory_stats_converted
        
        # Get Nash validator stats
        nash_stats = self.nash_validator.get_performance_stats()
        # Convert any numpy types to Python native types
        nash_stats_converted = {}
        for k, v in nash_stats.items():
            if isinstance(v, (np.integer, np.int64, np.int32)):
                nash_stats_converted[k] = int(v)
            elif isinstance(v, (np.floating, np.float64, np.float32)):
                nash_stats_converted[k] = float(v)
            elif isinstance(v, np.ndarray):
                nash_stats_converted[k] = v.tolist()
            else:
                nash_stats_converted[k] = v
        self.results['nash_performance'] = nash_stats_converted
        
        # Save results using data manager
        experiment_dir = self.data_manager.save_experiment_results(
            self.experiment_name,
            self.results,
            create_visualizations=True
        )
        
        self.results['experiment_dir'] = experiment_dir
        
        return self.results
    
    def visualize_results(self):
        """Create visualizations of simulation results"""
        # Create directory for visualizations
        vis_dir = os.path.join(self.data_manager.base_dir, self.experiment_name, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Plot metrics
        self._plot_metrics(vis_dir)
        
        # Visualize strategic field
        fig = self.strategic_field.visualize(agents=self.agents, include_wave=True)
        fig.savefig(os.path.join(vis_dir, 'strategic_field.png'), dpi=300)
        plt.close(fig)
        
        # Visualize agent distribution by type
        self._plot_agent_distribution(vis_dir)
        
        # Visualize LLM performance
        self._plot_llm_performance(vis_dir)
        
        return vis_dir
    
    def _plot_metrics(self, vis_dir):
        """Plot simulation metrics"""
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Nash distance
        axs[0, 0].plot(self.metrics['nash_distance'], 'b-', linewidth=2)
        axs[0, 0].set_title('Nash Distance')
        axs[0, 0].set_xlabel('Time Step')
        axs[0, 0].set_ylabel('Distance')
        axs[0, 0].grid(True, alpha=0.3)
        
        # Coherence
        axs[0, 1].plot(self.metrics['coherence'], 'g-', linewidth=2)
        axs[0, 1].set_title('Field Coherence')
        axs[0, 1].set_xlabel('Time Step')
        axs[0, 1].set_ylabel('Coherence')
        axs[0, 1].grid(True, alpha=0.3)
        
        # Population
        axs[1, 0].plot(self.metrics['population'], 'r-', linewidth=2)
        axs[1, 0].set_title('Population')
        axs[1, 0].set_xlabel('Time Step')
        axs[1, 0].set_ylabel('Agents')
        axs[1, 0].grid(True, alpha=0.3)
        
        # Resources
        axs[1, 1].plot(self.metrics['resources'], 'y-', linewidth=2)
        axs[1, 1].set_title('Total Resources')
        axs[1, 1].set_xlabel('Time Step')
        axs[1, 1].set_ylabel('Resources')
        axs[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(os.path.join(vis_dir, 'metrics.png'), dpi=300)
        plt.close(fig)
    
    def _plot_agent_distribution(self, vis_dir):
        """Plot agent distribution by type"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = range(len(self.metrics['llm_agent_count']))
        ax.stackplot(x, 
                    [self.metrics['llm_agent_count'], self.metrics['standard_agent_count']], 
                    labels=['LLM Agents', 'Standard Agents'],
                    colors=['#3498db', '#e74c3c'],
                    alpha=0.7)
        
        ax.set_title('Agent Population by Type')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Number of Agents')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(os.path.join(vis_dir, 'agent_distribution.png'), dpi=300)
        plt.close(fig)
    
    def _plot_llm_performance(self, vis_dir):
        """Plot LLM performance metrics"""
        if not self.metrics['llm_response_times']:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(self.metrics['llm_response_times'], 'b-', linewidth=2)
        ax.set_title('LLM Response Times')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Average Response Time (s)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(os.path.join(vis_dir, 'llm_performance.png'), dpi=300)
        plt.close(fig)


def run_llm_unified_simulation(config):
    """
    Run a unified simulation with the specified configuration
    
    Parameters:
    - config: Dictionary containing simulation parameters
    
    Returns:
    - results: Dictionary containing simulation results
    - simulation: The simulation object
    """
    # Create logger
    logger = logging.getLogger("llm_unified_simulation")
    logger.setLevel(config.get('log_level', logging.INFO))
    
    # Create simulation
    simulation = LLMUnifiedSimulation(
        grid_size=config.get('grid_size', 20),
        n_agents=config.get('n_agents', 10),
        max_agents=config.get('max_agents', 20),
        n_strategies=config.get('n_strategies', 3),
        dt=config.get('dt', 0.01),
        t_scale=config.get('t_scale', 10),
        T_scale=config.get('T_scale', 5),
        env_type=config.get('env_type', EnvironmentType.STATIC),
        enable_reproduction=config.get('enable_reproduction', True),
        enable_resources=config.get('enable_resources', True),
        enable_dynamic_population=config.get('enable_dynamic_population', True),
        max_steps=config.get('max_steps', 20),
        equilibrium_threshold=config.get('equilibrium_threshold', 0.1),
        data_dir=config.get('data_dir', './results'),
        experiment_name=config.get('experiment_name', None),
        model_id=config.get('model_id', "anthropic.claude-3-sonnet-20240229-v1:0"),
        region=config.get('region', "us-east-1"),
        llm_agent_ratio=config.get('llm_agent_ratio', 0.5),
        logger=logger
    )
    
    # Run simulation
    results = simulation.run()
    
    # Create visualizations
    simulation.visualize_results()
    
    return results, simulation
