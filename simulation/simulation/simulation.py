#!/usr/bin/env python3
"""
Simulation for the unified MMAI system

This module implements the simulation for the unified MMAI system.
"""

import numpy as np
import logging
from core.agent import Agent

class Simulation:
    """
    Simulation for the unified MMAI system
    """
    
    def __init__(self, strategic_field=None, n_agents=100, n_strategies=3, payoff_matrix=None,
                 fractal_time_manager=None, nash_validator=None, growth_rate=0.01, logger=None):
        """
        Initialize simulation
        
        Parameters:
        - strategic_field: Strategic field
        - n_agents: Number of agents
        - n_strategies: Number of strategies
        - payoff_matrix: Payoff matrix
        - fractal_time_manager: Fractal time manager
        - nash_validator: Nash validator
        - growth_rate: Growth rate for population dynamics
        - logger: Logger
        """
        self.n_agents = n_agents
        self.n_strategies = n_strategies
        self.payoff_matrix = payoff_matrix
        self.fractal_time_manager = fractal_time_manager
        self.nash_validator = nash_validator
        self.growth_rate = growth_rate
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize strategic field if not provided
        if strategic_field is None:
            from core.strategic_field import StrategicField
            self.strategic_field = StrategicField(grid_size=50, n_strategies=n_strategies)
        else:
            self.strategic_field = strategic_field
            
        # Initialize agents
        self.agents = []
        grid_size = self.strategic_field.grid_size
        
        for i in range(n_agents):
            # Random position
            position = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
            
            # Random strategy
            strategy = np.zeros(n_strategies)
            strategy[np.random.randint(0, n_strategies)] = 1.0
            
            # Create agent
            agent = Agent(
                agent_id=i,
                position=position,
                strategy=strategy,
                payoff_matrix=payoff_matrix
            )
            
            self.agents.append(agent)
            
        self.logger.info(f"Initialized simulation with {n_agents} agents")
        
    def run(self, steps=1000):
        """
        Run simulation
        
        Parameters:
        - steps: Number of steps
        
        Returns:
        - results: Simulation results
        """
        self.logger.info(f"Running simulation for {steps} steps")
        
        # Initialize results
        results = {
            'strategy_distribution': [],
            'field_snapshots': [],
            'agent_positions': [],
            'nash_equilibria': []
        }
        
        # Run simulation
        for step in range(steps):
            if step % 50 == 0:
                self.logger.info(f"Simulation step {step}/{steps}")
                
            # Update strategic field
            self.strategic_field.update(self.agents)
            
            # Update agents
            for agent in self.agents:
                # Make decision
                agent.decide(self.strategic_field)
                
                # Move randomly
                agent.move(grid_size=self.strategic_field.grid_size)
                
            # Record results
            if step % 10 == 0:
                # Record strategy distribution
                strategy_dist = np.zeros(self.n_strategies)
                for agent in self.agents:
                    strategy_dist += agent.strategy
                strategy_dist /= self.n_agents
                results['strategy_distribution'].append(strategy_dist)
                
                # Record field snapshot
                results['field_snapshots'].append(self.strategic_field.field.copy())
                
                # Record agent positions
                positions = np.array([agent.position for agent in self.agents])
                results['agent_positions'].append(positions)
                
                # Record Nash equilibrium
                if self.nash_validator:
                    nash_eq = self.nash_validator.validate(strategy_dist)
                    results['nash_equilibria'].append(nash_eq)
                    
            # Update fractal time
            if self.fractal_time_manager:
                dt = self.fractal_time_manager.update()
                
                # Check if simulation should end
                if dt >= steps:
                    self.logger.info(f"T-scale update: T={self.fractal_time_manager.T}, t={self.fractal_time_manager.t}, dt={dt}")
                    break
                    
        self.logger.info(f"Simulation completed in {step+1} steps")
        
        return results
