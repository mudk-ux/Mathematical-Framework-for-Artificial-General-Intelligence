#!/usr/bin/env python3
"""
Enhanced Agent for the unified MMAI system

This module implements an enhanced agent with memory integration, hypersensitive
points detection, and strategic decision-making capabilities.
"""

import numpy as np
import logging

class Agent:
    """
    Enhanced agent with IRN integration, hypersensitive points, and strategic decision-making
    
    The agent implements:
    - Strategic decision-making based on field, IRN, and best response
    - Energy dynamics for resource management
    - Reproduction capability
    - Hypersensitive points detection
    - Memory integration with the IRN
    """
    def __init__(self, agent_id, position, strategy=None, payoff_matrix=None, memory_depth=10, 
                 initial_energy=1.0, energy_decay=0.01, reproduction_threshold=2.0,
                 hypersensitive_threshold=0.1, logger=None):
        """
        Initialize the agent
        
        Parameters:
        - agent_id: Unique identifier for the agent
        - position: Initial position [x, y]
        - strategy: Strategy distribution or index (if None, random strategy is generated)
        - payoff_matrix: Payoff matrix for strategy evaluation
        - memory_depth: Depth of memory for outcomes
        - initial_energy: Initial energy level
        - energy_decay: Rate of energy decay per step
        - reproduction_threshold: Energy threshold for reproduction
        - hypersensitive_threshold: Threshold for hypersensitive points
        """
        self.agent_id = agent_id
        self.position = np.array(position)
        self.logger = logger or logging.getLogger(__name__)
        self.payoff_matrix = payoff_matrix
        
        # Initialize strategy
        if strategy is None:
            # Random strategy
            self.n_strategies = 3  # Default
            self.strategy = np.random.random(self.n_strategies)
            self.strategy = self.strategy / np.sum(self.strategy)
        elif isinstance(strategy, int):
            # Pure strategy
            self.n_strategies = payoff_matrix.shape[0] if payoff_matrix is not None else 3
            self.strategy = np.zeros(self.n_strategies)
            self.strategy[strategy] = 1.0
        else:
            # Strategy distribution
            self.strategy = strategy
            self.n_strategies = len(strategy)
        
        # Memory of past states and outcomes
        self.memory_depth = memory_depth
        self.memory = []
        
        # Performance tracking
        self.outcomes = []
        self.strategy_history = []
        
        # Energy dynamics
        self.energy = initial_energy
        self.energy_decay = energy_decay
        self.reproduction_threshold = reproduction_threshold
        self.age = 0
        
        # Hypersensitive points
        self.hypersensitive_threshold = hypersensitive_threshold
        self.hypersensitive_points = []
        
    def decide(self, field):
        """
        Make a decision based on the strategic field
        
        Parameters:
        - field: Strategic field
        
        Returns:
        - strategy: New strategy
        """
        # Get field information at current position
        x, y = self.position.astype(int)
        field_info = field.field[x, y]
        
        # Calculate expected payoffs
        payoffs = np.zeros(self.n_strategies)
        
        # Simple decision: choose strategy with highest field value
        new_strategy = np.zeros(self.n_strategies)
        new_strategy[np.argmax(field_info)] = 1.0
        
        # Update strategy
        self.strategy = new_strategy
        
        # Add to memory
        if len(self.memory) >= self.memory_depth:
            self.memory.pop(0)
        self.memory.append(new_strategy)
        
        return new_strategy
        
    def update_energy(self, outcome):
        """
        Update agent energy based on outcome
        
        Parameters:
        - outcome: Outcome value
        
        Returns:
        - can_reproduce: Whether agent can reproduce
        """
        # Apply outcome to energy
        self.energy += outcome
        
        # Apply energy decay
        self.energy -= self.energy_decay
        
        # Ensure energy is non-negative
        self.energy = max(0.0, self.energy)
        
        # Increment age
        self.age += 1
        
        # Check if agent can reproduce
        can_reproduce = self.energy >= self.reproduction_threshold
        
        return can_reproduce
        
    def reproduce(self):
        """
        Reproduce agent
        
        Returns:
        - child: New agent
        """
        # Create child with same strategy but at slightly different position
        child_position = self.position + np.random.uniform(-1, 1, 2)
        
        # Ensure position is within bounds
        child_position = np.clip(child_position, 0, 49)  # Assuming grid size 50
        
        # Create child with slight mutation in strategy
        child_strategy = self.strategy.copy()
        
        # Apply mutation
        mutation = np.random.normal(0, 0.1, self.n_strategies)
        child_strategy += mutation
        
        # Ensure strategy is valid
        child_strategy = np.clip(child_strategy, 0, 1)
        child_strategy = child_strategy / np.sum(child_strategy)
        
        # Create child
        child = Agent(
            agent_id=self.agent_id * 1000 + np.random.randint(1000),
            position=child_position,
            strategy=child_strategy,
            payoff_matrix=self.payoff_matrix,
            memory_depth=self.memory_depth,
            initial_energy=0.5,  # Start with less energy
            energy_decay=self.energy_decay,
            reproduction_threshold=self.reproduction_threshold,
            hypersensitive_threshold=self.hypersensitive_threshold
        )
        
        # Reduce parent energy
        self.energy -= 0.5
        
        return child
        
    def detect_hypersensitive_points(self, field):
        """
        Detect hypersensitive points in the field
        
        Parameters:
        - field: Strategic field
        
        Returns:
        - points: List of hypersensitive points
        """
        # Get field information at current position
        x, y = self.position.astype(int)
        
        # Check neighboring positions
        neighbors = [
            (x-1, y), (x+1, y), (x, y-1), (x, y+1),
            (x-1, y-1), (x-1, y+1), (x+1, y-1), (x+1, y+1)
        ]
        
        # Filter out positions outside grid
        grid_size = field.field.shape[0]
        neighbors = [(nx, ny) for nx, ny in neighbors if 0 <= nx < grid_size and 0 <= ny < grid_size]
        
        # Check for hypersensitive points
        points = []
        for nx, ny in neighbors:
            # Calculate gradient
            gradient = np.linalg.norm(field.field[nx, ny] - field.field[x, y])
            
            # Check if gradient exceeds threshold
            if gradient > self.hypersensitive_threshold:
                points.append((nx, ny, gradient))
        
        # Store hypersensitive points
        self.hypersensitive_points = points
        
        return points
        
    def move(self, direction=None, grid_size=50):
        """
        Move agent in specified direction or randomly
        
        Parameters:
        - direction: Direction vector [dx, dy] or None for random
        - grid_size: Size of the grid
        
        Returns:
        - new_position: New position
        """
        if direction is None:
            # Move randomly
            direction = np.random.uniform(-1, 1, 2)
            
        # Normalize direction
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
            
        # Apply movement
        self.position += direction
        
        # Ensure position is within bounds
        self.position = np.clip(self.position, 0, grid_size - 1)
        
        return self.position
