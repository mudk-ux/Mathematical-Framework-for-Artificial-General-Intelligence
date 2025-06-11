#!/usr/bin/env python3
"""
Environment System for the unified MMAI system

This module implements different types of environmental dynamics,
including static, periodic, chaotic, and shock patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import logging

class EnvironmentType(Enum):
    """Types of environment dynamics"""
    STATIC = 0
    PERIODIC = 1
    CHAOTIC = 2
    SHOCK = 3

class EnvironmentSystem:
    """
    Environment system with different dynamics patterns
    
    The environment can have different dynamics:
    - Static: Constant environment
    - Periodic: Regular oscillations
    - Chaotic: Chaotic patterns based on logistic map
    - Shock: Sudden changes at random intervals
    """
    def __init__(self, grid_size=50, env_type=EnvironmentType.STATIC, 
                 periodic_frequency=0.1, chaotic_r=3.9, shock_probability=0.01,
                 logger=None):
        """
        Initialize the environment system
        
        Parameters:
        - grid_size: Size of the environment grid
        - env_type: Type of environment dynamics
        - periodic_frequency: Frequency of periodic oscillations
        - chaotic_r: r parameter for logistic map (chaotic when r > 3.57)
        - shock_probability: Probability of shock per time step
        """
        self.grid_size = grid_size
        self.env_type = env_type
        self.periodic_frequency = periodic_frequency
        self.chaotic_r = chaotic_r
        self.shock_probability = shock_probability
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize environment state
        self.state = np.zeros((grid_size, grid_size))
        self.global_state = 0.5  # Global environment state (0 to 1)
        self.previous_global_state = 0.5
        
        # For chaotic dynamics
        self.chaotic_value = 0.5
        
        # For shock dynamics
        self.shock_active = False
        self.shock_duration = 0
        self.shock_intensity = 0.0
        
        # History tracking
        self.global_state_history = []
        self.shock_history = []
        
        self.logger.info(f"Initialized environment system with type {env_type.name}")
    
    def update(self, dt=0.1, current_time=0):
        """
        Update the environment state
        
        Parameters:
        - dt: Time step size
        - current_time: Current simulation time
        
        Returns:
        - state_change: Magnitude of state change
        """
        self.previous_global_state = self.global_state
        
        if self.env_type == EnvironmentType.STATIC:
            # Static environment - no change
            state_change = 0.0
        
        elif self.env_type == EnvironmentType.PERIODIC:
            # Periodic environment - sinusoidal oscillation
            self.global_state = 0.5 + 0.4 * np.sin(2 * np.pi * self.periodic_frequency * current_time * dt)
            state_change = abs(self.global_state - self.previous_global_state)
        
        elif self.env_type == EnvironmentType.CHAOTIC:
            # Chaotic environment - logistic map
            self.chaotic_value = self.chaotic_r * self.chaotic_value * (1 - self.chaotic_value)
            self.global_state = self.chaotic_value
            state_change = abs(self.global_state - self.previous_global_state)
        
        elif self.env_type == EnvironmentType.SHOCK:
            # Shock environment - sudden changes
            if self.shock_active:
                # Continue existing shock
                self.shock_duration -= 1
                if self.shock_duration <= 0:
                    # End shock
                    self.shock_active = False
                    self.global_state = 0.5  # Return to normal
                    state_change = abs(self.global_state - self.previous_global_state)
                    self.logger.info(f"Environmental shock ended at time {current_time}")
                else:
                    # Maintain shock
                    state_change = 0.0
            else:
                # Check for new shock
                if np.random.random() < self.shock_probability:
                    # New shock
                    self.shock_active = True
                    self.shock_duration = np.random.randint(10, 50)  # Duration in time steps
                    self.shock_intensity = 0.5 + 0.5 * np.random.random()  # Intensity (0.5 to 1.0)
                    self.global_state = self.shock_intensity
                    state_change = abs(self.global_state - self.previous_global_state)
                    
                    self.logger.info(f"Environmental shock started at time {current_time} with intensity {self.shock_intensity:.2f}")
                    self.shock_history.append((current_time, self.shock_intensity, self.shock_duration))
                else:
                    # No shock
                    state_change = 0.0
        
        # Update environment state grid
        # Create spatial patterns based on global state
        if state_change > 0:
            # Create spatial gradient
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    # Distance from center
                    center = self.grid_size // 2
                    dist = np.sqrt((i - center)**2 + (j - center)**2) / self.grid_size
                    
                    # State decreases with distance from center
                    self.state[i, j] = self.global_state * (1 - 0.5 * dist)
        
        # Record history
        self.global_state_history.append(self.global_state)
        
        return state_change
    
    def get_state_at(self, position):
        """
        Get environment state at a specific position
        
        Parameters:
        - position: Position array [x, y]
        
        Returns:
        - state: Environment state at the position (0 to 1)
        """
        x, y = position.astype(int)
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            return self.state[x, y]
        else:
            return self.global_state
    
    def get_global_state(self):
        """
        Get the global environment state
        
        Returns:
        - global_state: Global environment state (0 to 1)
        """
        return self.global_state
    
    def visualize(self, title=None):
        """
        Visualize the environment state
        
        Parameters:
        - title: Optional title for the plot
        
        Returns:
        - fig: Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(self.state, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='Environment State')
        
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(f'Environment State ({self.env_type.name})', fontsize=14)
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        return fig
    
    def visualize_history(self, title=None):
        """
        Visualize the history of global environment state
        
        Parameters:
        - title: Optional title for the plot
        
        Returns:
        - fig: Matplotlib figure object
        """
        if not self.global_state_history:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.global_state_history, linewidth=2)
        
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(f'Global Environment State History ({self.env_type.name})', fontsize=14)
        
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Global State', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Mark shocks if applicable
        if self.env_type == EnvironmentType.SHOCK and self.shock_history:
            for time, intensity, duration in self.shock_history:
                ax.axvspan(time, time + duration, alpha=0.2, color='red')
                ax.axvline(x=time, color='red', linestyle='--', alpha=0.7)
        
        return fig
