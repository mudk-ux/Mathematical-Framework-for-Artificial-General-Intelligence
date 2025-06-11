#!/usr/bin/env python3
"""
Strategic Field implementation for the unified MMAI system

This module implements the concept of strategic fields as described in
"Steps Towards AGI," where strategic information propagates through space
as wave-like patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from matplotlib.colors import LinearSegmentedColormap

class StrategicField:
    """
    Strategic field that propagates strategic information through space
    
    The field represents the distribution of strategies across the spatial grid,
    with wave-like propagation of strategic information through diffusion.
    """
    def __init__(self, grid_size=50, n_strategies=3, diffusion_rate=0.2, 
                 wave_speed=1.0, logger=None):
        """
        Initialize the strategic field
        
        Parameters:
        - grid_size: Size of the environment grid
        - n_strategies: Number of strategies
        - diffusion_rate: Rate of diffusion for strategic information
        - wave_speed: Speed of wave propagation
        """
        self.grid_size = grid_size
        self.n_strategies = n_strategies
        self.diffusion_rate = diffusion_rate
        self.wave_speed = wave_speed
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize field with uniform distribution
        self.field = np.ones((grid_size, grid_size, n_strategies)) / n_strategies
        
        # Initialize field for tracking wave propagation
        self.wave_field = np.zeros((grid_size, grid_size))
        
        # History for tracking
        self.field_history = []
        self.coherence_history = []
        self.wave_history = []
        
        self.logger.info(f"Initialized strategic field with grid size {grid_size} and {n_strategies} strategies")
    
    def update(self, agents, resources=None, dt=0.1):
        """
        Update the field based on agent positions, strategies, and resources
        
        Parameters:
        - agents: List of Agent objects
        - resources: Optional ResourceSystem object
        - dt: Time step size
        
        Returns:
        - coherence: Field coherence measure
        """
        # Each agent contributes to field based on position and strategy
        for agent in agents:
            x, y = agent.position.astype(int)
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                # Agent's contribution is weighted by energy if available
                weight = getattr(agent, 'energy', 1.0)
                self.field[x, y] = (1 - dt) * self.field[x, y] + dt * agent.strategy * weight
                
                # Create wave at agent's position
                self.wave_field[x, y] += weight * 0.1
        
        # Resource influence if available
        if resources is not None:
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    resource_level = resources.get_resources_at(np.array([x, y]))
                    if resource_level > 0:
                        # Resources bias field toward certain strategies
                        # Higher resources favor exploitation (strategy 0)
                        # Lower resources favor exploration (strategy 1+)
                        bias = np.zeros(self.n_strategies)
                        bias[0] = resource_level  # Exploitation
                        for i in range(1, self.n_strategies):
                            bias[i] = (1 - resource_level) / (self.n_strategies - 1)  # Exploration
                        
                        # Apply resource bias
                        self.field[x, y] = (1 - resource_level * 0.1) * self.field[x, y] + resource_level * 0.1 * bias
        
        # Normalize field
        field_sum = np.sum(self.field, axis=2)
        field_sum = np.where(field_sum > 0, field_sum, 1)  # Avoid division by zero
        for s in range(self.n_strategies):
            self.field[:, :, s] = self.field[:, :, s] / field_sum
        
        # Diffuse strategic field
        new_field = self.field.copy()
        for x in range(1, self.grid_size-1):
            for y in range(1, self.grid_size-1):
                new_field[x, y] = (1-self.diffusion_rate) * self.field[x, y] + self.diffusion_rate/4 * (
                    self.field[x-1, y] + self.field[x+1, y] + 
                    self.field[x, y-1] + self.field[x, y+1]
                )
        self.field = new_field
        
        # Propagate wave field
        new_wave = self.wave_field.copy()
        for x in range(1, self.grid_size-1):
            for y in range(1, self.grid_size-1):
                new_wave[x, y] = (1-self.wave_speed*dt) * self.wave_field[x, y] + self.wave_speed*dt/4 * (
                    self.wave_field[x-1, y] + self.wave_field[x+1, y] + 
                    self.wave_field[x, y-1] + self.wave_field[x, y+1]
                )
        self.wave_field = new_wave * 0.99  # Decay
        
        # Calculate coherence
        coherence = self.calculate_coherence()
        self.coherence_history.append(coherence)
        
        # Store field state (periodically to save memory)
        if len(self.field_history) % 10 == 0:
            self.field_history.append(self.field.copy())
            self.wave_history.append(self.wave_field.copy())
        
        return coherence
    
    def calculate_coherence(self):
        """
        Calculate field coherence
        
        Coherence is measured by the average standard deviation across strategies.
        Higher std means more coherent (one strategy dominates at each point).
        
        Returns:
        - coherence: Coherence measure (0 to 1)
        """
        # Coherence is measured by the average standard deviation across strategies
        # Higher std means more coherent (one strategy dominates at each point)
        return np.mean(np.std(self.field, axis=2))
    
    def get_strategy_at_position(self, position):
        """
        Get strategy vector at a specific position
        
        Parameters:
        - position: Position array [x, y]
        
        Returns:
        - strategy: Strategy vector at the position
        """
        x, y = position.astype(int)
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            return self.field[x, y].copy()
        else:
            # Return uniform strategy if position is out of bounds
            return np.ones(self.n_strategies) / self.n_strategies
    
    def get_wave_at_position(self, position):
        """
        Get wave value at a specific position
        
        Parameters:
        - position: Position array [x, y]
        
        Returns:
        - wave: Wave value at the position
        """
        x, y = position.astype(int)
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            return self.wave_field[x, y]
        else:
            return 0.0
    
    def visualize(self, strategy_idx=None, title=None, include_wave=False, agents=None):
        """
        Visualize the strategic field
        
        Parameters:
        - strategy_idx: If provided, visualize only this strategy component
                       If None, visualize dominant strategy at each point
        - title: Optional title for the plot
        - include_wave: Whether to include wave field visualization
        - agents: Optional list of agents to plot on the field
        
        Returns:
        - fig: Matplotlib figure object
        """
        if include_wave:
            fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        else:
            fig, axs = plt.subplots(1, 1, figsize=(10, 8))
            axs = [axs]
        
        if strategy_idx is not None:
            # Visualize specific strategy component
            im = axs[0].imshow(self.field[:, :, strategy_idx], cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(im, ax=axs[0], label=f'Strategy {strategy_idx+1} Strength')
            if title:
                axs[0].set_title(title, fontsize=14)
            else:
                axs[0].set_title(f'Strategic Field - Strategy {strategy_idx+1}', fontsize=14)
        else:
            # Visualize dominant strategy at each point
            dominant_field = np.argmax(self.field, axis=2)
            
            # Create custom colormap for strategies
            colors = plt.cm.tab10(np.linspace(0, 1, self.n_strategies))
            cmap = LinearSegmentedColormap.from_list('strategies', colors, N=self.n_strategies)
            
            im = axs[0].imshow(dominant_field, cmap=cmap, interpolation='nearest', vmin=0, vmax=self.n_strategies-1)
            cbar = plt.colorbar(im, ax=axs[0], ticks=range(self.n_strategies))
            cbar.set_label('Dominant Strategy')
            cbar.set_ticklabels([f'S{i+1}' for i in range(self.n_strategies)])
            
            if title:
                axs[0].set_title(title, fontsize=14)
            else:
                axs[0].set_title('Strategic Field - Dominant Strategy', fontsize=14)
        
        # Plot agents if provided
        if agents:
            agent_x = [agent.position[0] for agent in agents]
            agent_y = [agent.position[1] for agent in agents]
            axs[0].scatter(agent_y, agent_x, c='red', s=10, alpha=0.7)
        
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        
        # Visualize wave field if requested
        if include_wave:
            im2 = axs[1].imshow(self.wave_field, cmap='plasma', vmin=0)
            plt.colorbar(im2, ax=axs[1], label='Wave Intensity')
            axs[1].set_title('Strategic Wave Propagation', fontsize=14)
            axs[1].set_xticks([])
            axs[1].set_yticks([])
            
            # Plot agents if provided
            if agents:
                agent_x = [agent.position[0] for agent in agents]
                agent_y = [agent.position[1] for agent in agents]
                axs[1].scatter(agent_y, agent_x, c='white', s=10, alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def visualize_coherence(self, title=None):
        """
        Visualize coherence over time
        
        Parameters:
        - title: Optional title for the plot
        
        Returns:
        - fig: Matplotlib figure object
        """
        if not self.coherence_history:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.coherence_history, linewidth=2)
        
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title('Field Coherence Over Time', fontsize=14)
        
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Coherence', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def create_wave_animation_data(self):
        """
        Prepare data for wave animation
        
        Returns:
        - wave_data: List of wave field snapshots
        """
        return self.wave_history
    
    def create_field_animation_data(self):
        """
        Prepare data for field animation
        
        Returns:
        - field_data: List of field snapshots
        """
        return self.field_history
