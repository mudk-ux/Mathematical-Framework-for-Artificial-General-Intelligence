#!/usr/bin/env python3
"""
Resource System for the unified MMAI system

This module implements resource dynamics with growth and consumption models.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging

class ResourceSystem:
    """
    Resource system with growth and consumption dynamics
    
    The resource system manages:
    - Spatial distribution of resources
    - Resource growth based on logistic model
    - Resource consumption by agents
    - Resource diffusion across the grid
    """
    def __init__(self, grid_size=50, initial_density=0.3, growth_rate=0.05, 
                 carrying_capacity=1.0, diffusion_rate=0.01, logger=None):
        """
        Initialize the resource system
        
        Parameters:
        - grid_size: Size of the environment grid
        - initial_density: Initial resource density (0 to 1)
        - growth_rate: Rate of resource growth
        - carrying_capacity: Maximum resource level
        - diffusion_rate: Rate of resource diffusion
        """
        self.grid_size = grid_size
        self.growth_rate = growth_rate
        self.carrying_capacity = carrying_capacity
        self.diffusion_rate = diffusion_rate
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize resources with random distribution
        self.resources = np.random.random((grid_size, grid_size)) * initial_density
        
        # Create resource clusters
        self.create_resource_clusters(n_clusters=5, cluster_size=5)
        
        # History tracking
        self.total_resources_history = []
        self.consumption_history = []
        
        self.logger.info(f"Initialized resource system with initial density {initial_density}")
    
    def create_resource_clusters(self, n_clusters=5, cluster_size=5):
        """
        Create clusters of resources
        
        Parameters:
        - n_clusters: Number of resource clusters
        - cluster_size: Size of each cluster
        """
        for _ in range(n_clusters):
            # Random cluster center
            center_x = np.random.randint(0, self.grid_size)
            center_y = np.random.randint(0, self.grid_size)
            
            # Create cluster with Gaussian distribution
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    # Distance from cluster center
                    dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                    
                    # Add resources based on distance (Gaussian falloff)
                    if dist < cluster_size * 2:
                        resource_value = np.exp(-dist**2 / (2 * cluster_size**2)) * self.carrying_capacity
                        self.resources[i, j] = max(self.resources[i, j], resource_value)
    
    def update(self, agents=None, dt=0.1, current_time=0, environment=None):
        """
        Update resource levels
        
        Parameters:
        - agents: List of Agent objects
        - dt: Time step size
        - current_time: Current simulation time
        - environment: Optional EnvironmentSystem object
        
        Returns:
        - total_resources: Total resources after update
        """
        # Track consumption
        total_consumption = 0.0
        
        # Resource consumption by agents
        if agents is not None:
            for agent in agents:
                x, y = agent.position.astype(int)
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    # Consumption rate based on agent energy
                    consumption_rate = 0.1 * (1.0 - agent.energy / 2.0)  # Higher when energy is low
                    
                    # Limit consumption by available resources
                    consumption = min(self.resources[x, y], consumption_rate)
                    
                    # Update resources
                    self.resources[x, y] -= consumption
                    
                    # Agent consumes resources
                    if hasattr(agent, 'consume_resources'):
                        agent.consume_resources(consumption)
                    
                    total_consumption += consumption
        
        # Resource growth (logistic model)
        growth_factor = self.growth_rate * dt
        
        # Environment influence on growth if available
        if environment is not None:
            # Higher environment state increases growth rate
            env_state = environment.get_global_state()
            growth_factor *= 1.0 + (env_state - 0.5)
        
        # Apply logistic growth
        self.resources += growth_factor * self.resources * (1.0 - self.resources / self.carrying_capacity)
        
        # Resource diffusion
        if self.diffusion_rate > 0:
            new_resources = self.resources.copy()
            for i in range(1, self.grid_size-1):
                for j in range(1, self.grid_size-1):
                    # Average of neighboring cells
                    neighbors_avg = (self.resources[i-1, j] + self.resources[i+1, j] + 
                                    self.resources[i, j-1] + self.resources[i, j+1]) / 4
                    
                    # Diffuse resources
                    new_resources[i, j] = (1-self.diffusion_rate) * self.resources[i, j] + self.diffusion_rate * neighbors_avg
            
            self.resources = new_resources
        
        # Ensure resources are within bounds
        self.resources = np.clip(self.resources, 0, self.carrying_capacity)
        
        # Calculate total resources
        total_resources = np.sum(self.resources)
        
        # Record history
        self.total_resources_history.append(total_resources)
        self.consumption_history.append(total_consumption)
        
        return total_resources
    
    def get_resources_at(self, position):
        """
        Get resource level at a specific position
        
        Parameters:
        - position: Position array [x, y]
        
        Returns:
        - resources: Resource level at the position (0 to carrying_capacity)
        """
        x, y = position.astype(int)
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            return self.resources[x, y]
        else:
            return 0.0
    
    def get_total_resources(self):
        """
        Get total resources in the system
        
        Returns:
        - total_resources: Sum of all resources
        """
        return np.sum(self.resources)
    
    def get_resource_gradient(self, position, radius=3):
        """
        Get resource gradient at a specific position
        
        Parameters:
        - position: Position array [x, y]
        - radius: Radius to check for gradient
        
        Returns:
        - gradient: Resource gradient vector [dx, dy]
        """
        x, y = position.astype(int)
        gradient = np.zeros(2)
        
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            # Check resources in each direction
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    if dx == 0 and dy == 0:
                        continue
                    
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        # Weight by distance and resource level
                        dist = np.sqrt(dx**2 + dy**2)
                        weight = 1.0 / (dist + 0.1)  # Avoid division by zero
                        
                        # Add weighted contribution to gradient
                        resource_diff = self.resources[nx, ny] - self.resources[x, y]
                        if resource_diff > 0:  # Only consider positive gradients
                            gradient[0] += dx * weight * resource_diff
                            gradient[1] += dy * weight * resource_diff
        
        # Normalize gradient
        norm = np.linalg.norm(gradient)
        if norm > 0:
            gradient /= norm
        
        return gradient
    
    def consume_at(self, position, amount):
        """
        Consume resources at a specific position
        
        Parameters:
        - position: Position array [x, y]
        - amount: Amount of resources to consume
        
        Returns:
        - consumed: Actual amount consumed
        """
        x, y = position.astype(int)
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            consumed = min(self.resources[x, y], amount)
            self.resources[x, y] -= consumed
            return consumed
        else:
            return 0.0
    
    def visualize(self, agents=None, title=None):
        """
        Visualize the resource distribution
        
        Parameters:
        - agents: Optional list of agents to plot on the resource map
        - title: Optional title for the plot
        
        Returns:
        - fig: Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(self.resources, cmap='YlGn', vmin=0, vmax=self.carrying_capacity)
        plt.colorbar(im, ax=ax, label='Resource Level')
        
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title('Resource Distribution', fontsize=14)
        
        # Plot agents if provided
        if agents:
            agent_x = [agent.position[0] for agent in agents]
            agent_y = [agent.position[1] for agent in agents]
            ax.scatter(agent_y, agent_x, c='red', s=10, alpha=0.7)
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        return fig
    
    def visualize_history(self, title=None):
        """
        Visualize the history of total resources and consumption
        
        Parameters:
        - title: Optional title for the plot
        
        Returns:
        - fig: Matplotlib figure object
        """
        if not self.total_resources_history:
            return None
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot total resources
        color = 'tab:green'
        ax1.set_xlabel('Time Step', fontsize=12)
        ax1.set_ylabel('Total Resources', color=color, fontsize=12)
        ax1.plot(self.total_resources_history, color=color, linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Create second y-axis for consumption
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Consumption', color=color, fontsize=12)
        ax2.plot(self.consumption_history, color=color, linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)
        
        if title:
            plt.title(title, fontsize=14)
        else:
            plt.title('Resource History', fontsize=14)
        
        plt.grid(True, alpha=0.3)
        
        return fig
