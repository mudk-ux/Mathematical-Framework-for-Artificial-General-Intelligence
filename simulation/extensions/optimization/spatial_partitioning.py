#!/usr/bin/env python3
"""
Spatial Partitioning for Optimized Agent Interactions

This module implements spatial partitioning techniques to optimize agent
interactions, reducing complexity from O(n²) to O(n) for local interactions.
"""

import numpy as np
import logging
from collections import defaultdict

class SpatialPartitioning:
    """
    Implements spatial partitioning for efficient agent interactions
    
    This class divides the environment into a grid of cells and tracks which
    agents are in each cell, enabling efficient queries for nearby agents.
    """
    def __init__(self, grid_size=50, cell_size=5, logger=None):
        """
        Initialize the spatial partitioning system
        
        Parameters:
        - grid_size: Size of the environment grid
        - cell_size: Size of each partition cell
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.logger = logger or logging.getLogger(__name__)
        
        # Calculate number of cells in each dimension
        self.n_cells = int(np.ceil(grid_size / cell_size))
        
        # Initialize grid
        self.grid = defaultdict(list)
        
        # Performance tracking
        self.query_count = 0
        self.total_checks = 0
        self.naive_checks = 0
        
        self.logger.info(f"Initialized spatial partitioning with {self.n_cells}x{self.n_cells} cells")
    
    def update(self, agents):
        """
        Update the spatial partitioning grid with current agent positions
        
        Parameters:
        - agents: List of Agent objects
        """
        # Clear the grid
        self.grid.clear()
        
        # Add agents to grid cells
        for agent in agents:
            cell = self._position_to_cell(agent.position)
            self.grid[cell].append(agent)
    
    def _position_to_cell(self, position):
        """
        Convert a position to a cell index
        
        Parameters:
        - position: Position array [x, y]
        
        Returns:
        - cell: Tuple (cell_x, cell_y)
        """
        cell_x = int(position[0] / self.cell_size)
        cell_y = int(position[1] / self.cell_size)
        
        # Clamp to valid range
        cell_x = max(0, min(cell_x, self.n_cells - 1))
        cell_y = max(0, min(cell_y, self.n_cells - 1))
        
        return (cell_x, cell_y)
    
    def get_nearby_agents(self, position, radius):
        """
        Get agents near a position
        
        Parameters:
        - position: Position array [x, y]
        - radius: Search radius
        
        Returns:
        - nearby_agents: List of nearby Agent objects
        """
        self.query_count += 1
        
        # Convert position to cell
        center_cell = self._position_to_cell(position)
        
        # Calculate cell range to check
        cell_radius = int(np.ceil(radius / self.cell_size))
        min_x = max(0, center_cell[0] - cell_radius)
        max_x = min(self.n_cells - 1, center_cell[0] + cell_radius)
        min_y = max(0, center_cell[1] - cell_radius)
        max_y = min(self.n_cells - 1, center_cell[1] + cell_radius)
        
        # Collect agents from cells in range
        nearby_agents = []
        position_array = np.array(position)
        
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                cell = (x, y)
                for agent in self.grid[cell]:
                    self.total_checks += 1
                    
                    # Check actual distance
                    distance = np.linalg.norm(agent.position - position_array)
                    if distance <= radius:
                        nearby_agents.append(agent)
        
        # For performance comparison, calculate how many checks would be needed without partitioning
        self.naive_checks += len(sum(self.grid.values(), []))
        
        return nearby_agents
    
    def get_performance_stats(self):
        """
        Get performance statistics
        
        Returns:
        - stats: Dictionary of performance statistics
        """
        if self.query_count == 0:
            return {
                'query_count': 0,
                'avg_checks_per_query': 0,
                'avg_naive_checks_per_query': 0,
                'efficiency_ratio': 1.0
            }
        
        avg_checks = self.total_checks / self.query_count
        avg_naive = self.naive_checks / self.query_count
        
        if avg_naive == 0:
            efficiency_ratio = 1.0
        else:
            efficiency_ratio = avg_checks / avg_naive
        
        return {
            'query_count': self.query_count,
            'avg_checks_per_query': avg_checks,
            'avg_naive_checks_per_query': avg_naive,
            'efficiency_ratio': efficiency_ratio
        }
    
    def visualize(self, ax=None):
        """
        Visualize the spatial partitioning grid
        
        Parameters:
        - ax: Optional matplotlib axis
        
        Returns:
        - ax: Matplotlib axis
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10))
        
        # Draw grid lines
        for i in range(self.n_cells + 1):
            x = i * self.cell_size
            ax.axvline(x=x, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=x, color='gray', linestyle='--', alpha=0.5)
        
        # Draw cell occupancy
        for cell, agents in self.grid.items():
            if agents:
                x = cell[0] * self.cell_size + self.cell_size / 2
                y = cell[1] * self.cell_size + self.cell_size / 2
                ax.text(x, y, str(len(agents)), ha='center', va='center')
        
        # Set limits
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        
        # Add title
        stats = self.get_performance_stats()
        ax.set_title(f"Spatial Partitioning (Efficiency: {stats['efficiency_ratio']:.2f}x)")
        
        return ax
