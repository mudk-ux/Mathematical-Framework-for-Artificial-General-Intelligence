#!/usr/bin/env python3
"""
Optimized Strategic Field Implementation

This module implements an optimized version of the strategic field using
sparse matrices and FFT-based diffusion for improved computational efficiency.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy import sparse, signal

class OptimizedStrategicField:
    """
    Optimized strategic field implementation using sparse matrices and FFT
    
    This class provides an optimized implementation of the strategic field
    concept, using sparse matrices for memory efficiency and FFT-based
    diffusion for computational efficiency.
    """
    def __init__(self, grid_size=50, n_strategies=3, diffusion_rate=0.2, 
                 use_sparse=True, use_fft=True, logger=None):
        """
        Initialize the optimized strategic field
        
        Parameters:
        - grid_size: Size of the environment grid
        - n_strategies: Number of strategies
        - diffusion_rate: Rate of diffusion for strategic information
        - use_sparse: Whether to use sparse matrices
        - use_fft: Whether to use FFT-based diffusion
        """
        self.grid_size = grid_size
        self.n_strategies = n_strategies
        self.diffusion_rate = diffusion_rate
        self.use_sparse = use_sparse
        self.use_fft = use_fft and grid_size > 32  # Only use FFT for larger grids
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize field
        if use_sparse:
            self.field = [sparse.lil_matrix((grid_size, grid_size)) for _ in range(n_strategies)]
            # Initialize with uniform distribution
            for s in range(n_strategies):
                for i in range(grid_size):
                    for j in range(grid_size):
                        self.field[s][i, j] = 1.0 / n_strategies
        else:
            self.field = np.ones((grid_size, grid_size, n_strategies)) / n_strategies
        
        # For wave visualization
        self.wave_field = np.zeros((grid_size, grid_size))
        
        # For FFT-based diffusion
        if self.use_fft:
            self.prepare_fft_kernel()
        
        # History tracking
        self.coherence_history = []
        self.update_times = []
        
        # Performance tracking
        self.sparse_density = []
        
        self.logger.info(
            f"Initialized optimized strategic field with grid size {grid_size}, "
            f"{n_strategies} strategies, sparse={use_sparse}, FFT={self.use_fft}"
        )
    
    def prepare_fft_kernel(self):
        """
        Prepare kernel for FFT-based diffusion
        """
        # Create diffusion kernel
        kernel = np.zeros((3, 3))
        kernel[1, 0] = kernel[0, 1] = kernel[2, 1] = kernel[1, 2] = self.diffusion_rate / 4
        kernel[1, 1] = 1 - self.diffusion_rate
        self.kernel = kernel
    
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
        import time
        start_time = time.time()
        
        # Update field based on agent positions
        if self.use_sparse:
            self._update_sparse(agents, resources, dt)
        else:
            self._update_dense(agents, resources, dt)
        
        # Calculate coherence
        coherence = self.calculate_coherence()
        self.coherence_history.append(coherence)
        
        # Track update time
        self.update_times.append(time.time() - start_time)
        
        # Track sparse density if using sparse matrices
        if self.use_sparse:
            total_elements = self.grid_size * self.grid_size * self.n_strategies
            nonzero_elements = sum(field.nnz for field in self.field)
            density = nonzero_elements / total_elements
            self.sparse_density.append(density)
        
        return coherence
    
    def _update_sparse(self, agents, resources, dt):
        """
        Update using sparse matrices
        
        Parameters:
        - agents: List of Agent objects
        - resources: Optional ResourceSystem object
        - dt: Time step size
        """
        # Each agent contributes to field based on position and strategy
        for agent in agents:
            x, y = agent.position.astype(int)
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                # Agent's contribution is weighted by energy if available
                weight = getattr(agent, 'energy', 1.0)
                for s in range(self.n_strategies):
                    self.field[s][x, y] = (1 - dt) * self.field[s][x, y] + dt * agent.strategy[s] * weight
                
                # Create wave at agent's position
                self.wave_field[x, y] += weight * 0.1
        
        # Resource influence if available
        if resources is not None:
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    resource_level = resources.get_resources_at(np.array([x, y]))
                    if resource_level > 0:
                        # Resources bias field toward certain strategies
                        bias = np.zeros(self.n_strategies)
                        bias[0] = resource_level  # Exploitation
                        for i in range(1, self.n_strategies):
                            bias[i] = (1 - resource_level) / (self.n_strategies - 1)  # Exploration
                        
                        # Apply resource bias
                        for s in range(self.n_strategies):
                            self.field[s][x, y] = (1 - resource_level * 0.1) * self.field[s][x, y] + resource_level * 0.1 * bias[s]
        
        # Normalize field
        self._normalize_sparse()
        
        # Apply diffusion
        if self.use_fft:
            self._diffuse_fft()
        else:
            self._diffuse_sparse()
        
        # Propagate wave field
        self._propagate_wave(dt)
    
    def _update_dense(self, agents, resources, dt):
        """
        Update using dense matrices
        
        Parameters:
        - agents: List of Agent objects
        - resources: Optional ResourceSystem object
        - dt: Time step size
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
        
        # Apply diffusion
        if self.use_fft:
            self._diffuse_fft_dense()
        else:
            self._diffuse_dense()
        
        # Propagate wave field
        self._propagate_wave(dt)
    
    def _normalize_sparse(self):
        """
        Normalize sparse field to ensure valid probability distributions
        """
        # For each position, ensure strategy probabilities sum to 1
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                total = sum(field[x, y] for field in self.field)
                if total > 0:
                    for s in range(self.n_strategies):
                        self.field[s][x, y] /= total
    
    def _diffuse_sparse(self):
        """
        Apply diffusion using sparse matrix operations
        """
        for s in range(self.n_strategies):
            # Convert to CSR format for efficient arithmetic
            field_csr = self.field[s].tocsr()
            new_field = sparse.lil_matrix((self.grid_size, self.grid_size))
            
            # Get non-zero elements
            rows, cols = field_csr.nonzero()
            
            # Update only non-zero elements and their neighbors
            for r, c in zip(rows, cols):
                val = field_csr[r, c] * (1 - self.diffusion_rate)
                new_field[r, c] += val
                
                # Update neighbors
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        new_field[nr, nc] += field_csr[r, c] * self.diffusion_rate / 4
            
            self.field[s] = new_field
    
    def _diffuse_fft(self):
        """
        Apply diffusion using FFT-based convolution
        """
        for s in range(self.n_strategies):
            # Convert sparse to dense for FFT
            dense_field = self.field[s].toarray()
            
            # Apply convolution using FFT
            dense_field = signal.convolve2d(dense_field, self.kernel, mode='same', boundary='wrap')
            
            # Convert back to sparse
            self.field[s] = sparse.lil_matrix(dense_field)
    
    def _diffuse_fft_dense(self):
        """
        Apply diffusion using FFT-based convolution for dense fields
        """
        for s in range(self.n_strategies):
            self.field[:, :, s] = signal.convolve2d(
                self.field[:, :, s], 
                self.kernel, 
                mode='same', 
                boundary='wrap'
            )
    
    def _diffuse_dense(self):
        """
        Apply diffusion using dense matrix operations
        """
        new_field = self.field.copy()
        for x in range(1, self.grid_size-1):
            for y in range(1, self.grid_size-1):
                new_field[x, y] = (1-self.diffusion_rate) * self.field[x, y] + self.diffusion_rate/4 * (
                    self.field[x-1, y] + self.field[x+1, y] + 
                    self.field[x, y-1] + self.field[x, y+1]
                )
        self.field = new_field
    
    def _propagate_wave(self, dt):
        """
        Propagate wave field
        
        Parameters:
        - dt: Time step size
        """
        wave_speed = 1.0
        new_wave = self.wave_field.copy()
        for x in range(1, self.grid_size-1):
            for y in range(1, self.grid_size-1):
                new_wave[x, y] = (1-wave_speed*dt) * self.wave_field[x, y] + wave_speed*dt/4 * (
                    self.wave_field[x-1, y] + self.wave_field[x+1, y] + 
                    self.wave_field[x, y-1] + self.wave_field[x, y+1]
                )
        self.wave_field = new_wave * 0.99  # Decay
    
    def calculate_coherence(self):
        """
        Calculate field coherence
        
        Coherence is measured by the average standard deviation across strategies.
        Higher std means more coherent (one strategy dominates at each point).
        
        Returns:
        - coherence: Coherence measure (0 to 1)
        """
        if self.use_sparse:
            # For sparse fields, calculate coherence from strategy distributions
            total_std = 0
            count = 0
            
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    strategy_values = np.array([field[x, y] for field in self.field])
                    if np.sum(strategy_values) > 0:
                        total_std += np.std(strategy_values)
                        count += 1
            
            return total_std / count if count > 0 else 0
        else:
            # For dense fields, use the original calculation
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
            if self.use_sparse:
                return np.array([field[x, y] for field in self.field])
            else:
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
    
    def get_performance_stats(self):
        """
        Get performance statistics
        
        Returns:
        - stats: Dictionary of performance statistics
        """
        stats = {
            'avg_update_time': np.mean(self.update_times) if self.update_times else 0,
            'max_update_time': np.max(self.update_times) if self.update_times else 0,
            'min_update_time': np.min(self.update_times) if self.update_times else 0,
            'use_sparse': self.use_sparse,
            'use_fft': self.use_fft
        }
        
        if self.use_sparse and self.sparse_density:
            stats['avg_sparse_density'] = np.mean(self.sparse_density)
            stats['memory_reduction'] = 1 - np.mean(self.sparse_density)
        
        return stats
    
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
        
        # Convert sparse to dense if needed
        if self.use_sparse:
            if strategy_idx is not None:
                field_to_plot = self.field[strategy_idx].toarray()
            else:
                # Create dense array for dominant strategy visualization
                dominant_field = np.zeros((self.grid_size, self.grid_size))
                for x in range(self.grid_size):
                    for y in range(self.grid_size):
                        strategy_values = np.array([field[x, y] for field in self.field])
                        dominant_field[x, y] = np.argmax(strategy_values)
                field_to_plot = dominant_field
        else:
            if strategy_idx is not None:
                field_to_plot = self.field[:, :, strategy_idx]
            else:
                field_to_plot = np.argmax(self.field, axis=2)
        
        if strategy_idx is not None:
            # Visualize specific strategy component
            im = axs[0].imshow(field_to_plot, cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(im, ax=axs[0], label=f'Strategy {strategy_idx+1} Strength')
            if title:
                axs[0].set_title(title, fontsize=14)
            else:
                axs[0].set_title(f'Strategic Field - Strategy {strategy_idx+1}', fontsize=14)
        else:
            # Visualize dominant strategy at each point
            # Create custom colormap for strategies
            colors = plt.cm.tab10(np.linspace(0, 1, self.n_strategies))
            cmap = plt.cm.colors.LinearSegmentedColormap.from_list('strategies', colors, N=self.n_strategies)
            
            im = axs[0].imshow(field_to_plot, cmap=cmap, interpolation='nearest', vmin=0, vmax=self.n_strategies-1)
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
    
    def visualize_performance(self):
        """
        Visualize performance metrics
        
        Returns:
        - fig: Matplotlib figure object
        """
        if not self.update_times:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot update times
        ax1.plot(self.update_times, linewidth=2)
        ax1.set_title('Update Time per Step', fontsize=14)
        ax1.set_xlabel('Step', fontsize=12)
        ax1.set_ylabel('Time (seconds)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Add stats
        stats = self.get_performance_stats()
        ax1.text(
            0.05, 0.95, 
            f"Avg: {stats['avg_update_time']:.4f}s\nMax: {stats['max_update_time']:.4f}s",
            transform=ax1.transAxes,
            verticalalignment='top',
            fontsize=10
        )
        
        # Plot sparse density if available
        if self.use_sparse and self.sparse_density:
            ax2.plot(self.sparse_density, linewidth=2)
            ax2.set_title('Sparse Matrix Density', fontsize=14)
            ax2.set_xlabel('Step', fontsize=12)
            ax2.set_ylabel('Density', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Add memory reduction info
            memory_reduction = 1 - np.mean(self.sparse_density)
            ax2.text(
                0.05, 0.95, 
                f"Memory reduction: {memory_reduction:.1%}",
                transform=ax2.transAxes,
                verticalalignment='top',
                fontsize=10
            )
        else:
            ax2.set_title('Memory Usage (Dense Matrix)', fontsize=14)
            ax2.text(
                0.5, 0.5, 
                f"Using dense matrices\nMemory: {self.grid_size}x{self.grid_size}x{self.n_strategies} = {self.grid_size**2 * self.n_strategies} elements",
                transform=ax2.transAxes,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=12
            )
            ax2.axis('off')
        
        plt.tight_layout()
        return fig
