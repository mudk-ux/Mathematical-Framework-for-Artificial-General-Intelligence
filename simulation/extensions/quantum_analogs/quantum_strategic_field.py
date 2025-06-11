#!/usr/bin/env python3
"""
Quantum-Inspired Strategic Field Implementation

This module implements a quantum-inspired version of the strategic field,
using complex-valued wave functions to model strategic information propagation
and interference patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging

class QuantumStrategicField:
    """
    Strategic field implementation using quantum-inspired wave functions
    
    This class models strategic information as complex-valued wave functions,
    enabling interference patterns and measurement-like operations.
    """
    def __init__(self, grid_size=50, n_strategies=3, diffusion_rate=0.2, 
                 wave_speed=1.0, logger=None):
        """
        Initialize the quantum strategic field
        
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
        
        # Initialize complex-valued wave functions for each strategy
        self.psi = np.ones((grid_size, grid_size, n_strategies), dtype=complex) / np.sqrt(n_strategies)
        
        # Add random phases to create initial interference patterns
        phases = np.random.uniform(0, 2*np.pi, (grid_size, grid_size, n_strategies))
        self.psi *= np.exp(1j * phases)
        
        # Phase factors for wave evolution
        self.phase_factors = np.exp(1j * np.random.uniform(0, 2*np.pi, n_strategies))
        
        # For tracking interference patterns
        self.interference_history = []
        self.coherence_history = []
        
        # For tracking measurement outcomes
        self.measurement_history = []
        
        self.logger.info(f"Initialized quantum strategic field with grid size {grid_size} and {n_strategies} strategies")
    
    def update(self, agents, dt=0.1):
        """
        Update quantum strategic field
        
        Parameters:
        - agents: List of Agent objects
        - dt: Time step
        
        Returns:
        - interference: Current interference pattern
        """
        # Apply phase evolution
        for s in range(self.n_strategies):
            self.psi[:, :, s] *= np.exp(1j * dt * s)
        
        # Agent contributions (measurement-like interactions)
        for agent in agents:
            x, y = agent.position.astype(int)
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                # Agent's strategy acts like a measurement operator
                strategy_amplitudes = np.sqrt(agent.strategy)
                
                # Apply measurement-like interaction
                self.psi[x, y] = strategy_amplitudes * np.exp(1j * np.angle(self.psi[x, y]))
                
                # Normalize
                self.psi[x, y] /= np.linalg.norm(self.psi[x, y])
        
        # Apply diffusion (similar to Schrödinger equation)
        new_psi = self.psi.copy()
        for x in range(1, self.grid_size-1):
            for y in range(1, self.grid_size-1):
                laplacian = (
                    self.psi[x-1, y] + self.psi[x+1, y] + 
                    self.psi[x, y-1] + self.psi[x, y+1] - 
                    4 * self.psi[x, y]
                )
                new_psi[x, y] += 1j * dt * self.diffusion_rate * laplacian
        
        self.psi = new_psi
        
        # Normalize wave functions
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                norm = np.linalg.norm(self.psi[x, y])
                if norm > 0:
                    self.psi[x, y] /= norm
        
        # Calculate interference pattern
        interference = np.zeros((self.grid_size, self.grid_size))
        for s in range(self.n_strategies):
            interference += np.abs(self.psi[:, :, s])**2
        
        # Store interference pattern periodically
        if len(self.interference_history) % 10 == 0:
            self.interference_history.append(interference.copy())
        
        # Calculate coherence
        coherence = self.calculate_coherence()
        self.coherence_history.append(coherence)
        
        return interference
    
    def calculate_coherence(self):
        """
        Calculate field coherence
        
        Coherence is measured by the average standard deviation of probability
        distributions across strategies.
        
        Returns:
        - coherence: Coherence measure (0 to 1)
        """
        # Calculate probability distributions
        probabilities = np.abs(self.psi)**2
        
        # Calculate standard deviation across strategies at each point
        std_dev = np.std(probabilities, axis=2)
        
        # Average across the grid
        return np.mean(std_dev)
    
    def get_probability_distribution(self, position):
        """
        Get probability distribution at position
        
        Parameters:
        - position: Position array [x, y]
        
        Returns:
        - probabilities: Probability distribution across strategies
        """
        x, y = position.astype(int)
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            # Born rule: probability is amplitude squared
            return np.abs(self.psi[x, y])**2
        else:
            return np.ones(self.n_strategies) / self.n_strategies
    
    def measure(self, position):
        """
        Perform a measurement at position
        
        Parameters:
        - position: Position array [x, y]
        
        Returns:
        - outcome: Measured strategy index
        - probabilities: Probability distribution before measurement
        """
        x, y = position.astype(int)
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            # Get probabilities
            probabilities = np.abs(self.psi[x, y])**2
            
            # Perform measurement (collapse)
            outcome = np.random.choice(self.n_strategies, p=probabilities)
            
            # Collapse wave function
            self.psi[x, y] = np.zeros(self.n_strategies, dtype=complex)
            self.psi[x, y, outcome] = 1.0
            
            # Record measurement
            self.measurement_history.append({
                'position': (x, y),
                'probabilities': probabilities.copy(),
                'outcome': outcome
            })
            
            return outcome, probabilities
        else:
            return np.random.randint(self.n_strategies), np.ones(self.n_strategies) / self.n_strategies
    
    def create_double_slit_pattern(self, slit_width=2, slit_separation=10, source_distance=15):
        """
        Create a double-slit interference pattern
        
        Parameters:
        - slit_width: Width of each slit
        - slit_separation: Distance between slits
        - source_distance: Distance from source to slits
        
        Returns:
        - interference: Interference pattern
        """
        # Reset field
        self.psi = np.zeros((self.grid_size, self.grid_size, self.n_strategies), dtype=complex)
        
        # Calculate center position
        center_x = self.grid_size // 2
        center_y = self.grid_size // 2
        
        # Create source
        source_y = center_y - source_distance
        if 0 <= source_y < self.grid_size:
            self.psi[center_x, source_y] = np.ones(self.n_strategies, dtype=complex) / np.sqrt(self.n_strategies)
        
        # Create barrier with slits
        barrier_y = center_y
        for x in range(self.grid_size):
            # Skip the slits
            slit1_start = center_x - slit_separation//2 - slit_width//2
            slit1_end = center_x - slit_separation//2 + slit_width//2
            slit2_start = center_x + slit_separation//2 - slit_width//2
            slit2_end = center_x + slit_separation//2 + slit_width//2
            
            if (slit1_start <= x <= slit1_end) or (slit2_start <= x <= slit2_end):
                continue
            
            # Create barrier (zero amplitude)
            if 0 <= barrier_y < self.grid_size:
                self.psi[x, barrier_y] = np.zeros(self.n_strategies, dtype=complex)
        
        # Propagate waves
        for step in range(100):
            self.update([], dt=0.1)
        
        # Calculate interference pattern
        interference = np.zeros((self.grid_size, self.grid_size))
        for s in range(self.n_strategies):
            interference += np.abs(self.psi[:, :, s])**2
        
        return interference
    
    def visualize(self, strategy_idx=None, show_phase=False, title=None, agents=None):
        """
        Visualize the quantum strategic field
        
        Parameters:
        - strategy_idx: If provided, visualize only this strategy component
                       If None, visualize interference pattern
        - show_phase: Whether to show phase information
        - title: Optional title for the plot
        - agents: Optional list of agents to plot on the field
        
        Returns:
        - fig: Matplotlib figure object
        """
        if show_phase:
            fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        else:
            fig, axs = plt.subplots(1, 1, figsize=(10, 8))
            axs = [axs]
        
        if strategy_idx is not None:
            # Visualize specific strategy component
            probabilities = np.abs(self.psi[:, :, strategy_idx])**2
            im = axs[0].imshow(probabilities, cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(im, ax=axs[0], label=f'Strategy {strategy_idx+1} Probability')
            if title:
                axs[0].set_title(title, fontsize=14)
            else:
                axs[0].set_title(f'Quantum Strategic Field - Strategy {strategy_idx+1}', fontsize=14)
        else:
            # Visualize interference pattern
            interference = np.zeros((self.grid_size, self.grid_size))
            for s in range(self.n_strategies):
                interference += np.abs(self.psi[:, :, s])**2
            
            im = axs[0].imshow(interference, cmap='viridis')
            plt.colorbar(im, ax=axs[0], label='Interference Pattern')
            
            if title:
                axs[0].set_title(title, fontsize=14)
            else:
                axs[0].set_title('Quantum Strategic Field - Interference Pattern', fontsize=14)
        
        # Plot agents if provided
        if agents:
            agent_x = [agent.position[0] for agent in agents]
            agent_y = [agent.position[1] for agent in agents]
            axs[0].scatter(agent_y, agent_x, c='red', s=10, alpha=0.7)
        
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        
        # Visualize phase if requested
        if show_phase:
            if strategy_idx is not None:
                phases = np.angle(self.psi[:, :, strategy_idx])
                im2 = axs[1].imshow(phases, cmap='hsv', vmin=-np.pi, vmax=np.pi)
                plt.colorbar(im2, ax=axs[1], label='Phase')
                axs[1].set_title(f'Phase - Strategy {strategy_idx+1}', fontsize=14)
            else:
                # Show average phase
                avg_phase = np.zeros((self.grid_size, self.grid_size))
                for s in range(self.n_strategies):
                    avg_phase += np.angle(self.psi[:, :, s])
                avg_phase /= self.n_strategies
                
                im2 = axs[1].imshow(avg_phase, cmap='hsv', vmin=-np.pi, vmax=np.pi)
                plt.colorbar(im2, ax=axs[1], label='Average Phase')
                axs[1].set_title('Average Phase', fontsize=14)
            
            axs[1].set_xticks([])
            axs[1].set_yticks([])
            
            # Plot agents if provided
            if agents:
                agent_x = [agent.position[0] for agent in agents]
                agent_y = [agent.position[1] for agent in agents]
                axs[1].scatter(agent_y, agent_x, c='white', s=10, alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def visualize_double_slit(self):
        """
        Visualize double-slit interference pattern
        
        Returns:
        - fig: Matplotlib figure object
        """
        interference = self.create_double_slit_pattern()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(interference, cmap='viridis')
        plt.colorbar(im, ax=ax, label='Interference Pattern')
        
        ax.set_title('Double-Slit Interference Pattern', fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        
        return fig
    
    def visualize_coherence(self):
        """
        Visualize coherence over time
        
        Returns:
        - fig: Matplotlib figure object
        """
        if not self.coherence_history:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.coherence_history, linewidth=2)
        
        ax.set_title('Quantum Field Coherence Over Time', fontsize=14)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Coherence', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def visualize_measurement_statistics(self):
        """
        Visualize measurement statistics
        
        Returns:
        - fig: Matplotlib figure object
        """
        if not self.measurement_history:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Count outcomes
        outcomes = [m['outcome'] for m in self.measurement_history]
        unique, counts = np.unique(outcomes, return_counts=True)
        
        # Plot outcome histogram
        ax1.bar(unique, counts / len(outcomes))
        ax1.set_title('Measurement Outcomes', fontsize=14)
        ax1.set_xlabel('Strategy', fontsize=12)
        ax1.set_ylabel('Probability', fontsize=12)
        ax1.set_xticks(range(self.n_strategies))
        ax1.grid(True, alpha=0.3)
        
        # Plot measurement positions
        positions = np.array([m['position'] for m in self.measurement_history])
        if len(positions) > 0:
            ax2.scatter(positions[:, 1], positions[:, 0], c=outcomes, cmap='tab10', alpha=0.7)
            ax2.set_title('Measurement Positions', fontsize=14)
            ax2.set_xlim(0, self.grid_size)
            ax2.set_ylim(self.grid_size, 0)  # Invert y-axis to match image coordinates
            ax2.set_aspect('equal')
        else:
            ax2.set_title('No Measurements Recorded', fontsize=14)
        
        plt.tight_layout()
        return fig
