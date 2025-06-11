#!/usr/bin/env python3
"""
Quantum-Inspired Agent Implementation

This module implements a quantum-inspired agent with superposition of strategies,
measurement-based decision making, and entanglement-like correlations.
"""

import numpy as np
import logging

class QuantumAgent:
    """
    Agent with quantum-inspired decision making
    
    This class implements an agent that uses quantum-inspired concepts:
    - Strategies represented as complex amplitudes (superposition)
    - Decision making as measurement/collapse
    - Entanglement-like correlations between agents
    """
    def __init__(self, agent_id, position, n_strategies=3, logger=None):
        """
        Initialize the quantum agent
        
        Parameters:
        - agent_id: Unique identifier for the agent
        - position: Initial position [x, y]
        - n_strategies: Number of strategies
        """
        self.agent_id = agent_id
        self.position = np.array(position)
        self.n_strategies = n_strategies
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize in superposition of strategies
        self.amplitudes = np.ones(n_strategies, dtype=complex) / np.sqrt(n_strategies)
        
        # Classical strategy (for compatibility with existing code)
        self.strategy = np.abs(self.amplitudes)**2
        
        # Entanglement partners
        self.entangled_partners = []
        
        # History
        self.measurement_history = []
        self.phase_history = []
        
        # Energy dynamics (for compatibility)
        self.energy = 1.0
        self.age = 0
        
        self.logger.debug(f"Initialized quantum agent {agent_id} at position {position}")
    
    def update_amplitudes(self, field_influence=None, memory_influence=None, dt=0.1):
        """
        Update quantum amplitudes
        
        Parameters:
        - field_influence: Influence from strategic field (complex amplitudes)
        - memory_influence: Influence from memory (complex amplitudes)
        - dt: Time step
        """
        # Phase evolution
        self.amplitudes *= np.exp(1j * dt * np.arange(self.n_strategies))
        
        # Field influence (like external potential)
        if field_influence is not None:
            # Ensure field_influence is complex
            if not np.iscomplexobj(field_influence):
                field_influence = field_influence.astype(complex)
            
            # Apply field influence through phase adjustment
            field_phase = np.angle(field_influence)
            self.amplitudes *= np.exp(1j * dt * field_phase)
        
        # Memory influence (like measurement)
        if memory_influence is not None:
            # Convert to complex if needed
            if not np.iscomplexobj(memory_influence):
                memory_influence = np.sqrt(memory_influence) * np.exp(1j * np.zeros(self.n_strategies))
            
            # Partial measurement based on memory
            memory_prob = np.abs(memory_influence)**2
            memory_phase = np.angle(memory_influence)
            
            # Apply partial collapse toward memory state
            self.amplitudes = np.sqrt(memory_prob) * np.exp(1j * (0.8 * np.angle(self.amplitudes) + 0.2 * memory_phase))
        
        # Normalize
        self.amplitudes /= np.linalg.norm(self.amplitudes)
        
        # Update classical strategy
        self.strategy = np.abs(self.amplitudes)**2
        
        # Record phase
        self.phase_history.append(np.angle(self.amplitudes).copy())
    
    def make_decision(self):
        """
        Make a decision by measuring quantum state
        
        Returns:
        - decision: Strategy index chosen
        """
        # Calculate probabilities
        probabilities = np.abs(self.amplitudes)**2
        
        # Make measurement
        decision = np.random.choice(self.n_strategies, p=probabilities)
        
        # Collapse state
        self.amplitudes = np.zeros(self.n_strategies, dtype=complex)
        self.amplitudes[decision] = 1.0
        
        # Update classical strategy
        self.strategy = np.abs(self.amplitudes)**2
        
        # Record measurement
        self.measurement_history.append({
            'probabilities': probabilities.copy(),
            'decision': decision
        })
        
        # Update entangled partners
        for partner in self.entangled_partners:
            partner.entanglement_update(decision)
        
        return decision
    
    def entanglement_update(self, partner_decision):
        """
        Update based on entangled partner's decision
        
        Parameters:
        - partner_decision: Decision made by entangled partner
        """
        # Implement entanglement effect
        # This is a simplified model - real quantum entanglement is more complex
        
        # Bias amplitudes toward complementary strategy
        complementary = (partner_decision + 1) % self.n_strategies
        
        # Increase amplitude of complementary strategy
        self.amplitudes *= 0.8
        self.amplitudes[complementary] = 0.6 + 0.4j
        
        # Normalize
        self.amplitudes /= np.linalg.norm(self.amplitudes)
        
        # Update classical strategy
        self.strategy = np.abs(self.amplitudes)**2
    
    def entangle_with(self, other_agent):
        """
        Create entanglement-like correlation with another agent
        
        Parameters:
        - other_agent: Another QuantumAgent to entangle with
        
        Returns:
        - success: Whether entanglement was successful
        """
        if other_agent in self.entangled_partners:
            return False
        
        # Add to entangled partners
        self.entangled_partners.append(other_agent)
        other_agent.entangled_partners.append(self)
        
        # Create initial entanglement
        # Create a Bell-like state where strategies are anti-correlated
        
        # Reset amplitudes
        self.amplitudes = np.zeros(self.n_strategies, dtype=complex)
        other_agent.amplitudes = np.zeros(self.n_strategies, dtype=complex)
        
        # Create entangled state
        for i in range(self.n_strategies):
            j = (i + 1) % self.n_strategies  # Complementary strategy
            
            # Set amplitudes for Bell-like state
            self.amplitudes[i] = 1.0 / np.sqrt(self.n_strategies)
            other_agent.amplitudes[j] = 1.0 / np.sqrt(self.n_strategies)
        
        # Update classical strategies
        self.strategy = np.abs(self.amplitudes)**2
        other_agent.strategy = np.abs(other_agent.amplitudes)**2
        
        return True
    
    def move(self, grid_size, quantum_field=None, resource_field=None, move_distance=1):
        """
        Move based on quantum field, resource gradient, or randomly
        
        Parameters:
        - grid_size: Size of the environment grid
        - quantum_field: Optional QuantumStrategicField object
        - resource_field: Optional ResourceSystem object
        - move_distance: Maximum distance to move in one step
        
        Returns:
        - new_position: The agent's new position
        """
        self.age += 1
        
        # Default to random movement
        move_type = 'random'
        
        if quantum_field is not None:
            # Get interference pattern gradient at current position
            x, y = self.position.astype(int)
            if 0 <= x < grid_size-1 and 0 <= y < grid_size-1:
                # Calculate gradient using central difference
                interference = np.zeros((self.n_strategies, 3, 3))
                for dx in range(3):
                    for dy in range(3):
                        nx, ny = x + dx - 1, y + dy - 1
                        if 0 <= nx < grid_size and 0 <= ny < grid_size:
                            for s in range(self.n_strategies):
                                interference[s, dx, dy] = np.abs(quantum_field.psi[nx, ny, s])**2
                
                # Weight gradient by strategy probabilities
                gradient_x = 0
                gradient_y = 0
                for s in range(self.n_strategies):
                    # Calculate gradient for this strategy
                    s_grad_x = interference[s, 2, 1] - interference[s, 0, 1]
                    s_grad_y = interference[s, 1, 2] - interference[s, 1, 0]
                    
                    # Weight by strategy probability
                    gradient_x += s_grad_x * self.strategy[s]
                    gradient_y += s_grad_y * self.strategy[s]
                
                # Normalize gradient
                gradient = np.array([gradient_x, gradient_y])
                gradient_norm = np.linalg.norm(gradient)
                
                if gradient_norm > 0.1:
                    # Move along gradient
                    move_direction = gradient / gradient_norm
                    self.position += move_direction * move_distance
                    move_type = 'field'
        
        if move_type == 'random':
            # Random movement
            direction = np.random.rand(2) * 2 - 1
            direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else direction
            self.position += direction * move_distance
        
        # Ensure position is within bounds
        self.position = np.clip(self.position, 0, grid_size - 1)
        
        return self.position
    
    def visualize_amplitudes(self):
        """
        Visualize quantum amplitudes
        
        Returns:
        - fig: Matplotlib figure object
        """
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot probabilities
        probabilities = np.abs(self.amplitudes)**2
        ax1.bar(range(self.n_strategies), probabilities)
        ax1.set_title('Strategy Probabilities', fontsize=14)
        ax1.set_xlabel('Strategy', fontsize=12)
        ax1.set_ylabel('Probability', fontsize=12)
        ax1.set_xticks(range(self.n_strategies))
        ax1.grid(True, alpha=0.3)
        
        # Plot phases
        phases = np.angle(self.amplitudes)
        ax2.bar(range(self.n_strategies), phases)
        ax2.set_title('Strategy Phases', fontsize=14)
        ax2.set_xlabel('Strategy', fontsize=12)
        ax2.set_ylabel('Phase (radians)', fontsize=12)
        ax2.set_xticks(range(self.n_strategies))
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def visualize_phase_history(self):
        """
        Visualize phase history
        
        Returns:
        - fig: Matplotlib figure object
        """
        import matplotlib.pyplot as plt
        
        if not self.phase_history:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Convert to numpy array
        phase_history = np.array(self.phase_history)
        
        # Plot phase history for each strategy
        for s in range(self.n_strategies):
            ax.plot(phase_history[:, s], label=f'Strategy {s+1}')
        
        ax.set_title('Phase History', fontsize=14)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Phase (radians)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
