#!/usr/bin/env python3
"""
Nash Equilibrium Validator for the unified MMAI system

This module implements the mass-action interpretation of Nash equilibrium
as described in "Steps Towards AGI," validating that strategic patterns
represent true Nash equilibria proportional to system growth.
"""

import numpy as np
import logging
import matplotlib.pyplot as plt

class NashValidator:
    """
    Validates that strategic patterns represent true Nash equilibria
    
    The Nash validator implements the mass-action interpretation of Nash equilibrium,
    tracking how strategic choices distribute across populations and validating
    that they converge to stable equilibria proportional to system growth.
    """
    def __init__(self, n_strategies=3, equilibrium_threshold=0.1, growth_rate=0.05, logger=None):
        """
        Initialize the Nash validator
        
        Parameters:
        - n_strategies: Number of strategies
        - equilibrium_threshold: Maximum distance to consider as equilibrium
        - growth_rate: System growth rate (g)
        """
        self.n_strategies = n_strategies
        self.equilibrium_threshold = equilibrium_threshold
        self.growth_rate = growth_rate
        self.logger = logger or logging.getLogger(__name__)
        
        # History tracking
        self.nash_distance_history = []
        self.payoff_matrix_history = []
        self.best_response_history = []
        self.equilibrium_history = []
        self.population_history = []
        
        # For mass-action interpretation
        self.strategy_distribution_history = []
        self.perceived_payoff_history = []
        
        self.logger.info(f"Initialized Nash validator with {n_strategies} strategies")
    
    def calculate_payoff_matrix(self, agents):
        """
        Calculate the payoff matrix based on agent strategies and outcomes
        
        Parameters:
        - agents: List of Agent objects
        
        Returns:
        - payoff_matrix: Matrix where payoff_matrix[i,j] is the payoff for strategy i against strategy j
        """
        payoff_matrix = np.zeros((self.n_strategies, self.n_strategies))
        
        # Extract strategies and outcomes
        strategies = [agent.strategy for agent in agents]
        outcomes = [np.mean(agent.outcomes) if hasattr(agent, 'outcomes') and agent.outcomes else 0.5 for agent in agents]
        
        # For each pair of strategies, calculate the average payoff
        for i in range(self.n_strategies):
            for j in range(self.n_strategies):
                # Find agents using primarily strategy i against agents using primarily strategy j
                i_users = [idx for idx, s in enumerate(strategies) if np.argmax(s) == i]
                j_users = [idx for idx, s in enumerate(strategies) if np.argmax(s) == j]
                
                if i_users and j_users:
                    # Calculate average payoff for i against j
                    payoffs = []
                    for i_idx in i_users:
                        for j_idx in j_users:
                            if i_idx != j_idx:  # Avoid self-interaction
                                # Weight by strategy probability
                                payoff = outcomes[i_idx] * strategies[i_idx][i] * strategies[j_idx][j]
                                payoffs.append(payoff)
                    
                    if payoffs:
                        payoff_matrix[i, j] = np.mean(payoffs)
                    else:
                        # Default payoff if no interactions
                        payoff_matrix[i, j] = 0.5
                else:
                    # Default payoff if no users of these strategies
                    payoff_matrix[i, j] = 0.5
        
        # Store payoff matrix
        self.payoff_matrix_history.append(payoff_matrix)
        
        return payoff_matrix
    
    def calculate_nash_distance(self, agents, payoff_matrix=None, population_size=None):
        """
        Calculate how far the current strategy distribution is from Nash equilibrium
        
        Parameters:
        - agents: List of Agent objects
        - payoff_matrix: Optional payoff matrix (calculated if not provided)
        - population_size: Optional population size for growth tracking
        
        Returns:
        - nash_distance: A measure of distance from Nash equilibrium
        - best_responses: Dictionary mapping agent index to its best response strategy
        """
        if payoff_matrix is None:
            payoff_matrix = self.calculate_payoff_matrix(agents)
        
        # Calculate average strategy distribution
        strategies = [agent.strategy for agent in agents]
        if not strategies:  # Check if the list is empty
            self.logger.warning("No agents available for Nash distance calculation. Using uniform strategy.")
            avg_strategy = np.ones(self.n_strategies) / self.n_strategies
        else:
            avg_strategy = np.mean(strategies, axis=0)
        
        # Store strategy distribution
        self.strategy_distribution_history.append(avg_strategy)
        
        # Calculate expected payoff for each pure strategy against the average strategy
        expected_payoffs = np.zeros(self.n_strategies)
        for i in range(self.n_strategies):
            for j in range(self.n_strategies):
                expected_payoffs[i] += payoff_matrix[i, j] * avg_strategy[j]
        
        # Store perceived payoffs
        self.perceived_payoff_history.append(expected_payoffs)
        
        # Find best response (strategy with highest expected payoff)
        best_response_idx = np.argmax(expected_payoffs)
        best_response = np.zeros(self.n_strategies)
        best_response[best_response_idx] = 1.0
        
        # Calculate Nash distance as the difference between current average strategy and best response
        nash_distance = np.linalg.norm(avg_strategy - best_response)
        
        # Store Nash distance
        self.nash_distance_history.append(nash_distance)
        
        # Store population size if provided
        if population_size is not None:
            self.population_history.append(population_size)
        
        # Calculate best response for each agent
        best_responses = {}
        for idx, agent in enumerate(agents):
            # Calculate expected payoff against all other agents
            element_expected_payoffs = np.zeros(self.n_strategies)
            for i in range(self.n_strategies):
                for other_idx, other_agent in enumerate(agents):
                    if other_idx != idx:
                        for j in range(self.n_strategies):
                            element_expected_payoffs[i] += payoff_matrix[i, j] * other_agent.strategy[j] / (len(agents) - 1)
            
            # Find best response for this agent
            element_best_response_idx = np.argmax(element_expected_payoffs)
            element_best_response = np.zeros(self.n_strategies)
            element_best_response[element_best_response_idx] = 1.0
            best_responses[idx] = element_best_response
        
        # Store best responses
        self.best_response_history.append(best_responses)
        
        # Check if this is an equilibrium
        is_equilibrium = nash_distance <= self.equilibrium_threshold
        self.equilibrium_history.append(is_equilibrium)
        
        if is_equilibrium:
            self.logger.info(f"Nash equilibrium detected with distance {nash_distance:.4f}")
        
        return nash_distance, best_responses
    
    def is_nash_equilibrium(self, agents, threshold=None):
        """
        Determine if the current strategy distribution is a Nash equilibrium
        
        Parameters:
        - agents: List of Agent objects
        - threshold: Optional threshold (uses instance threshold if not provided)
        
        Returns:
        - is_equilibrium: Boolean indicating if this is a Nash equilibrium
        - nash_distance: The calculated Nash distance
        """
        if threshold is None:
            threshold = self.equilibrium_threshold
            
        nash_distance, _ = self.calculate_nash_distance(agents)
        is_equilibrium = nash_distance <= threshold
        
        return is_equilibrium, nash_distance
    
    def calculate_growth_proportional_equilibrium(self):
        """
        Calculate if Nash equilibrium is proportional to system growth
        
        According to the mass-action interpretation, Nash equilibria should emerge
        proportional to system growth rate.
        
        Returns:
        - is_proportional: Boolean indicating if equilibrium is proportional to growth
        - proportionality: Measure of proportionality
        """
        if not self.nash_distance_history or not self.population_history:
            return False, 0.0
        
        # Calculate population growth rate
        if len(self.population_history) > 1:
            growth_rates = [
                (self.population_history[i] - self.population_history[i-1]) / self.population_history[i-1]
                for i in range(1, len(self.population_history))
            ]
            avg_growth_rate = np.mean(growth_rates)
        else:
            avg_growth_rate = self.growth_rate
        
        # Calculate equilibrium convergence rate
        if len(self.nash_distance_history) > 1:
            convergence_rates = [
                (self.nash_distance_history[i-1] - self.nash_distance_history[i]) / self.nash_distance_history[i-1]
                for i in range(1, len(self.nash_distance_history))
                if self.nash_distance_history[i-1] > 0
            ]
            avg_convergence_rate = np.mean(convergence_rates) if convergence_rates else 0
        else:
            avg_convergence_rate = 0
        
        # Calculate proportionality
        # Perfect proportionality would be avg_convergence_rate ≈ avg_growth_rate
        proportionality = 1.0 - min(1.0, abs(avg_convergence_rate - avg_growth_rate) / max(0.001, avg_growth_rate))
        
        # Determine if proportional (within 20%)
        is_proportional = proportionality > 0.8
        
        if is_proportional:
            self.logger.info(f"Growth-proportional equilibrium detected: {proportionality:.2f}")
        
        return is_proportional, proportionality
    
    def visualize_nash_distance(self):
        """
        Visualize Nash distance over time
        
        Returns:
        - fig: Matplotlib figure object
        """
        if not self.nash_distance_history:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.nash_distance_history, linewidth=2)
        ax.axhline(y=self.equilibrium_threshold, color='r', linestyle='--', alpha=0.7, label=f'Equilibrium Threshold ({self.equilibrium_threshold})')
        
        ax.set_title('Nash Distance Over Time', fontsize=14)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Nash Distance', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def visualize_strategy_distribution(self):
        """
        Visualize strategy distribution over time
        
        Returns:
        - fig: Matplotlib figure object
        """
        if not self.strategy_distribution_history:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Convert to numpy array for easier slicing
        strategy_history = np.array(self.strategy_distribution_history)
        
        for i in range(self.n_strategies):
            ax.plot(strategy_history[:, i], label=f'Strategy {i+1}', linewidth=2)
        
        ax.set_title('Strategy Distribution Over Time', fontsize=14)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Strategy Probability', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def visualize_payoff_matrix(self, index=-1):
        """
        Visualize the payoff matrix
        
        Parameters:
        - index: Index of payoff matrix to visualize (-1 for latest)
        
        Returns:
        - fig: Matplotlib figure object
        """
        if not self.payoff_matrix_history:
            return None
        
        payoff_matrix = self.payoff_matrix_history[index]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(payoff_matrix, cmap='viridis')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Payoff', rotation=270, labelpad=15)
        
        # Add labels
        ax.set_title('Payoff Matrix', fontsize=14)
        ax.set_xlabel('Strategy (Column Player)', fontsize=12)
        ax.set_ylabel('Strategy (Row Player)', fontsize=12)
        
        # Add ticks
        ax.set_xticks(np.arange(self.n_strategies))
        ax.set_yticks(np.arange(self.n_strategies))
        ax.set_xticklabels([f'S{i+1}' for i in range(self.n_strategies)])
        ax.set_yticklabels([f'S{i+1}' for i in range(self.n_strategies)])
        
        # Add values in cells
        for i in range(self.n_strategies):
            for j in range(self.n_strategies):
                ax.text(j, i, f'{payoff_matrix[i, j]:.2f}', 
                        ha='center', va='center', 
                        color='white' if payoff_matrix[i, j] < 0.5 else 'black')
        
        plt.tight_layout()
        return fig
    
    def visualize_growth_proportional_equilibrium(self):
        """
        Visualize the relationship between growth rate and equilibrium convergence
        
        Returns:
        - fig: Matplotlib figure object
        """
        if not self.nash_distance_history or not self.population_history:
            return None
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot Nash distance
        color = 'tab:red'
        ax1.set_xlabel('Time Step', fontsize=12)
        ax1.set_ylabel('Nash Distance', color=color, fontsize=12)
        ax1.plot(self.nash_distance_history, color=color, linewidth=2, label='Nash Distance')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.axhline(y=self.equilibrium_threshold, color='r', linestyle='--', alpha=0.7, label=f'Equilibrium Threshold ({self.equilibrium_threshold})')
        
        # Create second y-axis for population
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Population Size', color=color, fontsize=12)
        ax2.plot(self.population_history, color=color, linewidth=2, label='Population Size')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Add title
        is_proportional, proportionality = self.calculate_growth_proportional_equilibrium()
        title = f'Growth-Proportional Equilibrium Analysis (Proportionality: {proportionality:.2f})'
        plt.title(title, fontsize=14)
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
