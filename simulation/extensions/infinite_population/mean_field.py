#!/usr/bin/env python3
"""
Mean Field Approximation for Infinite Population Dynamics

This module implements mean-field approximation techniques to model infinite
population dynamics as described in "Steps Towards AGI," bridging the gap
between finite agent simulations and theoretical infinite populations.
"""

import numpy as np
import logging
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class MeanFieldApproximation:
    """
    Implements mean-field approximation for infinite population dynamics
    
    This class models population-level dynamics using differential equations
    based on the PPP (Population-Payoff-Perception) dynamics from the theory.
    """
    def __init__(self, n_strategies=3, growth_rate=0.05, perception_rate=0.1, logger=None):
        """
        Initialize the mean field approximation
        
        Parameters:
        - n_strategies: Number of strategies
        - growth_rate: System growth rate (g)
        - perception_rate: Rate of perception adjustment (δ)
        """
        self.n_strategies = n_strategies
        self.growth_rate = growth_rate
        self.perception_rate = perception_rate
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize strategy distribution and perceived payoffs
        self.strategy_distribution = np.ones(n_strategies) / n_strategies
        self.perceived_payoffs = np.ones(n_strategies) / n_strategies
        
        # History tracking
        self.strategy_history = []
        self.payoff_history = []
        self.nash_distance_history = []
        self.time_points = []
        
        self.logger.info(f"Initialized mean field approximation with {n_strategies} strategies")
    
    def update(self, payoff_matrix, dt=0.1):
        """
        Update strategy distribution using PPP dynamics
        
        Parameters:
        - payoff_matrix: Game payoff matrix
        - dt: Time step
        
        Returns:
        - nash_distance: Distance from Nash equilibrium
        """
        # Calculate expected payoffs
        expected_payoffs = np.zeros(self.n_strategies)
        for i in range(self.n_strategies):
            for j in range(self.n_strategies):
                expected_payoffs[i] += payoff_matrix[i, j] * self.strategy_distribution[j]
        
        # Update perceived payoffs (PPP dynamics)
        self.perceived_payoffs = (1 - self.perception_rate * dt) * self.perceived_payoffs + self.perception_rate * dt * expected_payoffs
        
        # Calculate strategy adjustment (replicator dynamics)
        avg_payoff = np.sum(self.strategy_distribution * self.perceived_payoffs)
        adjustment = self.strategy_distribution * (self.perceived_payoffs - avg_payoff)
        
        # Apply growth-proportional adjustment
        self.strategy_distribution += dt * self.growth_rate * adjustment
        
        # Normalize
        self.strategy_distribution = np.maximum(0, self.strategy_distribution)
        self.strategy_distribution /= np.sum(self.strategy_distribution)
        
        # Calculate Nash distance
        best_response_idx = np.argmax(self.perceived_payoffs)
        best_response = np.zeros(self.n_strategies)
        best_response[best_response_idx] = 1.0
        nash_distance = np.linalg.norm(self.strategy_distribution - best_response)
        
        # Store history
        self.strategy_history.append(self.strategy_distribution.copy())
        self.payoff_history.append(self.perceived_payoffs.copy())
        self.nash_distance_history.append(nash_distance)
        self.time_points.append(len(self.time_points) * dt)
        
        return nash_distance
    
    def simulate_continuous(self, payoff_matrix, t_span=(0, 100), t_points=1000):
        """
        Simulate continuous-time dynamics using ODE solver
        
        Parameters:
        - payoff_matrix: Game payoff matrix
        - t_span: Time span for simulation (start, end)
        - t_points: Number of time points
        
        Returns:
        - results: Dictionary of simulation results
        """
        # Define the ODE system for PPP dynamics
        def ppp_dynamics(t, y):
            # Extract strategy distribution and perceived payoffs
            s = y[:self.n_strategies]
            p = y[self.n_strategies:]
            
            # Calculate expected payoffs
            expected_payoffs = np.zeros(self.n_strategies)
            for i in range(self.n_strategies):
                for j in range(self.n_strategies):
                    expected_payoffs[i] += payoff_matrix[i, j] * s[j]
            
            # Update perceived payoffs
            dp_dt = self.perception_rate * (expected_payoffs - p)
            
            # Calculate average payoff
            avg_payoff = np.sum(s * p)
            
            # Update strategy distribution
            ds_dt = self.growth_rate * s * (p - avg_payoff)
            
            return np.concatenate([ds_dt, dp_dt])
        
        # Initial conditions
        y0 = np.concatenate([
            self.strategy_distribution,
            self.perceived_payoffs
        ])
        
        # Solve ODE system
        t_eval = np.linspace(t_span[0], t_span[1], t_points)
        solution = solve_ivp(
            ppp_dynamics,
            t_span,
            y0,
            method='RK45',
            t_eval=t_eval
        )
        
        # Extract results
        t = solution.t
        y = solution.y
        
        # Extract strategy distribution and perceived payoffs
        strategy_history = y[:self.n_strategies, :].T
        payoff_history = y[self.n_strategies:, :].T
        
        # Calculate Nash distance history
        nash_distance_history = []
        for i in range(len(t)):
            s = strategy_history[i]
            p = payoff_history[i]
            
            # Normalize strategy distribution
            s = np.maximum(0, s)
            s_sum = np.sum(s)
            if s_sum > 0:
                s = s / s_sum
            
            # Calculate Nash distance
            best_response_idx = np.argmax(p)
            best_response = np.zeros(self.n_strategies)
            best_response[best_response_idx] = 1.0
            nash_distance = np.linalg.norm(s - best_response)
            nash_distance_history.append(nash_distance)
        
        # Store results
        self.time_points = t
        self.strategy_history = strategy_history
        self.payoff_history = payoff_history
        self.nash_distance_history = nash_distance_history
        
        return {
            'time': t,
            'strategy_history': strategy_history,
            'payoff_history': payoff_history,
            'nash_distance_history': nash_distance_history
        }
    def visualize_strategy_distribution(self):
        """
        Visualize strategy distribution over time
        
        Returns:
        - fig: Matplotlib figure
        """
        if len(self.strategy_history) == 0:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Convert to numpy array for easier slicing
        strategy_history = np.array(self.strategy_history)
        
        for i in range(self.n_strategies):
            ax.plot(self.time_points, strategy_history[:, i], label=f'Strategy {i+1}', linewidth=2)
        
        ax.set_title('Strategy Distribution Over Time (Mean Field)', fontsize=14)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Strategy Probability', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def visualize_nash_distance(self):
        """
        Visualize Nash distance over time
        
        Returns:
        - fig: Matplotlib figure
        """
        if not self.nash_distance_history:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.time_points, self.nash_distance_history, linewidth=2)
        
        ax.set_title('Nash Distance Over Time (Mean Field)', fontsize=14)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Nash Distance', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def compare_with_finite(self, finite_nash_distance, finite_time_points=None):
        """
        Compare mean field results with finite population simulation
        
        Parameters:
        - finite_nash_distance: Nash distance history from finite simulation
        - finite_time_points: Time points for finite simulation (optional)
        
        Returns:
        - fig: Matplotlib figure
        """
        if not self.nash_distance_history or not finite_nash_distance:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot mean field results
        ax.plot(self.time_points, self.nash_distance_history, 
                linewidth=2, label='Mean Field (Infinite)', color='blue')
        
        # Plot finite population results
        if finite_time_points is not None:
            ax.plot(finite_time_points, finite_nash_distance, 
                    linewidth=2, label='Finite Population', color='red', linestyle='--')
        else:
            # Assume same time scale
            ax.plot(self.time_points[:len(finite_nash_distance)], finite_nash_distance, 
                    linewidth=2, label='Finite Population', color='red', linestyle='--')
        
        ax.set_title('Nash Distance: Mean Field vs. Finite Population', fontsize=14)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Nash Distance', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
