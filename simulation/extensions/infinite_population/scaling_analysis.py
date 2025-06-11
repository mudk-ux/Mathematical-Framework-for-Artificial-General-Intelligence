#!/usr/bin/env python3
"""
Scaling Analysis for Infinite Population Approximation

This module implements tools for comparing finite and infinite population
dynamics, analyzing how results scale with population size, and extrapolating
to the infinite limit.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from ..infinite_population.mean_field import MeanFieldApproximation

class ScalingAnalysis:
    """
    Implements scaling analysis for comparing finite and infinite populations
    
    This class provides tools for:
    - Running simulations with different population sizes
    - Comparing results between finite and infinite approximations
    - Extrapolating to the infinite limit
    """
    def __init__(self, n_strategies=3, growth_rate=0.05, logger=None):
        """
        Initialize the scaling analysis
        
        Parameters:
        - n_strategies: Number of strategies
        - growth_rate: System growth rate (g)
        """
        self.n_strategies = n_strategies
        self.growth_rate = growth_rate
        self.logger = logger or logging.getLogger(__name__)
        
        # For storing results
        self.population_sizes = []
        self.finite_results = {}
        self.infinite_results = {}
        self.extrapolation_results = {}
        
        self.logger.info(f"Initialized scaling analysis with {n_strategies} strategies")
    
    def run_scaling_analysis(self, simulation_class, payoff_matrix, 
                            min_agents=10, max_agents=1000, steps=5, 
                            max_steps=1000, dt=0.1):
        """
        Run scaling analysis to compare finite and infinite approximations
        
        Parameters:
        - simulation_class: Class for running finite simulations
        - payoff_matrix: Game payoff matrix
        - min_agents: Minimum number of agents
        - max_agents: Maximum number of agents
        - steps: Number of population sizes to test
        - max_steps: Maximum simulation steps
        - dt: Time step size
        
        Returns:
        - results: Dictionary of scaling analysis results
        """
        # Generate population sizes on logarithmic scale
        population_sizes = np.logspace(
            np.log10(min_agents), 
            np.log10(max_agents), 
            steps
        ).astype(int)
        
        self.population_sizes = population_sizes
        self.logger.info(f"Running scaling analysis with population sizes: {population_sizes}")
        
        # Run infinite population simulation
        mean_field = MeanFieldApproximation(
            n_strategies=self.n_strategies,
            growth_rate=self.growth_rate
        )
        
        infinite_results = {
            'nash_distance': [],
            'convergence_time': None,
            'equilibrium_threshold': 0.1
        }
        
        # Run infinite simulation
        for step in range(max_steps):
            nash_distance = mean_field.update(payoff_matrix, dt)
            infinite_results['nash_distance'].append(nash_distance)
            
            # Check for convergence
            if nash_distance < 0.1 and infinite_results['convergence_time'] is None:
                infinite_results['convergence_time'] = step
        
        if infinite_results['convergence_time'] is None:
            infinite_results['convergence_time'] = max_steps
        
        self.infinite_results = infinite_results
        
        # Run finite population simulations for each size
        for size in population_sizes:
            self.logger.info(f"Running simulation with {size} agents")
            
            # Initialize simulation
            simulation = simulation_class(
                n_agents=size,
                n_strategies=self.n_strategies,
                growth_rate=self.growth_rate
            )
            
            # Run simulation
            results = simulation.run(max_steps=max_steps)
            
            # Store results
            self.finite_results[size] = {
                'nash_distance': results['nash_distance_history'],
                'convergence_time': results.get('convergence_time', max_steps),
                'final_nash_distance': results['nash_distance_history'][-1]
            }
        
        # Perform extrapolation analysis
        self._extrapolate_to_infinity()
        
        return {
            'population_sizes': population_sizes,
            'finite_results': self.finite_results,
            'infinite_results': self.infinite_results,
            'extrapolation': self.extrapolation_results
        }
    
    def _extrapolate_to_infinity(self):
        """
        Extrapolate results to the infinite population limit
        
        Uses curve fitting to estimate behavior as population size approaches infinity
        """
        if len(self.population_sizes) == 0 or not self.finite_results:
            return
        
        # Extract convergence times and final Nash distances
        sizes = np.array(self.population_sizes)
        convergence_times = np.array([
            self.finite_results[size]['convergence_time'] 
            for size in sizes
        ])
        final_distances = np.array([
            self.finite_results[size]['final_nash_distance'] 
            for size in sizes
        ])
        
        # Fit power law: y = a * x^b
        # Using log-log linear regression: log(y) = log(a) + b*log(x)
        log_sizes = np.log(sizes)
        
        # For convergence time
        log_times = np.log(convergence_times)
        time_coeffs = np.polyfit(log_sizes, log_times, 1)
        time_a = np.exp(time_coeffs[1])
        time_b = time_coeffs[0]
        
        # For final Nash distance
        log_distances = np.log(final_distances + 1e-10)  # Add small value to avoid log(0)
        dist_coeffs = np.polyfit(log_sizes, log_distances, 1)
        dist_a = np.exp(dist_coeffs[1])
        dist_b = dist_coeffs[0]
        
        # Extrapolate to larger population sizes
        extrapolation_sizes = np.logspace(
            np.log10(self.population_sizes[-1]),
            np.log10(self.population_sizes[-1] * 100),
            10
        ).astype(int)
        
        extrapolated_times = time_a * extrapolation_sizes ** time_b
        extrapolated_distances = dist_a * extrapolation_sizes ** dist_b
        
        # Store extrapolation results
        self.extrapolation_results = {
            'sizes': extrapolation_sizes,
            'convergence_times': extrapolated_times,
            'nash_distances': extrapolated_distances,
            'time_model': {'a': time_a, 'b': time_b},
            'distance_model': {'a': dist_a, 'b': dist_b}
        }
        
        self.logger.info(f"Extrapolation models - Time: {time_a:.4f}*N^{time_b:.4f}, Distance: {dist_a:.4f}*N^{dist_b:.4f}")
    
    def visualize_scaling(self):
        """
        Visualize scaling analysis results
        
        Returns:
        - fig: Matplotlib figure
        """
        if len(self.population_sizes) == 0 or not self.finite_results:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Extract data
        sizes = np.array(self.population_sizes)
        convergence_times = np.array([
            self.finite_results[size]['convergence_time'] 
            for size in sizes
        ])
        final_distances = np.array([
            self.finite_results[size]['final_nash_distance'] 
            for size in sizes
        ])
        
        # Plot convergence time vs population size
        ax1.loglog(sizes, convergence_times, 'o-', linewidth=2, label='Finite Simulations')
        ax1.set_title('Convergence Time vs Population Size', fontsize=14)
        ax1.set_xlabel('Population Size (log scale)', fontsize=12)
        ax1.set_ylabel('Convergence Time (log scale)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Add infinite result
        if self.infinite_results:
            ax1.axhline(
                y=self.infinite_results['convergence_time'],
                color='r', linestyle='--', 
                label='Infinite Approximation'
            )
        
        # Add extrapolation
        if self.extrapolation_results:
            extra_sizes = self.extrapolation_results['sizes']
            extra_times = self.extrapolation_results['convergence_times']
            ax1.loglog(
                extra_sizes, extra_times, 
                'k--', linewidth=1.5, 
                label='Extrapolation'
            )
            
            # Add model equation
            model = self.extrapolation_results['time_model']
            ax1.text(
                0.05, 0.95, 
                f"Model: {model['a']:.2f} × N^{model['b']:.2f}",
                transform=ax1.transAxes,
                verticalalignment='top',
                fontsize=10
            )
        
        ax1.legend()
        
        # Plot final Nash distance vs population size
        ax2.loglog(sizes, final_distances, 'o-', linewidth=2, label='Finite Simulations')
        ax2.set_title('Final Nash Distance vs Population Size', fontsize=14)
        ax2.set_xlabel('Population Size (log scale)', fontsize=12)
        ax2.set_ylabel('Nash Distance (log scale)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add infinite result
        if self.infinite_results:
            ax2.axhline(
                y=self.infinite_results['nash_distance'][-1],
                color='r', linestyle='--', 
                label='Infinite Approximation'
            )
        
        # Add extrapolation
        if self.extrapolation_results:
            extra_sizes = self.extrapolation_results['sizes']
            extra_distances = self.extrapolation_results['nash_distances']
            ax2.loglog(
                extra_sizes, extra_distances, 
                'k--', linewidth=1.5, 
                label='Extrapolation'
            )
            
            # Add model equation
            model = self.extrapolation_results['distance_model']
            ax2.text(
                0.05, 0.95, 
                f"Model: {model['a']:.2f} × N^{model['b']:.2f}",
                transform=ax2.transAxes,
                verticalalignment='top',
                fontsize=10
            )
        
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def visualize_nash_distance_comparison(self):
        """
        Visualize Nash distance comparison between finite and infinite simulations
        
        Returns:
        - fig: Matplotlib figure
        """
        if len(self.finite_results) == 0 or not self.infinite_results:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot infinite result
        ax.plot(
            range(len(self.infinite_results['nash_distance'])),
            self.infinite_results['nash_distance'],
            'r-', linewidth=2, label='Infinite Approximation'
        )
        
        # Plot finite results for each population size
        for size in self.population_sizes:
            nash_distance = self.finite_results[size]['nash_distance']
            ax.plot(
                range(len(nash_distance)),
                nash_distance,
                '--', linewidth=1.5, alpha=0.7,
                label=f'N = {size}'
            )
        
        ax.set_title('Nash Distance Comparison', fontsize=14)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Nash Distance', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig
