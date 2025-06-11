#!/usr/bin/env python3
"""
Fractal Time Manager for the unified MMAI system

This module implements the fractal time architecture described in
"Steps Towards AGI," with three nested temporal scales (dt, t, T)
that enable coherent pattern formation across multiple time scales.
"""

import numpy as np
from collections import defaultdict
import logging

class FractalTimeManager:
    """
    Manages nested temporal scales (dt, t, T) for the integrated system
    
    The fractal time architecture consists of three scales:
    - dt: Finest time scale for individual decisions
    - t: Intermediate scale for pattern formation
    - T: Largest scale for global behaviors
    """
    def __init__(self, dt=0.01, t_scale=50, T_scale=20, logger=None):
        """
        Initialize the fractal time manager
        
        Parameters:
        - dt: Finest time scale
        - t_scale: Number of dt steps in one t step
        - T_scale: Number of t steps in one T step
        """
        self.dt = dt  # Finest time scale
        self.t_scale = t_scale  # Number of dt steps in one t step
        self.T_scale = T_scale  # Number of t steps in one T step
        self.logger = logger or logging.getLogger(__name__)
        
        self.current_dt_step = 0
        self.current_t_step = 0
        self.current_T_step = 0
        
        self.dt_events = []
        self.t_events = []
        self.T_events = []
        
        self.dt_metrics = defaultdict(list)
        self.t_metrics = defaultdict(list)
        self.T_metrics = defaultdict(list)
        
        # For tracking temporal resonance
        self.resonance_patterns = []
        
        self.logger.info(f"Initialized fractal time manager with dt={dt}, t_scale={t_scale}, T_scale={T_scale}")
    
    def step(self):
        """
        Advance one dt step and update all time scales
        
        Returns:
        - event_type: String indicating the highest scale that was updated ('dt', 't', or 'T')
        """
        # Update dt
        self.current_dt_step += 1
        event_type = 'dt'
        
        # Check if we should update t
        if self.current_dt_step % self.t_scale == 0:
            self.current_t_step += 1
            event_type = 't'
            
            # Aggregate dt metrics to t scale
            for metric, values in self.dt_metrics.items():
                if values:
                    # Take average of recent dt values
                    recent_values = values[-self.t_scale:]
                    self.t_metrics[metric].append(np.mean(recent_values))
            
            # Record t event
            self.t_events.append({
                'dt': self.current_dt_step,
                't': self.current_t_step,
                'T': self.current_T_step
            })
            
            self.logger.debug(f"t-scale update: t={self.current_t_step}, dt={self.current_dt_step}")
        
        # Check if we should update T
        if self.current_t_step > 0 and self.current_t_step % self.T_scale == 0:
            self.current_T_step += 1
            event_type = 'T'
            
            # Aggregate t metrics to T scale
            for metric, values in self.t_metrics.items():
                if values:
                    # Take average of recent t values
                    recent_values = values[-self.T_scale:]
                    self.T_metrics[metric].append(np.mean(recent_values))
            
            # Record T event
            self.T_events.append({
                'dt': self.current_dt_step,
                't': self.current_t_step,
                'T': self.current_T_step
            })
            
            # Calculate temporal resonance
            self._calculate_resonance()
            
            self.logger.info(f"T-scale update: T={self.current_T_step}, t={self.current_t_step}, dt={self.current_dt_step}")
        
        # Record dt event
        self.dt_events.append({
            'dt': self.current_dt_step,
            't': self.current_t_step,
            'T': self.current_T_step
        })
        
        return event_type
    
    def _calculate_resonance(self):
        """
        Calculate temporal resonance patterns across scales
        
        Temporal resonance measures how patterns at different time scales
        align and reinforce each other.
        """
        resonance = {}
        
        # Find metrics that exist across all scales
        common_metrics = set(self.dt_metrics.keys()) & set(self.t_metrics.keys()) & set(self.T_metrics.keys())
        
        for metric in common_metrics:
            # Get the latest values at each scale
            dt_val = self.dt_metrics[metric][-1] if self.dt_metrics[metric] else 0
            t_val = self.t_metrics[metric][-1] if self.t_metrics[metric] else 0
            T_val = self.T_metrics[metric][-1] if self.T_metrics[metric] else 0
            
            # Calculate correlation between scales
            dt_t_corr = self._correlation(
                self.dt_metrics[metric][-self.t_scale:], 
                [t_val] * min(self.t_scale, len(self.dt_metrics[metric]))
            )
            
            t_T_corr = self._correlation(
                self.t_metrics[metric][-self.T_scale:], 
                [T_val] * min(self.T_scale, len(self.t_metrics[metric]))
            )
            
            # Calculate resonance as the product of correlations
            # High resonance means patterns are aligned across all scales
            scale_resonance = dt_t_corr * t_T_corr
            
            resonance[metric] = {
                'dt_value': dt_val,
                't_value': t_val,
                'T_value': T_val,
                'dt_t_correlation': dt_t_corr,
                't_T_correlation': t_T_corr,
                'resonance': scale_resonance
            }
        
        self.resonance_patterns.append({
            'dt': self.current_dt_step,
            't': self.current_t_step,
            'T': self.current_T_step,
            'resonance': resonance
        })
        
        # Log high resonance events
        for metric, data in resonance.items():
            if data['resonance'] > 0.8:
                self.logger.info(f"High resonance detected for {metric}: {data['resonance']:.2f}")
    
    def _correlation(self, x, y):
        """
        Calculate correlation between two arrays
        
        Parameters:
        - x: First array
        - y: Second array
        
        Returns:
        - correlation: Correlation coefficient
        """
        if len(x) != len(y) or len(x) < 2:
            return 0
        
        x = np.array(x)
        y = np.array(y)
        
        # If either array has no variance, correlation is undefined
        if np.std(x) == 0 or np.std(y) == 0:
            return 0
        
        return np.corrcoef(x, y)[0, 1]
    
    def record_metric(self, name, value):
        """
        Record a metric at the dt scale
        
        Parameters:
        - name: Metric name
        - value: Metric value
        """
        self.dt_metrics[name].append(value)
    
    def get_current_time(self):
        """
        Get the current time at all scales
        
        Returns:
        - time: Dictionary with current time at all scales
        """
        return {
            'dt': self.current_dt_step,
            't': self.current_t_step,
            'T': self.current_T_step,
            'real_time': self.current_dt_step * self.dt
        }
    
    def get_dt(self):
        """
        Get the current dt value
        
        Returns:
        - dt: Time step size
        """
        return self.dt
    
    def get_resonance_metrics(self):
        """
        Get resonance metrics for visualization
        
        Returns:
        - metrics: Dictionary of resonance metrics
        """
        if not self.resonance_patterns:
            return {}
        
        # Extract resonance values for each metric
        metrics = {}
        for pattern in self.resonance_patterns:
            for metric, data in pattern['resonance'].items():
                if metric not in metrics:
                    metrics[metric] = []
                metrics[metric].append(data['resonance'])
        
        return metrics
    
    def visualize_resonance(self, metric=None):
        """
        Visualize temporal resonance
        
        Parameters:
        - metric: Optional specific metric to visualize
        
        Returns:
        - fig: Matplotlib figure object
        """
        import matplotlib.pyplot as plt
        
        if not self.resonance_patterns:
            return None
        
        metrics = self.get_resonance_metrics()
        
        if metric is not None and metric in metrics:
            # Visualize specific metric
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(metrics[metric], linewidth=2)
            ax.set_title(f'Temporal Resonance - {metric}', fontsize=14)
            ax.set_xlabel('T Steps', fontsize=12)
            ax.set_ylabel('Resonance', fontsize=12)
            ax.grid(True, alpha=0.3)
        else:
            # Visualize all metrics
            fig, ax = plt.subplots(figsize=(12, 8))
            for name, values in metrics.items():
                ax.plot(values, label=name, linewidth=2, alpha=0.7)
            
            ax.set_title('Temporal Resonance Across Metrics', fontsize=14)
            ax.set_xlabel('T Steps', fontsize=12)
            ax.set_ylabel('Resonance', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        return fig
