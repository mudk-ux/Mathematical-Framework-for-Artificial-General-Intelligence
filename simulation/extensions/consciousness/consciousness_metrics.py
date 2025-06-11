#!/usr/bin/env python3
"""
Consciousness Metrics Implementation

This module implements metrics for measuring consciousness-like properties
in the MMAI system, including integration across temporal and spatial scales,
information-theoretic metrics, and visualization tools.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from collections import defaultdict

class ConsciousnessMetrics:
    """
    Implements metrics for measuring consciousness-like properties
    
    This class provides tools for:
    - Measuring integration across temporal and spatial scales
    """
    
    def __init__(self, max_history=10):
        """
        Initialize consciousness metrics
        
        Parameters:
        - max_history: Maximum history length
        """
        self.field_history = []
        self.max_history = max_history
        
    def calculate_metrics(self, field):
        """
        Calculate all consciousness metrics for a field
        
        Parameters:
        - field: Strategic field data
        
        Returns:
        - metrics: Dictionary of consciousness metrics
        """
        # Store field for later calculations
        self.field_history.append(field.copy())
        if len(self.field_history) > self.max_history:
            self.field_history.pop(0)
            
        # Calculate metrics
        metrics = {
            'temporal_integration': self.calculate_temporal_integration(),
            'spatial_integration': self.calculate_spatial_integration(),
            'phi': self.calculate_phi_measure(),
            'complexity': self.calculate_complexity()
        }
        
        return metrics
        
    def calculate_temporal_integration(self):
        """
        Calculate temporal integration
        
        Returns:
        - temporal_integration: Temporal integration value
        """
        if len(self.field_history) < 2:
            return 0.0
            
        # Calculate correlation between consecutive time steps
        correlation = 0.0
        count = 0
        
        for t in range(1, len(self.field_history)):
            prev = self.field_history[t-1].flatten()
            curr = self.field_history[t].flatten()
            
            # Calculate correlation
            if len(prev) == len(curr) and len(prev) > 0:
                corr = np.corrcoef(prev, curr)[0, 1]
                if not np.isnan(corr):
                    correlation += corr
                    count += 1
        
        # Normalize by number of correlations
        if count > 0:
            correlation /= count
            
        return float(correlation)
        
    def calculate_spatial_integration(self):
        """
        Calculate spatial integration
        
        Returns:
        - spatial_integration: Spatial integration value
        """
        if not self.field_history:
            return 0.0
            
        field = self.field_history[-1]
        
        # Check if field is empty
        if field.size == 0:
            return 0.0
            
        # Get dimensions
        if len(field.shape) == 3:
            grid_size_x, grid_size_y = field.shape[0], field.shape[1]
        else:
            grid_size_x, grid_size_y = field.shape
            
        # Calculate correlation between neighboring cells
        correlation = 0.0
        count = 0
        
        for i in range(grid_size_x):
            for j in range(grid_size_y):
                # Get neighbors (with periodic boundary)
                i_next = (i + 1) % grid_size_x
                j_next = (j + 1) % grid_size_y
                
                # Get cell values
                if len(field.shape) == 3:
                    cell = field[i, j, :]
                    right_neighbor = field[i_next, j, :]
                    bottom_neighbor = field[i, j_next, :]
                else:
                    cell = field[i, j]
                    right_neighbor = field[i_next, j]
                    bottom_neighbor = field[i, j_next]
                
                # Calculate correlations if possible
                if isinstance(cell, np.ndarray) and isinstance(right_neighbor, np.ndarray) and len(cell) > 1:
                    try:
                        right_corr = np.corrcoef(cell, right_neighbor)[0, 1]
                        if not np.isnan(right_corr):
                            correlation += right_corr
                            count += 1
                    except:
                        pass
                        
                if isinstance(cell, np.ndarray) and isinstance(bottom_neighbor, np.ndarray) and len(cell) > 1:
                    try:
                        bottom_corr = np.corrcoef(cell, bottom_neighbor)[0, 1]
                        if not np.isnan(bottom_corr):
                            correlation += bottom_corr
                            count += 1
                    except:
                        pass
        
        # Normalize by number of correlations
        if count > 0:
            correlation /= count
            
        return float(correlation)
        
    def calculate_phi_measure(self):
        """
        Calculate a simplified version of integrated information (Phi)
        
        Returns:
        - phi: Phi value
        """
        if not self.field_history:
            return 0.0
            
        field = self.field_history[-1]
        
        # Check if field is empty
        if field.size == 0:
            return 0.0
            
        # Get dimensions
        if len(field.shape) == 3:
            grid_size_x, grid_size_y = field.shape[0], field.shape[1]
        else:
            grid_size_x, grid_size_y = field.shape
            
        # Divide the grid into 4 quadrants
        half_x = grid_size_x // 2
        half_y = grid_size_y // 2
        
        # Extract quadrants
        if len(field.shape) == 3:
            q1 = field[:half_x, :half_y, :].flatten()
            q2 = field[:half_x, half_y:, :].flatten()
            q3 = field[half_x:, :half_y, :].flatten()
            q4 = field[half_x:, half_y:, :].flatten()
        else:
            q1 = field[:half_x, :half_y].flatten()
            q2 = field[:half_x, half_y:].flatten()
            q3 = field[half_x:, :half_y].flatten()
            q4 = field[half_x:, half_y:].flatten()
        
        # Calculate mutual information between quadrants
        mi = 0.0
        count = 0
        
        # Calculate mutual information between all pairs of quadrants
        quadrants = [q1, q2, q3, q4]
        for i in range(len(quadrants)):
            for j in range(i+1, len(quadrants)):
                if len(quadrants[i]) > 0 and len(quadrants[j]) > 0:
                    try:
                        # Use correlation as a proxy for mutual information
                        corr = np.corrcoef(quadrants[i], quadrants[j])[0, 1]
                        if not np.isnan(corr):
                            # Convert correlation to mutual information
                            # For Gaussian variables, MI = -0.5 * log(1 - corr^2)
                            mi_value = -0.5 * np.log(1 - corr**2 + 1e-10)
                            mi += mi_value
                            count += 1
                    except:
                        pass
        
        # Normalize by number of pairs
        if count > 0:
            mi /= count
            
        return float(mi)
        
    def calculate_complexity(self):
        """
        Calculate complexity as a balance between integration and differentiation
        
        Returns:
        - complexity: Complexity value
        """
        if not self.field_history:
            return 0.0
            
        # Calculate integration
        integration = self.calculate_spatial_integration()
        
        # Calculate differentiation (entropy)
        field = self.field_history[-1]
        if field.size == 0:
            return 0.0
            
        # Calculate entropy
        if len(field.shape) == 3:
            # For 3D field, calculate entropy of strategy distribution at each point
            entropy = 0.0
            count = 0
            
            for i in range(field.shape[0]):
                for j in range(field.shape[1]):
                    dist = field[i, j, :]
                    # Normalize if needed
                    if np.sum(dist) > 0:
                        dist = dist / np.sum(dist)
                        # Calculate entropy
                        ent = -np.sum(dist * np.log2(dist + 1e-10))
                        entropy += ent
                        count += 1
                        
            if count > 0:
                entropy /= count
        else:
            # For 2D field, calculate entropy of the entire field
            flat_field = field.flatten()
            if np.sum(flat_field) > 0:
                flat_field = flat_field / np.sum(flat_field)
                entropy = -np.sum(flat_field * np.log2(flat_field + 1e-10))
            else:
                entropy = 0.0
                
        # Complexity is the product of integration and differentiation
        complexity = integration * entropy
        
        return float(complexity)
