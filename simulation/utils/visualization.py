#!/usr/bin/env python3
"""
Visualization Utilities

This module provides visualization utilities for multi-agent systems.
"""

import numpy as np
import matplotlib.pyplot as plt

def create_figure(nrows=1, ncols=1, figsize=(10, 8)):
    """
    Create a figure with subplots
    
    Parameters:
    - nrows: Number of rows
    - ncols: Number of columns
    - figsize: Figure size
    
    Returns:
    - fig: Figure
    - axes: Axes
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    return fig, axes

def plot_field(ax, field, title=None):
    """
    Plot a strategic field
    
    Parameters:
    - ax: Matplotlib axis
    - field: Strategic field
    - title: Plot title
    """
    # Calculate dominant strategy at each position
    if len(field.shape) == 3:
        dominant = np.argmax(field, axis=2)
    else:
        dominant = field
    
    # Plot dominant strategy
    im = ax.imshow(dominant, cmap='viridis')
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Add title
    if title:
        ax.set_title(title)
    
    return im

def plot_time_series(ax, data, labels=None, title=None, xlabel=None, ylabel=None):
    """
    Plot time series data
    
    Parameters:
    - ax: Matplotlib axis
    - data: List of time series
    - labels: List of labels
    - title: Plot title
    - xlabel: X-axis label
    - ylabel: Y-axis label
    """
    # Plot data
    for i, series in enumerate(data):
        label = labels[i] if labels and i < len(labels) else f"Series {i+1}"
        ax.plot(series, label=label)
    
    # Add legend
    if labels:
        ax.legend()
    
    # Add title and labels
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    
    # Add grid
    ax.grid(True, alpha=0.3)
