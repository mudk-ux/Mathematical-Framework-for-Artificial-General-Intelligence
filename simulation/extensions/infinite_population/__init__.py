"""
Infinite Population Approximation Extension

This extension implements mean-field approximation techniques to model infinite
population dynamics, bridging the gap between finite agent simulations and
theoretical infinite populations.
"""

from .mean_field import MeanFieldApproximation
from .scaling_analysis import ScalingAnalysis

__all__ = ['MeanFieldApproximation', 'ScalingAnalysis']
