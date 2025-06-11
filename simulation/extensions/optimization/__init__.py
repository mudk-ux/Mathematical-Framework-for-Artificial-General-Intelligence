"""
Optimization Extension

This extension implements optimization techniques for the MMAI system,
including spatial partitioning, sparse matrix representation, and
parallel processing.
"""

from .spatial_partitioning import SpatialPartitioning
from .optimized_strategic_field import OptimizedStrategicField

__all__ = ['SpatialPartitioning', 'OptimizedStrategicField']
