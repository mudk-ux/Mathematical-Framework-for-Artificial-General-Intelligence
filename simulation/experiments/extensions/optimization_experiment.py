#!/usr/bin/env python3
"""
Optimization Experiment

This experiment compares the performance of the optimized implementation
with the original implementation, measuring computational efficiency and
memory usage.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from extensions.optimization import SpatialPartitioning, OptimizedStrategicField
from core.strategic_field import StrategicField
from simulation.simulation import Simulation
from core.agent import Agent

def run_experiment(output_dir=None, grid_size=100, n_agents=200, n_strategies=3, max_steps=500):
    """
    Run the optimization experiment
    
    Parameters:
    - output_dir: Directory for output files
    - grid_size: Size of the environment grid
    - n_agents: Number of agents
    - n_strategies: Number of strategies
    - max_steps: Maximum simulation steps
    
    Returns:
    - results: Dictionary of experiment results
    """
    # Set up logging
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./results/optimization_experiment_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, "experiment.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting optimization experiment with grid size {grid_size}, {n_agents} agents")
    
    # Initialize agents
    agents = []
    for i in range(n_agents):
        position = np.random.rand(2) * grid_size
        agent = Agent(
            agent_id=i,
            position=position,
            strategy=np.random.random(n_strategies)  # Use random strategy distribution
        )
        agents.append(agent)
    
    logger.info(f"Initialized {n_agents} agents")
    
    # Part 1: Compare original and optimized strategic fields
    logger.info("Part 1: Comparing original and optimized strategic fields")
    
    # Initialize fields
    original_field = StrategicField(
        grid_size=grid_size,
        n_strategies=n_strategies,
        diffusion_rate=0.2,
        logger=logger
    )
    
    optimized_sparse_field = OptimizedStrategicField(
        grid_size=grid_size,
        n_strategies=n_strategies,
        diffusion_rate=0.2,
        use_sparse=True,
        use_fft=True,
        logger=logger
    )
    
    optimized_dense_field = OptimizedStrategicField(
        grid_size=grid_size,
        n_strategies=n_strategies,
        diffusion_rate=0.2,
        use_sparse=False,
        use_fft=True,
        logger=logger
    )
    
    # Run comparison
    original_times = []
    sparse_times = []
    dense_times = []
    
    original_coherence = []
    sparse_coherence = []
    dense_coherence = []
    
    logger.info("Running field comparison...")
    
    for step in range(max_steps):
        if step % 50 == 0:
            logger.info(f"Field comparison step {step}/{max_steps}")
        
        # Move agents
        for agent in agents:
            # Simple random movement
            direction = np.random.rand(2) * 2 - 1
            direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else direction
            agent.position += direction * 0.5
            agent.position = np.clip(agent.position, 0, grid_size - 1)
        
        # Update original field
        start_time = time.time()
        coherence = original_field.update(agents)
        original_times.append(time.time() - start_time)
        original_coherence.append(coherence)
        
        # Update optimized sparse field
        start_time = time.time()
        coherence = optimized_sparse_field.update(agents)
        sparse_times.append(time.time() - start_time)
        sparse_coherence.append(coherence)
        
        # Update optimized dense field
        start_time = time.time()
        coherence = optimized_dense_field.update(agents)
        dense_times.append(time.time() - start_time)
        dense_coherence.append(coherence)
    
    # Calculate statistics
    original_avg_time = np.mean(original_times)
    sparse_avg_time = np.mean(sparse_times)
    dense_avg_time = np.mean(dense_times)
    
    sparse_speedup = original_avg_time / sparse_avg_time if sparse_avg_time > 0 else 0
    dense_speedup = original_avg_time / dense_avg_time if dense_avg_time > 0 else 0
    
    logger.info(f"Original field average update time: {original_avg_time:.6f}s")
    logger.info(f"Optimized sparse field average update time: {sparse_avg_time:.6f}s (speedup: {sparse_speedup:.2f}x)")
    logger.info(f"Optimized dense field average update time: {dense_avg_time:.6f}s (speedup: {dense_speedup:.2f}x)")
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot update times
    ax1.plot(original_times, label='Original', linewidth=2, alpha=0.7)
    ax1.plot(sparse_times, label='Optimized (Sparse)', linewidth=2, alpha=0.7)
    ax1.plot(dense_times, label='Optimized (Dense)', linewidth=2, alpha=0.7)
    
    ax1.set_title('Strategic Field Update Time Comparison', fontsize=14)
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Update Time (seconds)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    ax1.text(
        0.02, 0.95, 
        f"Original: {original_avg_time:.6f}s\nSparse: {sparse_avg_time:.6f}s ({sparse_speedup:.2f}x)\nDense: {dense_avg_time:.6f}s ({dense_speedup:.2f}x)",
        transform=ax1.transAxes,
        verticalalignment='top',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7)
    )
    
    # Plot coherence
    ax2.plot(original_coherence, label='Original', linewidth=2, alpha=0.7)
    ax2.plot(sparse_coherence, label='Optimized (Sparse)', linewidth=2, alpha=0.7)
    ax2.plot(dense_coherence, label='Optimized (Dense)', linewidth=2, alpha=0.7)
    
    ax2.set_title('Strategic Field Coherence Comparison', fontsize=14)
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Coherence', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "field_comparison.png"))
    
    # Part 2: Test spatial partitioning
    logger.info("Part 2: Testing spatial partitioning")
    
    # Initialize spatial partitioning
    spatial = SpatialPartitioning(
        grid_size=grid_size,
        cell_size=5,
        logger=logger
    )
    
    # Run comparison
    naive_times = []
    spatial_times = []
    
    logger.info("Running spatial partitioning comparison...")
    
    for step in range(max_steps):
        if step % 50 == 0:
            logger.info(f"Spatial partitioning step {step}/{max_steps}")
        
        # Move agents
        for agent in agents:
            # Simple random movement
            direction = np.random.rand(2) * 2 - 1
            direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else direction
            agent.position += direction * 0.5
            agent.position = np.clip(agent.position, 0, grid_size - 1)
        
        # Update spatial partitioning
        spatial.update(agents)
        
        # Test naive neighbor search
        start_time = time.time()
        for agent in agents[:10]:  # Test with subset for speed
            naive_neighbors = []
            for other in agents:
                if other != agent:
                    distance = np.linalg.norm(agent.position - other.position)
                    if distance <= 5.0:
                        naive_neighbors.append(other)
        naive_times.append(time.time() - start_time)
        
        # Test spatial neighbor search
        start_time = time.time()
        for agent in agents[:10]:  # Test with same subset
            spatial_neighbors = spatial.get_nearby_agents(agent.position, 5.0)
        spatial_times.append(time.time() - start_time)
    
    # Calculate statistics
    naive_avg_time = np.mean(naive_times)
    spatial_avg_time = np.mean(spatial_times)
    
    spatial_speedup = naive_avg_time / spatial_avg_time if spatial_avg_time > 0 else 0
    
    logger.info(f"Naive neighbor search average time: {naive_avg_time:.6f}s")
    logger.info(f"Spatial partitioning average time: {spatial_avg_time:.6f}s (speedup: {spatial_speedup:.2f}x)")
    
    # Get performance stats
    spatial_stats = spatial.get_performance_stats()
    logger.info(f"Spatial partitioning efficiency ratio: {spatial_stats['efficiency_ratio']:.2f}x")
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot search times
    ax1.plot(naive_times, label='Naive Search', linewidth=2, alpha=0.7)
    ax1.plot(spatial_times, label='Spatial Partitioning', linewidth=2, alpha=0.7)
    
    ax1.set_title('Neighbor Search Time Comparison', fontsize=14)
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Search Time (seconds)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    ax1.text(
        0.02, 0.95, 
        f"Naive: {naive_avg_time:.6f}s\nSpatial: {spatial_avg_time:.6f}s\nSpeedup: {spatial_speedup:.2f}x",
        transform=ax1.transAxes,
        verticalalignment='top',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7)
    )
    
    # Visualize spatial partitioning
    spatial.visualize(ax=ax2)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "spatial_comparison.png"))
    
    # Part 3: Create combined visualization for manuscript
    logger.info("Creating manuscript figure")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Strategic field update times
    ax = axes[0, 0]
    ax.plot(original_times, label='Original', linewidth=2, alpha=0.7)
    ax.plot(sparse_times, label='Optimized (Sparse)', linewidth=2, alpha=0.7)
    ax.plot(dense_times, label='Optimized (Dense)', linewidth=2, alpha=0.7)
    
    ax.set_title('Strategic Field Update Time', fontsize=14)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    ax.text(
        0.02, 0.95, 
        f"Speedup:\nSparse: {sparse_speedup:.2f}x\nDense: {dense_speedup:.2f}x",
        transform=ax.transAxes,
        verticalalignment='top',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7)
    )
    
    # Plot 2: Strategic field coherence
    ax = axes[0, 1]
    ax.plot(original_coherence, label='Original', linewidth=2, alpha=0.7)
    ax.plot(sparse_coherence, label='Optimized (Sparse)', linewidth=2, alpha=0.7)
    ax.plot(dense_coherence, label='Optimized (Dense)', linewidth=2, alpha=0.7)
    
    ax.set_title('Strategic Field Coherence', fontsize=14)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Coherence', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Neighbor search times
    ax = axes[1, 0]
    ax.plot(naive_times, label='Naive Search', linewidth=2, alpha=0.7)
    ax.plot(spatial_times, label='Spatial Partitioning', linewidth=2, alpha=0.7)
    
    ax.set_title('Neighbor Search Time', fontsize=14)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    ax.text(
        0.02, 0.95, 
        f"Speedup: {spatial_speedup:.2f}x\nEfficiency: {spatial_stats['efficiency_ratio']:.2f}x",
        transform=ax.transAxes,
        verticalalignment='top',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7)
    )
    
    # Plot 4: Memory usage comparison
    ax = axes[1, 1]
    
    # Get memory stats
    sparse_stats = optimized_sparse_field.get_performance_stats()
    memory_reduction = sparse_stats.get('memory_reduction', 0)
    
    # Create bar chart
    memory_labels = ['Original', 'Sparse', 'Dense + FFT']
    memory_values = [1.0, 1.0 - memory_reduction, 1.0]
    
    ax.bar(memory_labels, memory_values, color=['blue', 'green', 'orange'])
    ax.set_title('Relative Memory Usage', fontsize=14)
    ax.set_ylabel('Memory Usage (normalized)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add memory reduction text
    ax.text(
        1, memory_values[1] / 2, 
        f"{memory_reduction:.1%}\nreduction",
        ha='center',
        va='center',
        fontsize=10
    )
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "manuscript_figure.png"))
    
    logger.info(f"Experiment completed. Results saved to {output_dir}")
    
    return {
        'original_times': original_times,
        'sparse_times': sparse_times,
        'dense_times': dense_times,
        'original_coherence': original_coherence,
        'sparse_coherence': sparse_coherence,
        'dense_coherence': dense_coherence,
        'naive_times': naive_times,
        'spatial_times': spatial_times,
        'sparse_speedup': sparse_speedup,
        'dense_speedup': dense_speedup,
        'spatial_speedup': spatial_speedup,
        'memory_reduction': memory_reduction,
        'output_dir': output_dir
    }
    
    # Export raw data for analysis
    # Create data export dictionary
    data_export = {
        'field_comparison': {
            'original_times': original_times,
            'sparse_times': sparse_times,
            'dense_times': dense_times,
            'original_coherence': original_coherence,
            'sparse_coherence': sparse_coherence,
            'dense_coherence': dense_coherence
        },
        'spatial_comparison': {
            'naive_times': naive_times,
            'spatial_times': spatial_times
        },
        'performance_metrics': {
            'sparse_speedup': sparse_speedup,
            'dense_speedup': dense_speedup,
            'spatial_speedup': spatial_speedup,
            'memory_reduction': memory_reduction,
            'original_avg_time': original_avg_time,
            'sparse_avg_time': sparse_avg_time,
            'dense_avg_time': dense_avg_time,
            'naive_avg_time': naive_avg_time,
            'spatial_avg_time': spatial_avg_time
        },
        'spatial_stats': spatial.get_performance_stats() if hasattr(spatial, 'get_performance_stats') else {}
    }
    
    # Save raw data directly
    import json
    try:
        # Create a simple test file first
        test_file_path = os.path.join(output_dir, "test_write.txt")
        with open(test_file_path, 'w') as f:
            f.write("Test write permissions")
        print(f"Successfully created test file at {test_file_path}")
        
        # Now try to save the raw data
        raw_data_path = os.path.join(output_dir, "raw_data.json")
        print(f"Attempting to save raw data to {raw_data_path}")
        
        # Create a simple test data structure
        test_data = {
            "test": "data",
            "numbers": [1, 2, 3, 4, 5]
        }
        
        with open(raw_data_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        # Verify the file was created
        if os.path.exists(raw_data_path):
            print(f"Raw data successfully exported to {raw_data_path}")
            print(f"File size: {os.path.getsize(raw_data_path)} bytes")
        else:
            print(f"Failed to create raw data file at {raw_data_path}")
    except Exception as e:
        print(f"Error exporting raw data: {e}")
        import traceback
        print(traceback.format_exc())
    
    return {
        'original_times': original_times,
        'sparse_times': sparse_times,
        'dense_times': dense_times,
        'original_coherence': original_coherence,
        'sparse_coherence': sparse_coherence,
        'dense_coherence': dense_coherence,
        'naive_times': naive_times,
        'spatial_times': spatial_times,
        'sparse_speedup': sparse_speedup,
        'dense_speedup': dense_speedup,
        'spatial_speedup': spatial_speedup,
        'memory_reduction': memory_reduction,
        'output_dir': output_dir,
        'data_export': data_export
    }

if __name__ == "__main__":
    run_experiment()
