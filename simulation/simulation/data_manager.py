#!/usr/bin/env python3
"""
Data Manager for the unified MMAI system

This module handles data storage, loading, and visualization for simulation results.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import pickle

class DataManager:
    """
    Manages data storage, loading, and visualization for simulation results
    
    The data manager handles:
    - Experiment configuration storage
    - Metrics recording and analysis
    - Checkpoint saving and loading
    - Visualization generation
    """
    def __init__(self, base_dir="./results", experiment_name=None, logger=None):
        """
        Initialize the data manager
        
        Parameters:
        - base_dir: Base directory for storing results
        - experiment_name: Name of the experiment (defaults to timestamp)
        """
        self.base_dir = base_dir
        self.logger = logger or logging.getLogger(__name__)
        
        # Create experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_name = f"experiment_{timestamp}"
        
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(base_dir, experiment_name)
        
        # Create directories
        self._create_directories()
        
        # Initialize metrics storage
        self.metrics = {}
        self.config = {}
        
        self.logger.info(f"Initialized data manager for experiment: {experiment_name}")
    
    def _create_directories(self):
        """Create necessary directories for the experiment"""
        # Main experiment directory
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Subdirectories
        os.makedirs(os.path.join(self.experiment_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "logs"), exist_ok=True)
        
        self.logger.debug(f"Created directories for experiment: {self.experiment_name}")
    
    def save_experiment_config(self, config):
        """
        Save experiment configuration
        
        Parameters:
        - config: Dictionary of configuration parameters
        """
        self.config = config
        
        # Save to file
        config_path = os.path.join(self.experiment_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Saved experiment configuration to {config_path}")
    
    def record_metrics(self, metrics, step):
        """
        Record metrics for a specific step
        
        Parameters:
        - metrics: Dictionary of metrics
        - step: Current step number
        """
        # Add step to metrics
        metrics['step'] = step
        
        # Store metrics
        if step not in self.metrics:
            self.metrics[step] = metrics
        else:
            self.metrics[step].update(metrics)
    
    def save_metrics(self, filename="metrics.json"):
        """
        Save all recorded metrics to file
        
        Parameters:
        - filename: Name of the metrics file
        """
        metrics_path = os.path.join(self.experiment_dir, "metrics", filename)
        
        # Convert metrics to list for easier processing
        metrics_list = [self.metrics[step] for step in sorted(self.metrics.keys())]
        
        # Convert NumPy types to Python native types
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(item) for item in obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            elif hasattr(obj, '__dict__'):
                # Handle custom objects by converting their __dict__
                return convert_numpy_types(obj.__dict__)
            else:
                # Try to convert to a basic type if possible
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, OverflowError):
                    return str(obj)
        
        # Convert metrics
        metrics_list = convert_numpy_types(metrics_list)
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_list, f, indent=2)
        
        self.logger.info(f"Saved metrics to {metrics_path}")
    
    def save_checkpoint(self, simulation, step):
        """
        Save a simulation checkpoint
        
        Parameters:
        - simulation: Simulation object
        - step: Current step number
        """
        checkpoint_path = os.path.join(self.experiment_dir, "checkpoints", f"checkpoint_{step}.pkl")
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(simulation, f)
            
            self.logger.info(f"Saved checkpoint at step {step} to {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, step=None):
        """
        Load a simulation checkpoint
        
        Parameters:
        - step: Step number to load (loads latest if None)
        
        Returns:
        - simulation: Loaded simulation object
        """
        checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        
        if step is None:
            # Find latest checkpoint
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")]
            if not checkpoints:
                self.logger.warning("No checkpoints found")
                return None
            
            # Extract step numbers and find latest
            steps = [int(f.split('_')[1].split('.')[0]) for f in checkpoints]
            step = max(steps)
        
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pkl")
        
        try:
            with open(checkpoint_path, 'rb') as f:
                simulation = pickle.load(f)
            
            self.logger.info(f"Loaded checkpoint from step {step}")
            return simulation
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def save_visualization(self, fig, filename):
        """
        Save a visualization figure
        
        Parameters:
        - fig: Matplotlib figure object
        - filename: Name of the visualization file
        """
        if fig is None:
            return
        
        viz_path = os.path.join(self.experiment_dir, "visualizations", filename)
        
        try:
            fig.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.debug(f"Saved visualization to {viz_path}")
        except Exception as e:
            self.logger.error(f"Failed to save visualization: {e}")
    
    def create_experiment_summary(self):
        """
        Create a summary of the experiment
        
        Returns:
        - summary: Dictionary containing experiment summary
        """
        # Calculate summary statistics
        summary = {
            'experiment_name': self.experiment_name,
            'config': self.config,
            'metrics_summary': {}
        }
        
        # Extract metrics for summary
        if self.metrics:
            steps = sorted(self.metrics.keys())
            last_step = steps[-1]
            
            # Get all metric names
            metric_names = set()
            for step in steps:
                metric_names.update(self.metrics[step].keys())
            
            # Calculate statistics for each metric
            for metric in metric_names:
                if metric == 'step':
                    continue
                
                values = [self.metrics[step].get(metric) for step in steps if metric in self.metrics[step]]
                values = [v for v in values if v is not None]
                
                if values:
                    # Convert NumPy types to Python native types
                    def convert_numpy_types(obj):
                        if isinstance(obj, np.integer):
                            return int(obj)
                        elif isinstance(obj, np.floating):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif isinstance(obj, np.bool_):
                            return bool(obj)
                        else:
                            return obj
                    
                    summary['metrics_summary'][metric] = {
                        'final': convert_numpy_types(values[-1]),
                        'mean': float(np.mean(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values))
                    }
        
        # Save summary to file
        summary_path = os.path.join(self.experiment_dir, "experiment_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Created experiment summary at {summary_path}")
        
        return summary
    
    def create_manuscript_figure(self, simulation):
        """
        Create a manuscript-quality figure from simulation results
        
        Parameters:
        - simulation: Simulation object with results
        
        Returns:
        - fig: Matplotlib figure object
        """
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data from metrics
        steps = sorted(self.metrics.keys())
        
        # Plot 1: Nash distance and population
        ax1 = axs[0, 0]
        
        # Nash distance
        nash_values = [self.metrics[step].get('nash_distance', None) for step in steps]
        nash_values = [v for v in nash_values if v is not None]
        if nash_values:
            ax1.plot(nash_values, 'r-', linewidth=2, label='Nash Distance')
        
        # Population
        pop_values = [self.metrics[step].get('population', None) for step in steps]
        pop_values = [v for v in pop_values if v is not None]
        if pop_values:
            ax1_twin = ax1.twinx()
            ax1_twin.plot(pop_values, 'b-', linewidth=2, label='Population')
            ax1_twin.set_ylabel('Population', color='b', fontsize=12)
        
        ax1.set_title('Nash Distance and Population', fontsize=14)
        ax1.set_xlabel('Time Step', fontsize=12)
        ax1.set_ylabel('Nash Distance', color='r', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        if 'ax1_twin' in locals():
            lines2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax1.legend(loc='upper right')
        
        # Plot 2: Strategic field coherence
        ax2 = axs[0, 1]
        
        coherence_values = [self.metrics[step].get('coherence', None) for step in steps]
        coherence_values = [v for v in coherence_values if v is not None]
        if coherence_values:
            ax2.plot(coherence_values, 'g-', linewidth=2)
        
        ax2.set_title('Strategic Field Coherence', fontsize=14)
        ax2.set_xlabel('Time Step', fontsize=12)
        ax2.set_ylabel('Coherence', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Resource dynamics
        ax3 = axs[1, 0]
        
        resource_values = [self.metrics[step].get('total_resources', None) for step in steps]
        resource_values = [v for v in resource_values if v is not None]
        if resource_values:
            ax3.plot(resource_values, 'g-', linewidth=2, label='Total Resources')
        
        # Consumption
        consumption_values = [self.metrics[step].get('consumption', None) for step in steps]
        consumption_values = [v for v in consumption_values if v is not None]
        if consumption_values:
            ax3_twin = ax3.twinx()
            ax3_twin.plot(consumption_values, 'm-', linewidth=2, label='Consumption')
            ax3_twin.set_ylabel('Consumption', color='m', fontsize=12)
        
        ax3.set_title('Resource Dynamics', fontsize=14)
        ax3.set_xlabel('Time Step', fontsize=12)
        ax3.set_ylabel('Total Resources', color='g', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax3.get_legend_handles_labels()
        if 'ax3_twin' in locals():
            lines2, labels2 = ax3_twin.get_legend_handles_labels()
            ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax3.legend(loc='upper right')
        
        # Plot 4: Hypersensitive points
        ax4 = axs[1, 1]
        
        hypersensitive_values = [self.metrics[step].get('hypersensitive_count', None) for step in steps]
        hypersensitive_values = [v for v in hypersensitive_values if v is not None]
        if hypersensitive_values:
            ax4.plot(hypersensitive_values, 'r-', linewidth=2)
        
        ax4.set_title('Hypersensitive Points', fontsize=14)
        ax4.set_xlabel('Time Step', fontsize=12)
        ax4.set_ylabel('Count', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # Add experiment details
        plt.figtext(0.5, 0.01, f"Experiment: {self.experiment_name}", ha='center', fontsize=12)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save the figure
        self.save_visualization(fig, "manuscript_figure.png")
        
        return fig
    
    def compare_experiments(self, experiment_dirs, metrics=None, title=None):
        """
        Create comparison visualizations across multiple experiments
        
        Parameters:
        - experiment_dirs: List of experiment directories to compare
        - metrics: List of metrics to compare (defaults to all common metrics)
        - title: Optional title for the comparison
        
        Returns:
        - figs: List of comparison figures
        """
        # Load summaries from each experiment
        summaries = []
        for exp_dir in experiment_dirs:
            summary_path = os.path.join(exp_dir, "experiment_summary.json")
            try:
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                summaries.append(summary)
            except Exception as e:
                self.logger.error(f"Failed to load summary from {exp_dir}: {e}")
        
        if not summaries:
            self.logger.warning("No experiment summaries found for comparison")
            return []
        
        # Find common metrics if not specified
        if metrics is None:
            metrics = set()
            for summary in summaries:
                if 'metrics_summary' in summary:
                    metrics.update(summary['metrics_summary'].keys())
            metrics = list(metrics)
        
        # Create comparison figures
        figs = []
        
        # Create bar chart for final values
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set up data for bar chart
        exp_names = [s['experiment_name'] for s in summaries]
        x = np.arange(len(exp_names))
        width = 0.8 / len(metrics)
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            values = []
            for summary in summaries:
                if ('metrics_summary' in summary and 
                    metric in summary['metrics_summary'] and
                    'final' in summary['metrics_summary'][metric]):
                    values.append(summary['metrics_summary'][metric]['final'])
                else:
                    values.append(0)
            
            ax.bar(x + i*width - 0.4 + width/2, values, width, label=metric)
        
        # Add labels and legend
        ax.set_xlabel('Experiment', fontsize=12)
        ax.set_ylabel('Final Value', fontsize=12)
        if title:
            ax.set_title(f"{title} - Comparison", fontsize=14)
        else:
            ax.set_title('Experiment Comparison', fontsize=14)
        
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        figs.append(fig)
        
        # Save the comparison figure
        comparison_dir = os.path.join(self.base_dir, "comparisons")
        os.makedirs(comparison_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        comparison_path = os.path.join(comparison_dir, f"comparison_{timestamp}.png")
        fig.savefig(comparison_path, dpi=300, bbox_inches='tight')
        
        self.logger.info(f"Saved experiment comparison to {comparison_path}")
        
        return figs
    def save_experiment_results(self, experiment_name, results, create_visualizations=True):
        """
        Save experiment results and create visualizations
        
        Parameters:
        - experiment_name: Name of the experiment
        - results: Dictionary containing experiment results
        - create_visualizations: Whether to create visualizations
        
        Returns:
        - experiment_dir: Path to the experiment directory
        """
        # Create experiment directory if it doesn't exist
        experiment_dir = os.path.join(self.base_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Save results to file
        results_path = os.path.join(experiment_dir, "results.json")
        
        # Convert NumPy types to Python native types
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Convert results
        results_json = convert_numpy_types(results)
        
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        self.logger.info(f"Saved experiment results to {results_path}")
        
        # Create visualizations if requested
        if create_visualizations:
            viz_dir = os.path.join(experiment_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            # Create basic visualizations
            self._create_basic_visualizations(results, viz_dir)
        
        return experiment_dir
    
    def _create_basic_visualizations(self, results, viz_dir):
        """
        Create basic visualizations from results
        
        Parameters:
        - results: Dictionary containing experiment results
        - viz_dir: Directory to save visualizations
        """
        # Extract metrics
        metrics = results.get('metrics', {})
        
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Nash distance
        ax1 = axs[0, 0]
        if 'nash_distance' in metrics:
            ax1.plot(metrics['nash_distance'], 'r-', linewidth=2)
            ax1.set_title('Nash Distance')
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Distance')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Coherence
        ax2 = axs[0, 1]
        if 'coherence' in metrics:
            ax2.plot(metrics['coherence'], 'b-', linewidth=2)
            ax2.set_title('Field Coherence')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Coherence')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Population
        ax3 = axs[1, 0]
        if 'population' in metrics:
            ax3.plot(metrics['population'], 'g-', linewidth=2)
            ax3.set_title('Population')
            ax3.set_xlabel('Time Step')
            ax3.set_ylabel('Agents')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Agent types
        ax4 = axs[1, 1]
        if 'llm_agent_count' in metrics and 'standard_agent_count' in metrics:
            ax4.stackplot(range(len(metrics['llm_agent_count'])),
                         [metrics['llm_agent_count'], metrics['standard_agent_count']],
                         labels=['LLM Agents', 'Standard Agents'],
                         colors=['#3498db', '#e74c3c'],
                         alpha=0.7)
            ax4.set_title('Agent Population by Type')
            ax4.set_xlabel('Time Step')
            ax4.set_ylabel('Number of Agents')
            ax4.legend(loc='upper left')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(os.path.join(viz_dir, 'summary.png'), dpi=300)
        plt.close(fig)
        
        # Create LLM performance visualization if available
        if 'llm_response_times' in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(metrics['llm_response_times'], 'b-', linewidth=2)
            ax.set_title('LLM Response Times')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Average Response Time (s)')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(os.path.join(viz_dir, 'llm_performance.png'), dpi=300)
            plt.close(fig)
