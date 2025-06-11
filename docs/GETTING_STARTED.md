# Getting Started with MMAI-AGI Framework

This guide will help you understand and run the Mathematical Framework for Artificial General Intelligence.

## Quick Overview

Our framework demonstrates how artificial general intelligence can emerge from the integration of:

1. **Strategic Fields**: Wave-like patterns of strategic information
2. **Memory Systems**: Individual and collective memory through the IRN
3. **Nash Equilibria**: Strategic equilibria emerging from population dynamics

## Installation

### Prerequisites
- Python 3.8 or higher
- NumPy, SciPy, Matplotlib
- At least 4GB RAM for basic experiments
- 8GB+ RAM recommended for larger simulations

### Setup
```bash
git clone https://github.com/yourusername/MMAI-AGI-Framework.git
cd MMAI-AGI-Framework
pip install -r requirements.txt
```

## Running Your First Experiment

### 1. Basic Nash Equilibrium Experiment

```python
from simulation.run_simulation import run_experiment

# Run a basic Nash equilibrium experiment
results = run_experiment(
    experiment_type='nash_equilibrium',
    growth_rate=0.1,
    n_agents=50,
    max_steps=500
)

# View results
print(f"Final Nash Distance: {results['final_nash_distance']}")
print(f"Convergence Time: {results['convergence_time']} steps")
```

### 2. Strategic Fields Visualization

```python
# Run strategic fields experiment
results = run_experiment(
    experiment_type='strategic_fields',
    diffusion_rate=0.2,
    grid_size=50,
    n_agents=30,
    max_steps=300
)

# The experiment automatically generates visualizations
# Check the results directory for field evolution images
```

### 3. Complete Experimental Suite

```python
from simulation.run_all_experiments import run_all_experiments

# Run all five validation experiments
all_results = run_all_experiments()

# Results are saved to the results directory
# with comprehensive analysis and visualizations
```

## Understanding the Results

### Key Metrics

1. **Nash Distance**: Measures how close the system is to Nash equilibrium
   - Lower values = closer to equilibrium
   - Convergence pattern validates theoretical predictions

2. **Field Coherence**: Measures spatial organization of strategic fields
   - Higher values = more organized strategic patterns
   - Demonstrates wave-like propagation

3. **Temporal Resonance**: Correlation between different time scales
   - Higher correlations = stronger fractal time architecture
   - Validates multi-scale temporal framework

4. **Memory Activation**: Individual vs collective memory usage
   - High collective/individual ratio = effective stigmergic coordination
   - Demonstrates emergent collective intelligence

### Interpreting Visualizations

- **Nash Convergence Plots**: Show equilibrium formation over time
- **Strategic Field Maps**: Visualize wave-like information propagation
- **Correlation Matrices**: Display temporal resonance patterns
- **Memory Evolution**: Track individual vs collective memory growth

## Customizing Experiments

### Parameter Modification

```python
# Customize experiment parameters
custom_config = {
    'n_agents': 100,           # Number of agents
    'max_steps': 1000,         # Simulation duration
    'growth_rate': 0.15,       # Population growth rate
    'diffusion_rate': 0.3,     # Strategic field diffusion
    'grid_size': 75,           # Spatial grid size
    'n_strategies': 4,         # Number of available strategies
    'environment_type': 'CHAOTIC'  # Environment dynamics
}

results = run_experiment('nash_equilibrium', **custom_config)
```

### Creating New Experiments

```python
from simulation.core import Simulation, StrategicField, NashValidator

# Create custom simulation
strategic_field = StrategicField(grid_size=60, n_strategies=3)
nash_validator = NashValidator(n_strategies=3)

simulation = Simulation(
    strategic_field=strategic_field,
    nash_validator=nash_validator,
    n_agents=80,
    growth_rate=0.12
)

# Run custom simulation
for step in range(800):
    simulation.step()
    
    # Custom analysis
    if step % 100 == 0:
        nash_distance = simulation.nash_validator.get_nash_distance()
        field_coherence = simulation.strategic_field.get_coherence()
        print(f"Step {step}: Nash={nash_distance:.3f}, Coherence={field_coherence:.3f}")
```

## Advanced Usage

### Parallel Experiments

```python
from simulation.utils import run_parallel_experiments

# Run multiple experiments in parallel
parameter_sets = [
    {'growth_rate': 0.05, 'n_agents': 50},
    {'growth_rate': 0.1, 'n_agents': 75},
    {'growth_rate': 0.2, 'n_agents': 100}
]

results = run_parallel_experiments('nash_equilibrium', parameter_sets)
```

### Data Analysis

```python
from simulation.analysis import analyze_results, create_publication_figures

# Analyze experimental results
analysis = analyze_results(results)

# Generate publication-quality figures
create_publication_figures(analysis, output_dir='figures/')
```

### Custom Metrics

```python
from simulation.core import Simulation

class CustomSimulation(Simulation):
    def calculate_custom_metric(self):
        # Implement your custom analysis
        return custom_value
    
    def step(self):
        super().step()
        # Add custom logging or analysis
        if self.current_step % 50 == 0:
            custom_metric = self.calculate_custom_metric()
            self.log_metric('custom_metric', custom_metric)
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce grid_size or n_agents for large simulations
2. **Slow Performance**: Use smaller max_steps for initial testing
3. **Import Errors**: Ensure all dependencies are installed
4. **Visualization Issues**: Check matplotlib backend configuration

### Performance Optimization

```python
# For large-scale experiments
config = {
    'n_agents': 200,
    'grid_size': 100,
    'max_steps': 2000,
    'parallel_processing': True,  # Enable parallel processing
    'memory_efficient': True,     # Use memory-efficient algorithms
    'checkpoint_interval': 100    # Save checkpoints for long runs
}
```

### Getting Help

- Check the `docs/` directory for detailed documentation
- Review example scripts in `examples/`
- Examine test cases in `simulation/tests/`
- Open an issue on GitHub for specific problems

## Next Steps

1. **Run Basic Experiments**: Start with the five validation experiments
2. **Explore Parameters**: Modify parameters to understand their effects
3. **Analyze Results**: Use the analysis tools to understand the data
4. **Read Theory**: Study the theoretical framework in `theory/`
5. **Contribute**: Consider contributing improvements or extensions

## Understanding the Theory

To fully appreciate the experimental results, we recommend reading:

1. `theory/mathematical_foundations.md` - Core mathematical framework
2. `theory/strategic_fields.md` - Wave-theoretic approach
3. `theory/nash_equilibrium_extension.md` - M.M.A.I formulation
4. `experiments/README.md` - Detailed experimental validation

The framework represents a novel approach to AGI that focuses on emergence through strategic interaction rather than computational complexity. The experiments demonstrate how intelligence can arise naturally from the integration of simple components through well-defined mathematical principles.
