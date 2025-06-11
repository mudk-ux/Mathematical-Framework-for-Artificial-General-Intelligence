# MMAI Simulation Experiment Guide

This guide outlines key experiments for validating the theoretical concepts in "Steps Towards AGI" using the unified MMAI simulation framework.

## Core Validation Experiments

### 1. Nash Equilibrium Proportional to Growth Test

This experiment validates the mass-action interpretation of Nash equilibrium and the proportionality between equilibrium convergence and system growth.

```bash
python run_simulation.py --experiment nash_equilibrium --max-steps 2000
```

**What to look for:**
- Examine the `growth_proportional_equilibrium.png` visualization
- Compare Nash distance convergence rates across different growth rates
- Verify that equilibrium convergence time is inversely proportional to growth rate
- Check the proportionality values in the experiment summary

**Theoretical validation:**
This experiment tests the core theoretical claim that Nash equilibria emerge proportional to system growth rate (g). The theory predicts that as population grows at rate g, Nash distance should decrease proportionally.

### 2. Strategic Field Wave Propagation Test

This experiment demonstrates how strategic information propagates through space as wave-like patterns.

```bash
python run_simulation.py --experiment strategic_fields --grid-size 100 --n-agents 75
```

**What to look for:**
- Examine the strategic field visualizations showing wave patterns
- Compare coherence values across different diffusion rates
- Observe how strategic information propagates from agent positions
- Check the wave field visualizations showing propagation patterns

**Theoretical validation:**
This experiment tests the theoretical concept of strategic fields as wave-like patterns that propagate through space. The theory predicts that strategic information should diffuse through space while maintaining coherent patterns.

### 3. Fractal Time Architecture Test

This experiment validates the multi-scale temporal architecture described in the theory.

```bash
python run_simulation.py --dt 0.005 --t-scale 100 --T-scale 30 --max-steps 3000 --n-agents 50
```

**What to look for:**
- Examine the temporal resonance visualizations
- Check the correlation between metrics at different time scales
- Observe how patterns at dt scale aggregate into patterns at t scale
- Verify that T-scale patterns emerge from t-scale patterns

**Theoretical validation:**
This experiment tests the theoretical concept of fractal time architecture with three nested temporal scales (dt, t, T). The theory predicts that patterns at different scales should reinforce each other, creating temporal resonance.

### 4. Hypersensitive Points and Strategic Decision Test

This experiment examines how hypersensitive points affect strategic decision-making.

```bash
python run_simulation.py --n-strategies 5 --n-agents 100 --env-type CHAOTIC --max-steps 1500
```

**What to look for:**
- Track the number of hypersensitive points over time
- Examine agent decisions at hypersensitive points
- Compare strategy distributions in hypersensitive vs. normal regions
- Observe how environmental chaos affects hypersensitivity

**Theoretical validation:**
This experiment tests the theoretical concept of hypersensitive points where small differences in strategy weights lead to different decisions. The theory predicts that these points should emerge naturally in strategy space and affect decision-making.

### 5. Stigmergic Coordination Through IRN Test

This experiment demonstrates how the Information Retrieval Network (IRN) enables stigmergic coordination.

```bash
python run_simulation.py --n-agents 150 --env-type SHOCK --max-steps 2000 --grid-size 75
```

**What to look for:**
- Compare individual vs. collective memory activation
- Observe strategic adaptation after environmental shocks
- Examine frame system utilization patterns
- Track coordination metrics during shock recovery

**Theoretical validation:**
This experiment tests the theoretical concept of stigmergic coordination through the IRN. The theory predicts that agents should coordinate their responses to environmental changes through indirect communication via the IRN.

## Advanced Experiments

### 6. Multi-Strategy Equilibrium Test

This experiment examines how multiple strategies can coexist in equilibrium.

```bash
python run_simulation.py --n-strategies 8 --n-agents 200 --max-steps 2500
```

**What to look for:**
- Observe the distribution of strategies at equilibrium
- Check if multiple strategies maintain non-zero proportions
- Examine the payoff matrix to understand strategy relationships
- Track how strategy distributions evolve over time

### 7. Environmental Adaptation Test

This experiment tests how the system adapts to different environmental dynamics.

```bash
python run_simulation.py --env-type PERIODIC --n-agents 100 --max-steps 2000
python run_simulation.py --env-type CHAOTIC --n-agents 100 --max-steps 2000
python run_simulation.py --env-type SHOCK --n-agents 100 --max-steps 2000
```

**What to look for:**
- Compare Nash distance patterns across different environment types
- Observe how strategic fields adapt to environmental changes
- Examine how the IRN stores and retrieves environmental patterns
- Track adaptation speed after environmental changes

### 8. Resource Competition Test

This experiment examines how resource dynamics affect strategic behavior.

```bash
python run_simulation.py --n-agents 150 --enable-resources --max-steps 2000
```

**What to look for:**
- Observe how agents compete for resources
- Track resource consumption patterns
- Examine how resource scarcity affects strategic choices
- Observe the emergence of resource-based niches

### 9. Population Scaling Test

This experiment tests how the system scales with increasing population size.

```bash
python run_simulation.py --n-agents 50 --max-steps 1000
python run_simulation.py --n-agents 100 --max-steps 1000
python run_simulation.py --n-agents 200 --max-steps 1000
```

**What to look for:**
- Compare convergence rates across different population sizes
- Examine how strategic fields scale with population
- Track computational performance with increasing agents
- Observe emergent patterns at different population scales

### 10. Long-Term Evolution Test

This experiment examines the long-term evolutionary dynamics of the system.

```bash
python run_simulation.py --n-agents 100 --max-steps 5000 --enable-reproduction
```

**What to look for:**
- Track strategy evolution over long time periods
- Observe the formation and dissolution of strategic equilibria
- Examine how hypersensitive points evolve over time
- Track the growth and adaptation of the IRN memory structure

## Analyzing Results

For each experiment, analyze the results using these approaches:

1. **Examine the manuscript figures** in the results directory
2. **Compare the results** with theoretical predictions
3. **Look for emergent patterns** that weren't explicitly programmed
4. **Analyze the relationship** between micro-level agent behaviors and macro-level system properties
5. **Identify resonance patterns** across different temporal and spatial scales

The key visualizations to examine include:
- `manuscript_figure.png`: Combined visualization of key metrics
- `strategic_field.png`: Visualization of the strategic field
- `nash_distance.png`: Graph of Nash distance over time
- `growth_proportional_equilibrium.png`: Analysis of equilibrium proportionality
- `payoff_matrix.png`: Visualization of the game-theoretic payoff matrix

## Comparing Experiments

To compare results across experiments:

```bash
python run_simulation.py --experiment strategic_fields
python run_simulation.py --experiment nash_equilibrium
```

Then use the data manager's comparison functionality:

```python
from simulation.data_manager import DataManager

dm = DataManager()
experiment_dirs = [
    "./results/strategic_fields_experiment_20250408_123456",
    "./results/nash_equilibrium_experiment_20250408_123456"
]
dm.compare_experiments(experiment_dirs, metrics=['nash_distance', 'coherence'])
```

This will generate comparative visualizations that help identify patterns and relationships across different experimental conditions.
