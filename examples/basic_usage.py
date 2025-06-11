#!/usr/bin/env python3
"""
Basic Usage Examples for MMAI-AGI Framework

This script demonstrates the basic usage of the Mathematical Framework 
for Artificial General Intelligence through simple examples.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'simulation'))

import numpy as np
import matplotlib.pyplot as plt
from core.strategic_field import StrategicField
from core.nash_validator import NashValidator
from core.memory_system import MemorySystem
from core.fractal_time_manager import FractalTimeManager
from core.agent import Agent
from core.environment_system import EnvironmentSystem

def example_1_basic_strategic_field():
    """
    Example 1: Basic Strategic Field Operations
    
    Demonstrates how strategic information propagates through space
    as wave-like patterns.
    """
    print("Example 1: Basic Strategic Field Operations")
    print("=" * 50)
    
    # Create a strategic field
    field = StrategicField(grid_size=50, n_strategies=3, diffusion_rate=0.2)
    
    # Add some initial strategic information
    field.add_strategy_influence(25, 25, strategy=0, strength=1.0)
    field.add_strategy_influence(10, 40, strategy=1, strength=0.8)
    field.add_strategy_influence(40, 10, strategy=2, strength=0.6)
    
    print(f"Initial field coherence: {field.get_coherence():.4f}")
    
    # Simulate field evolution
    coherence_history = []
    for step in range(100):
        field.update(dt=0.01)
        coherence = field.get_coherence()
        coherence_history.append(coherence)
        
        if step % 20 == 0:
            print(f"Step {step}: Coherence = {coherence:.4f}")
    
    print(f"Final field coherence: {coherence_history[-1]:.4f}")
    print(f"Coherence change: {coherence_history[-1] - coherence_history[0]:+.4f}")
    print()

def example_2_nash_equilibrium_tracking():
    """
    Example 2: Nash Equilibrium Tracking
    
    Demonstrates how Nash equilibria emerge and can be tracked
    over time in population dynamics.
    """
    print("Example 2: Nash Equilibrium Tracking")
    print("=" * 50)
    
    # Create Nash validator
    validator = NashValidator(n_strategies=3, growth_rate=0.1)
    
    # Simulate population evolution
    nash_distances = []
    population_states = []
    
    # Initial random population state
    current_state = np.random.random(3)
    current_state = current_state / np.sum(current_state)
    
    for step in range(200):
        # Simulate population dynamics (simplified)
        # In reality, this would be driven by agent interactions
        noise = np.random.normal(0, 0.01, 3)
        current_state += noise
        current_state = np.abs(current_state)
        current_state = current_state / np.sum(current_state)
        
        # Calculate Nash distance
        nash_distance = validator.calculate_nash_distance(current_state)
        nash_distances.append(nash_distance)
        population_states.append(current_state.copy())
        
        if step % 40 == 0:
            print(f"Step {step}: Nash Distance = {nash_distance:.4f}")
            print(f"  Population: [{current_state[0]:.3f}, {current_state[1]:.3f}, {current_state[2]:.3f}]")
    
    print(f"Final Nash distance: {nash_distances[-1]:.4f}")
    print(f"Convergence trend: {nash_distances[-1] - nash_distances[0]:+.4f}")
    print()

def example_3_memory_system():
    """
    Example 3: Information Retrieval Network (IRN)
    
    Demonstrates how individual and collective memory work together
    to enable stigmergic coordination.
    """
    print("Example 3: Information Retrieval Network (IRN)")
    print("=" * 50)
    
    # Create memory system
    memory = MemorySystem(memory_depth=50, collective_threshold=0.7)
    
    # Simulate agents adding experiences to memory
    n_agents = 10
    n_steps = 100
    
    individual_counts = []
    collective_counts = []
    
    for step in range(n_steps):
        # Each agent has a chance to add a memory frame
        for agent_id in range(n_agents):
            if np.random.random() < 0.3:  # 30% chance per step
                # Create a memory frame
                frame_data = {
                    'strategy': np.random.randint(0, 3),
                    'payoff': np.random.random(),
                    'context': np.random.random(5),
                    'importance': np.random.random()
                }
                memory.add_individual_frame(agent_id, frame_data)
        
        # Update collective memory
        memory.update_collective_memory()
        
        # Track memory growth
        individual_count = memory.get_individual_frame_count()
        collective_count = memory.get_collective_frame_count()
        individual_counts.append(individual_count)
        collective_counts.append(collective_count)
        
        if step % 20 == 0:
            print(f"Step {step}: Individual frames = {individual_count}, Collective frames = {collective_count}")
    
    print(f"Final individual frames: {individual_counts[-1]}")
    print(f"Final collective frames: {collective_counts[-1]}")
    print(f"Collective/Individual ratio: {collective_counts[-1] / max(individual_counts[-1], 1):.3f}")
    print()

def example_4_fractal_time():
    """
    Example 4: Fractal Time Architecture
    
    Demonstrates the multi-scale temporal framework with
    three nested time scales (dt, t, T).
    """
    print("Example 4: Fractal Time Architecture")
    print("=" * 50)
    
    # Create fractal time manager
    time_manager = FractalTimeManager(dt=0.01, t_scale=50, T_scale=20)
    
    # Track temporal evolution
    dt_events = []
    t_events = []
    T_events = []
    
    for step in range(1000):
        time_manager.update()
        
        # Record events at different scales
        dt_events.append(time_manager.dt_step)
        
        if time_manager.dt_step % time_manager.t_scale == 0:
            t_events.append(time_manager.t_step)
            print(f"t-scale event at step {step}: t_step = {time_manager.t_step}")
        
        if (time_manager.t_step % time_manager.T_scale == 0 and 
            time_manager.t_step > 0 and 
            len(T_events) == 0):  # Only record first T event for brevity
            T_events.append(time_manager.T_step)
            print(f"T-scale event at step {step}: T_step = {time_manager.T_step}")
    
    print(f"Final temporal state:")
    print(f"  dt_step: {time_manager.dt_step}")
    print(f"  t_step: {time_manager.t_step}")
    print(f"  T_step: {time_manager.T_step}")
    print(f"Total t-scale events: {len(t_events)}")
    print(f"Total T-scale events: {len(T_events)}")
    print()

def example_5_integrated_system():
    """
    Example 5: Integrated System
    
    Demonstrates how all components work together in a simple
    multi-agent system showing emergent intelligence.
    """
    print("Example 5: Integrated System")
    print("=" * 50)
    
    # Create system components
    strategic_field = StrategicField(grid_size=30, n_strategies=3, diffusion_rate=0.15)
    nash_validator = NashValidator(n_strategies=3, growth_rate=0.1)
    memory_system = MemorySystem(memory_depth=100)
    time_manager = FractalTimeManager(dt=0.01, t_scale=25, T_scale=10)
    environment = EnvironmentSystem(grid_size=30, env_type='STATIC')
    
    # Create agents
    n_agents = 20
    agents = []
    for i in range(n_agents):
        position = (np.random.randint(0, 30), np.random.randint(0, 30))
        agent = Agent(
            agent_id=i,
            position=position,
            strategy=np.random.randint(0, 3),
            memory_depth=10
        )
        agents.append(agent)
    
    # Run integrated simulation
    nash_history = []
    coherence_history = []
    memory_ratio_history = []
    
    for step in range(200):
        # Update time
        time_manager.update()
        
        # Update environment
        environment.update()
        
        # Agent decisions and actions
        strategy_counts = np.zeros(3)
        for agent in agents:
            # Agent perceives strategic field
            field_value = strategic_field.get_value_at_position(
                agent.position[0], agent.position[1]
            )
            
            # Agent makes decision
            decision = agent.decide(field_value, environment.get_local_state(agent.position))
            strategy_counts[decision] += 1
            
            # Agent influences strategic field
            strategic_field.add_strategy_influence(
                agent.position[0], agent.position[1],
                strategy=decision, strength=0.1
            )
            
            # Agent adds to memory
            memory_frame = {
                'strategy': decision,
                'field_value': field_value,
                'step': step,
                'importance': np.random.random()
            }
            memory_system.add_individual_frame(agent.agent_id, memory_frame)
        
        # Update systems
        strategic_field.update(dt=0.01)
        memory_system.update_collective_memory()
        
        # Calculate metrics
        population_state = strategy_counts / np.sum(strategy_counts)
        nash_distance = nash_validator.calculate_nash_distance(population_state)
        field_coherence = strategic_field.get_coherence()
        
        individual_frames = memory_system.get_individual_frame_count()
        collective_frames = memory_system.get_collective_frame_count()
        memory_ratio = collective_frames / max(individual_frames, 1)
        
        # Record metrics
        nash_history.append(nash_distance)
        coherence_history.append(field_coherence)
        memory_ratio_history.append(memory_ratio)
        
        if step % 40 == 0:
            print(f"Step {step}:")
            print(f"  Nash Distance: {nash_distance:.4f}")
            print(f"  Field Coherence: {field_coherence:.4f}")
            print(f"  Memory Ratio: {memory_ratio:.4f}")
            print(f"  Population: [{population_state[0]:.3f}, {population_state[1]:.3f}, {population_state[2]:.3f}]")
    
    print(f"\nFinal System State:")
    print(f"  Nash Distance: {nash_history[-1]:.4f} (change: {nash_history[-1] - nash_history[0]:+.4f})")
    print(f"  Field Coherence: {coherence_history[-1]:.4f} (change: {coherence_history[-1] - coherence_history[0]:+.4f})")
    print(f"  Memory Ratio: {memory_ratio_history[-1]:.4f}")
    print()

def main():
    """
    Run all examples to demonstrate the MMAI-AGI Framework.
    """
    print("MMAI-AGI Framework: Basic Usage Examples")
    print("=" * 60)
    print()
    
    # Run all examples
    example_1_basic_strategic_field()
    example_2_nash_equilibrium_tracking()
    example_3_memory_system()
    example_4_fractal_time()
    example_5_integrated_system()
    
    print("All examples completed successfully!")
    print("\nNext steps:")
    print("1. Run the full experimental suite: python simulation/run_all_experiments.py")
    print("2. Explore parameter variations in the examples above")
    print("3. Read the theory documentation in theory/")
    print("4. Check out the detailed experiments in experiments/")

if __name__ == "__main__":
    main()
