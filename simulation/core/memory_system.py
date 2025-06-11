#!/usr/bin/env python3
"""
Memory System (IRN) for the unified MMAI system

This module implements the Information Retrieval Network (IRN) as described in
"Steps Towards AGI," providing both individual and collective memory capabilities
through frame systems.
"""

import numpy as np
from collections import defaultdict
import logging

class Frame:
    """
    Implementation of a frame as described in Minsky's frame system theory
    
    A frame represents a stereotyped situation with:
    - Fixed conditions (always true about the situation)
    - Slots (variables that can be filled)
    - Default assignments (initial values for slots)
    - Marker conditions (conditions for slot filling)
    - Assignment conditions (constraints on slot values)
    """
    def __init__(self, name, fixed_conditions=None, slots=None, default_assignments=None):
        self.name = name
        self.fixed_conditions = fixed_conditions or {}
        self.slots = slots or {}
        self.default_assignments = default_assignments or {}
        self.marker_conditions = {}
        self.assignment_conditions = {}
        self.activation_level = 0.0
        self.last_access_time = 0
    
    def update_activation(self, activation_delta, current_time):
        """
        Update the activation level of this frame
        
        Parameters:
        - activation_delta: Change in activation
        - current_time: Current simulation time
        """
        # Decay based on time since last access
        time_delta = current_time - self.last_access_time
        if time_delta > 0:
            self.activation_level *= np.exp(-0.01 * time_delta)  # Exponential decay
        
        # Add new activation
        self.activation_level += activation_delta
        
        # Cap activation between 0 and 1
        self.activation_level = max(0.0, min(1.0, self.activation_level))
        
        # Update access time
        self.last_access_time = current_time
    
    def fill_slot(self, slot_name, value):
        """
        Fill a slot with a value
        
        Parameters:
        - slot_name: Name of the slot to fill
        - value: Value to assign to the slot
        
        Returns:
        - success: Boolean indicating if the slot was successfully filled
        """
        if slot_name not in self.slots:
            return False
        
        # Check marker conditions if they exist
        if slot_name in self.marker_conditions:
            if not self.marker_conditions[slot_name](value):
                return False
        
        # Check assignment conditions if they exist
        if slot_name in self.assignment_conditions:
            if not self.assignment_conditions[slot_name](value, self.slots):
                return False
        
        # Fill the slot
        self.slots[slot_name] = value
        return True
    
    def get_state(self):
        """
        Get the current state of the frame
        
        Returns:
        - state: Dictionary containing frame state
        """
        return {
            'name': self.name,
            'fixed_conditions': self.fixed_conditions.copy(),
            'slots': self.slots.copy(),
            'activation': self.activation_level,
            'last_access': self.last_access_time
        }


class MemorySystem:
    """
    Implementation of the Information Retrieval Network (IRN) that serves as both 
    individual and collective memory
    
    The IRN maintains:
    - Individual memory spaces for each agent
    - Collective memory shared across agents
    - Strategic memory for tracking strategy effectiveness
    """
    def __init__(self, update_frequency=1.0, memory_decay=0.01, logger=None):
        self.individual_memory = {}  # Maps agent_id to frames
        self.collective_memory = {}  # Shared frames
        self.current_time = 0
        self.strategic_memory = {}  # Maps strategy_id to activation level
        self.update_frequency = update_frequency  # Frequency of frame system updates (f = 1/dt)
        self.memory_decay = memory_decay  # Rate of memory decay
        self.logger = logger or logging.getLogger(__name__)
        
        # Tracking variables
        self.access_history = []
        self.activation_history = []
    
    def initialize_agent_memory(self, agent_id, n_strategies=3):
        """
        Initialize memory for a new agent
        
        Parameters:
        - agent_id: ID of the agent
        - n_strategies: Number of strategies
        """
        if agent_id not in self.individual_memory:
            self.individual_memory[agent_id] = {}
            
            # Create basic frames for strategies
            for i in range(n_strategies):
                frame_name = f"strategy_{i}"
                frame = Frame(
                    name=frame_name,
                    fixed_conditions={"strategy_id": i},
                    slots={"position": None, "outcome": None, "strategy_vector": None}
                )
                self.individual_memory[agent_id][frame_name] = frame
            
            self.logger.debug(f"Initialized memory for agent {agent_id} with {n_strategies} strategies")
    
    def update(self, dt=0.1):
        """
        Update the memory system
        
        Parameters:
        - dt: Time step size
        """
        self.current_time += dt
        
        # Only update at specified frequency
        if self.current_time % (1.0 / self.update_frequency) < dt:
            # Decay activations over time
            for agent_id, frames in self.individual_memory.items():
                for frame_name, frame in frames.items():
                    frame.update_activation(0, self.current_time)
            
            for frame_name, frame in self.collective_memory.items():
                frame.update_activation(0, self.current_time)
            
            # Record activation levels
            avg_individual = np.mean([
                np.mean([frame.activation_level for frame in frames.values()])
                for frames in self.individual_memory.values()
            ]) if self.individual_memory else 0
            
            avg_collective = np.mean([
                frame.activation_level for frame in self.collective_memory.values()
            ]) if self.collective_memory else 0
            
            self.activation_history.append({
                'time': self.current_time,
                'individual': avg_individual,
                'collective': avg_collective
            })
    
    def store_experience(self, agent_id, state, outcome):
        """
        Store an agent's experience in memory
        
        Parameters:
        - agent_id: ID of the agent
        - state: Dictionary containing state information
        - outcome: Outcome value (0 to 1)
        """
        # Ensure agent has memory initialized
        if agent_id not in self.individual_memory:
            self.initialize_agent_memory(agent_id)
        
        # Create a new frame for this experience
        frame_name = f"experience_{agent_id}_{self.current_time}"
        
        # Extract strategy information
        strategy_vector = state.get('strategy', None)
        dominant_strategy = np.argmax(strategy_vector) if strategy_vector is not None else None
        
        # Create frame
        frame = Frame(
            name=frame_name,
            fixed_conditions={
                "agent_id": agent_id,
                "time": self.current_time,
                "dominant_strategy": dominant_strategy
            },
            slots={
                "position": state.get('position', None),
                "strategy_vector": strategy_vector,
                "outcome": outcome
            }
        )
        
        # Store in individual memory
        self.individual_memory[agent_id][frame_name] = frame
        
        # Activate this frame
        frame.update_activation(outcome, self.current_time)
        
        # If outcome is good, also store in collective memory
        if outcome > 0.7:
            self.collective_memory[frame_name] = frame
        
        # Update strategic memory
        if dominant_strategy is not None:
            strategy_id = f"strategy_{dominant_strategy}"
            if strategy_id not in self.strategic_memory:
                self.strategic_memory[strategy_id] = 0.5  # Initial neutral activation
            
            # Update based on outcome
            current = self.strategic_memory[strategy_id]
            self.strategic_memory[strategy_id] = current * 0.9 + outcome * 0.1
        
        # Record access
        self.access_history.append({
            'time': self.current_time,
            'agent_id': agent_id,
            'frame': frame_name,
            'outcome': outcome
        })
    
    def retrieve_relevant_frames(self, agent_id, state, max_frames=3):
        """
        Retrieve frames relevant to the current state
        
        Parameters:
        - agent_id: ID of the agent
        - state: Dictionary containing state information
        - max_frames: Maximum number of frames to retrieve
        
        Returns:
        - frames: List of relevant frames
        """
        # Ensure agent has memory initialized
        if agent_id not in self.individual_memory:
            self.initialize_agent_memory(agent_id)
        
        # Calculate similarity between state and each frame
        similarities = []
        
        # Check individual memory
        for frame_name, frame in self.individual_memory[agent_id].items():
            similarity = self._calculate_similarity(state, frame)
            if similarity > 0:
                similarities.append((frame, similarity))
        
        # Check collective memory
        for frame_name, frame in self.collective_memory.items():
            similarity = self._calculate_similarity(state, frame)
            if similarity > 0:
                similarities.append((frame, similarity))
        
        # Sort by similarity and return top frames
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_frames]
    
    def _calculate_similarity(self, state, frame):
        """
        Calculate similarity between a state and a frame
        
        Parameters:
        - state: Dictionary containing state information
        - frame: Frame object
        
        Returns:
        - similarity: Similarity score (0 to 1)
        """
        similarity = 0.0
        
        # Check fixed conditions
        for key, value in frame.fixed_conditions.items():
            if key in state and state[key] == value:
                similarity += 0.5
        
        # Check position similarity if available
        if 'position' in state and 'position' in frame.slots and frame.slots['position'] is not None:
            state_pos = np.array(state['position'])
            frame_pos = np.array(frame.slots['position'])
            
            # Calculate distance-based similarity
            distance = np.linalg.norm(state_pos - frame_pos)
            position_similarity = max(0, 1 - distance / 10)  # Normalize by max expected distance
            similarity += position_similarity * 0.3
        
        # Check strategy similarity if available
        if 'strategy' in state and 'strategy_vector' in frame.slots and frame.slots['strategy_vector'] is not None:
            state_strat = np.array(state['strategy'])
            frame_strat = np.array(frame.slots['strategy_vector'])
            
            # Calculate cosine similarity
            dot_product = np.dot(state_strat, frame_strat)
            norm_product = np.linalg.norm(state_strat) * np.linalg.norm(frame_strat)
            
            if norm_product > 0:
                strategy_similarity = dot_product / norm_product
                similarity += strategy_similarity * 0.2
        
        return similarity
    
    def get_strategy_influence(self, agent_id, state):
        """
        Get strategy influence from memory based on current state
        
        Parameters:
        - agent_id: ID of the agent
        - state: Dictionary containing state information
        
        Returns:
        - strategy_influence: Strategy vector influenced by memory
        """
        # Retrieve relevant frames
        relevant_frames = self.retrieve_relevant_frames(agent_id, state)
        
        if not relevant_frames:
            # Return uniform strategy if no relevant frames
            n_strategies = len(state.get('strategy', []))
            if n_strategies > 0:
                return np.ones(n_strategies) / n_strategies
            else:
                return None
        
        # Weight strategies by similarity and outcome
        weighted_strategies = []
        total_weight = 0
        
        for frame, similarity in relevant_frames:
            if 'strategy_vector' in frame.slots and frame.slots['strategy_vector'] is not None:
                strategy = frame.slots['strategy_vector']
                outcome = frame.slots.get('outcome', 0.5)
                
                # Weight by similarity and outcome
                weight = similarity * (outcome + 0.5)  # Ensure even bad outcomes have some influence
                
                weighted_strategies.append((strategy, weight))
                total_weight += weight
        
        if not weighted_strategies or total_weight == 0:
            # Return uniform strategy if no valid strategies
            n_strategies = len(state.get('strategy', []))
            if n_strategies > 0:
                return np.ones(n_strategies) / n_strategies
            else:
                return None
        
        # Calculate weighted average strategy
        result = np.zeros_like(weighted_strategies[0][0])
        for strategy, weight in weighted_strategies:
            result += strategy * (weight / total_weight)
        
        return result
    
    def get_memory_stats(self):
        """
        Get statistics about the memory system
        
        Returns:
        - stats: Dictionary of memory statistics
        """
        individual_count = sum(len(frames) for frames in self.individual_memory.values())
        collective_count = len(self.collective_memory)
        
        # Calculate average activation
        individual_activation = np.mean([
            np.mean([frame.activation_level for frame in frames.values()])
            for frames in self.individual_memory.values()
        ]) if self.individual_memory else 0
        
        collective_activation = np.mean([
            frame.activation_level for frame in self.collective_memory.values()
        ]) if self.collective_memory else 0
        
        return {
            'individual_frames': individual_count,
            'collective_frames': collective_count,
            'individual_activation': individual_activation,
            'collective_activation': collective_activation,
            'strategic_memory': self.strategic_memory.copy()
        }
