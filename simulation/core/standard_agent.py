"""
Standard Agent implementation for the MMAI system.
This agent uses simple rules for decision making without LLM capabilities.
"""

import numpy as np
from .agent import Agent

class StandardAgent(Agent):
    """
    Standard agent implementation with rule-based decision making.
    """
    
    def __init__(self, agent_id, position, n_strategies, logger=None):
        """
        Initialize a standard agent
        
        Parameters:
        - agent_id: Unique identifier for the agent
        - position: Initial position as numpy array [x, y]
        - n_strategies: Number of strategies available
        - logger: Logger instance
        """
        super().__init__(agent_id, position, n_strategies, logger)
        self.type = "standard"
    
    def make_decision(self, observation):
        """
        Make a decision based on the current observation
        
        Parameters:
        - observation: Dictionary containing the agent's observation of the environment
        
        Returns:
        - decision: Dictionary containing the agent's decision
        """
        # Extract observation components
        position = np.array(observation["position"])
        field_state = np.array(observation["field_state"])
        env_state = observation["env_state"]
        resource_level = observation["resource_level"]
        energy = observation["energy"]
        nearby_agents = observation.get("nearby_agents", [])
        
        # Initialize decision
        decision = {
            "move_direction": np.zeros(2),
            "strategy_change": np.zeros(self.n_strategies),
            "action": "explore"
        }
        
        # Simple rule-based decision making
        
        # 1. Movement: Move toward resources if energy is low, otherwise explore
        if energy < 0.3:
            # Find direction with most resources
            if resource_level > 0.5:
                # Stay in place to gather resources
                decision["move_direction"] = np.zeros(2)
                decision["action"] = "gather"
            else:
                # Random movement to find resources
                decision["move_direction"] = np.array([
                    np.random.uniform(-1, 1),
                    np.random.uniform(-1, 1)
                ])
                decision["action"] = "search_resources"
        else:
            # Explore with random movement
            decision["move_direction"] = np.array([
                np.random.uniform(-1, 1),
                np.random.uniform(-1, 1)
            ])
            decision["action"] = "explore"
        
        # 2. Strategy: Adapt to environment and field state
        
        # Calculate dominant strategy in the field
        dominant_strategy_idx = np.argmax(field_state)
        
        # Calculate current strategy distribution
        strategy_distribution = self.strategy / np.sum(self.strategy)
        
        # Tendency to conform to the dominant strategy
        conformity_factor = 0.2
        
        # Tendency to differentiate if many nearby agents
        differentiation_factor = 0.1 * len(nearby_agents)
        
        # Calculate strategy change
        for i in range(self.n_strategies):
            if i == dominant_strategy_idx:
                # Increase probability of dominant strategy
                decision["strategy_change"][i] = conformity_factor * (1 - strategy_distribution[i])
            else:
                # Decrease probability of non-dominant strategies
                decision["strategy_change"][i] = -conformity_factor * strategy_distribution[i]
        
        # If many nearby agents, try to differentiate
        if len(nearby_agents) > 2:
            # Find least common strategy among nearby agents
            nearby_strategies = np.zeros(self.n_strategies)
            for agent in nearby_agents:
                if "strategy" in agent:
                    nearby_strategies += np.array(agent["strategy"])
            
            if np.sum(nearby_strategies) > 0:
                least_common_idx = np.argmin(nearby_strategies)
                
                # Increase probability of least common strategy
                decision["strategy_change"][least_common_idx] += differentiation_factor
                
                # Normalize strategy change to ensure sum is zero
                decision["strategy_change"] -= np.mean(decision["strategy_change"])
        
        # Ensure strategy changes are small
        decision["strategy_change"] = np.clip(decision["strategy_change"], -0.1, 0.1)
        
        return decision
    
    def update(self, dt, observation=None):
        """
        Update agent state based on observation and time step
        
        Parameters:
        - dt: Time step
        - observation: Dictionary containing the agent's observation of the environment
        
        Returns:
        - updated: Boolean indicating if the agent state was updated
        """
        if observation is None:
            return False
        
        # Make decision based on observation
        decision = self.make_decision(observation)
        
        # Update position based on decision
        move_direction = np.array(decision["move_direction"])
        if np.linalg.norm(move_direction) > 0:
            # Normalize direction vector
            move_direction = move_direction / np.linalg.norm(move_direction)
            
            # Calculate movement distance based on energy and dt
            move_speed = 1.0 * min(1.0, self.energy / 0.5)  # Reduce speed when low energy
            move_distance = move_speed * dt
            
            # Update position
            self.position += move_direction * move_distance
            
            # Ensure position is within bounds
            self.position = np.clip(self.position, 0, self.grid_size - 1)
            
            # Consume energy for movement
            self.energy -= 0.01 * move_distance
        
        # Update strategy based on decision
        strategy_change = np.array(decision["strategy_change"])
        if np.any(strategy_change != 0):
            self.strategy += strategy_change * dt
            self.strategy = np.clip(self.strategy, 0.01, 1.0)  # Ensure strategy values are positive
            self.strategy = self.strategy / np.sum(self.strategy)  # Normalize
        
        # Consume energy over time
        self.energy -= 0.005 * dt
        
        # Gather resources if available
        if decision["action"] == "gather" and "resource_level" in observation:
            resource_gain = min(0.1 * dt, observation["resource_level"])
            self.energy += resource_gain
        
        # Cap energy at maximum
        self.energy = min(self.energy, self.max_energy)
        
        # Check if agent is still alive
        if self.energy <= 0:
            self.alive = False
            if self.logger:
                self.logger.debug(f"Agent {self.agent_id} died due to energy depletion")
        
        return True
    
    def get_performance_stats(self):
        """
        Get performance statistics for the agent
        
        Returns:
        - stats: Dictionary containing performance statistics
        """
        return {
            "type": "standard",
            "lifetime": self.lifetime,
            "energy": float(self.energy),
            "position": self.position.tolist(),
            "strategy": self.strategy.tolist()
        }
