#!/usr/bin/env python3
"""
Simple test script for the LLM-enhanced MMAI system
"""

import os
import sys
import logging
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import simulation components
from core.environment_system import EnvironmentSystem, EnvironmentType
from core.resource_system import ResourceSystem
from core.strategic_field import StrategicField
from core.llm_agent import LLMAgent
from core.standard_agent import StandardAgent
from core.llm_enhanced_nash_validator import LLMEnhancedNashValidator

def setup_logging():
    """Setup logging for the simulation"""
    # Create logger
    logger = logging.getLogger("llm_test")
    logger.setLevel(logging.INFO)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(ch)
    
    return logger

def main():
    """Main entry point"""
    logger = setup_logging()
    logger.info("Starting simple LLM test")
    
    # Configuration
    grid_size = 10
    n_strategies = 3
    n_agents = 3
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    region = "us-east-1"
    
    # Initialize components
    logger.info("Initializing components")
    
    # Create environment
    env = EnvironmentSystem(grid_size=grid_size, env_type=EnvironmentType.STATIC, logger=logger)
    
    # Create strategic field
    field = StrategicField(grid_size=grid_size, n_strategies=n_strategies, logger=logger)
    
    # Create resource system
    resources = ResourceSystem(grid_size=grid_size, initial_density=0.3, logger=logger)
    
    # Create Nash validator
    nash_validator = LLMEnhancedNashValidator(
        n_strategies=n_strategies,
        equilibrium_threshold=0.1,
        growth_rate=0.05,
        model_id=model_id,
        region=region,
        logger=logger
    )
    
    # Create agents
    agents = []
    
    # Create standard agents
    for i in range(2):
        agent = StandardAgent(
            agent_id=i,
            position=np.array([np.random.randint(0, grid_size), np.random.randint(0, grid_size)]),
            n_strategies=n_strategies,
            logger=logger
        )
        agents.append(agent)
    
    # Create LLM agent
    llm_agent = LLMAgent(
        agent_id=2,
        position=np.array([np.random.randint(0, grid_size), np.random.randint(0, grid_size)]),
        n_strategies=n_strategies,
        model_id=model_id,
        region=region,
        logger=logger
    )
    agents.append(llm_agent)
    
    logger.info(f"Created {len(agents)} agents ({len(agents)-1} standard, 1 LLM)")
    
    # Run a simple test
    logger.info("Running simple test")
    
    # Update environment
    env_state = env.update(dt=0.1)
    logger.info(f"Environment state: {env_state:.4f}")
    
    # Update resources
    total_resources = resources.update(agents=agents, dt=0.1)
    logger.info(f"Total resources: {total_resources:.4f}")
    
    # Update strategic field
    field.update(agents=agents, dt=0.1)
    
    # Test Nash validator
    payoff_matrix = nash_validator.calculate_payoff_matrix(agents)
    logger.info(f"Payoff matrix shape: {payoff_matrix.shape}")
    
    nash_distance, best_responses = nash_validator.calculate_nash_distance(agents, payoff_matrix)
    logger.info(f"Nash distance: {nash_distance:.4f}")
    
    # Test LLM agent
    logger.info("Testing LLM agent decision making")
    
    # Get field state at agent position
    field_state = field.get_strategy_at_position(llm_agent.position)
    
    # Get environment state
    env_state = env.get_state_at(llm_agent.position)
    
    # Get resource level
    resource_level = resources.get_resources_at(llm_agent.position)
    
    # Create observation
    observation = {
        "position": llm_agent.position.tolist(),
        "field_state": field_state.tolist(),
        "env_state": float(env_state),
        "resource_level": float(resource_level),
        "energy": float(llm_agent.energy),
        "strategy": llm_agent.strategy.tolist(),
        "nearby_agents": []
    }
    
    # Create a simple environment state for testing
    environment_state = {
        "position": llm_agent.position.tolist(),
        "resources": float(resource_level),
        "environment": float(env_state),
        "nearby_agents": []
    }
    
    # Make decision
    try:
        # Create a mock memory system for testing
        from core.llm_enhanced_memory import LLMEnhancedMemorySystem
        memory_system = LLMEnhancedMemorySystem(
            capacity=10,
            n_frames=5,
            model_id=model_id,
            region=region,
            logger=logger
        )
        
        # Call the make_decision method with all required arguments
        decision = llm_agent.make_decision(environment_state, field, memory_system)
        logger.info(f"LLM agent decision: {decision}")
    except Exception as e:
        logger.error(f"Error in LLM agent decision making: {str(e)}")
    
    logger.info("Test completed")

if __name__ == "__main__":
    main()
