#!/usr/bin/env python3
"""
LLM-powered Agent for the unified MMAI system

This module implements an enhanced agent that leverages Amazon Bedrock
for reasoning and decision-making capabilities.
"""

import numpy as np
import logging
import json
import boto3
import time
from core.agent import Agent

class LLMAgent(Agent):
    """
    Agent powered by a large language model via Amazon Bedrock
    
    This agent extends the base Agent class with LLM-based reasoning capabilities,
    allowing for more sophisticated strategic decision-making and adaptation.
    """
    def __init__(self, agent_id, position, 
                 model_id="anthropic.claude-3-sonnet-20240229-v1:0", 
                 region="us-east-1",
                 system_prompt=None,
                 n_strategies=3, 
                 memory_depth=10,
                 initial_energy=1.0, 
                 energy_decay=0.01, 
                 reproduction_threshold=2.0,
                 hypersensitive_threshold=0.1, 
                 logger=None):
        """
        Initialize the LLM-powered agent
        
        Parameters:
        - agent_id: Unique identifier for the agent
        - position: Initial position [x, y]
        - model_id: Amazon Bedrock model ID
        - region: AWS region for Bedrock
        - system_prompt: System prompt for the LLM
        - n_strategies: Number of strategies
        - memory_depth: Depth of memory for outcomes
        - initial_energy: Initial energy level
        - energy_decay: Rate of energy decay per step
        - reproduction_threshold: Energy threshold for reproduction
        - hypersensitive_threshold: Threshold for hypersensitive points
        """
        super().__init__(agent_id, position, n_strategies, memory_depth, 
                        initial_energy, energy_decay, reproduction_threshold,
                        hypersensitive_threshold, logger)
        
        self.model_id = model_id
        self.region = region
        
        # Initialize Bedrock client
        self.bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name=self.region
        )
        
        # Set default system prompt if none provided
        if system_prompt is None:
            self.system_prompt = (
                "You are an intelligent agent in a multi-agent simulation based on the principles of collective behavioral intelligence. "
                "Your goal is to make optimal strategic decisions that: "
                "1. Contribute to the formation of coherent strategic fields across the environment "
                "2. Align with Nash equilibrium proportional to swarm growth "
                "3. Balance individual adaptation with collective coordination through stigmergic channels "
                "Your decisions should consider both individual payoffs and collective field coherence. "
                "Always provide your reasoning and return strategy weights as a JSON array of probabilities that sum to 1.0."
            )
        else:
            self.system_prompt = system_prompt
            
        # Conversation history for context
        self.conversation_history = []
        
        # Reasoning cache to avoid redundant LLM calls
        self.reasoning_cache = {}
        
        # Performance tracking
        self.llm_response_times = []
        self.llm_call_count = 0
        
        self.logger.info(f"Initialized LLM Agent {agent_id} with model {model_id}")
    
    def make_decision(self, environment_state, strategic_field, memory_system=None):
        """
        Make strategic decisions using LLM reasoning
        
        Parameters:
        - environment_state: Current state of the environment
        - strategic_field: Strategic field object
        - memory_system: Optional memory system for additional context
        
        Returns:
        - decision: Index of the selected strategy
        - reasoning: Reasoning behind the decision
        """
        # Check if we're at a hypersensitive point
        is_hypersensitive = self.check_hypersensitivity()
        
        # Construct cache key from state
        cache_key = self._construct_cache_key(environment_state, is_hypersensitive)
        
        # Check cache first
        if cache_key in self.reasoning_cache:
            self.logger.debug(f"Agent {self.agent_id} using cached reasoning")
            strategy_weights, reasoning = self.reasoning_cache[cache_key]
        else:
            # Construct prompt with environment state, strategic field info, and memory
            prompt = self._construct_prompt(environment_state, strategic_field, 
                                          memory_system, is_hypersensitive)
            
            # Get LLM response
            start_time = time.time()
            response = self._query_llm(prompt)
            elapsed_time = time.time() - start_time
            self.llm_response_times.append(elapsed_time)
            self.llm_call_count += 1
            
            # Parse response to extract strategy weights and reasoning
            strategy_weights, reasoning = self._parse_llm_response(response)
            
            # Cache the result
            self.reasoning_cache[cache_key] = (strategy_weights, reasoning)
        
        # Update strategy based on LLM reasoning
        self.strategy = strategy_weights
        
        # Store reasoning in memory if we have a memory system
        if memory_system is not None:
            memory_entry = {
                'agent_id': self.agent_id,
                'position': self.position.tolist(),
                'strategy': self.strategy.tolist(),
                'reasoning': reasoning,
                'is_hypersensitive': is_hypersensitive
            }
            memory_system.add_memory(memory_entry)
        
        # Make final decision
        decision = np.argmax(self.strategy)
        
        self.logger.debug(f"Agent {self.agent_id} made decision {decision} with reasoning: {reasoning[:50]}...")
        
        return decision, reasoning
    
    def _construct_cache_key(self, environment_state, is_hypersensitive):
        """Construct a cache key from the environment state"""
        # Simplify environment state to key components for caching
        if isinstance(environment_state, dict):
            key_components = []
            for k in sorted(environment_state.keys()):
                if isinstance(environment_state[k], (int, float, str, bool)):
                    key_components.append(f"{k}:{environment_state[k]}")
                elif isinstance(environment_state[k], (list, np.ndarray)):
                    # For arrays, use first few and last few elements
                    arr = environment_state[k]
                    if len(arr) > 6:
                        key_components.append(f"{k}:[{arr[0]},{arr[1]},{arr[2]}...{arr[-3]},{arr[-2]},{arr[-1]}]")
                    else:
                        key_components.append(f"{k}:{arr}")
            
            # Add position and hypersensitivity
            key_components.append(f"pos:{self.position.tolist()}")
            key_components.append(f"hyp:{is_hypersensitive}")
            
            return "|".join(key_components)
        else:
            # Fallback for non-dict states
            return f"state:{hash(str(environment_state))}|pos:{self.position.tolist()}|hyp:{is_hypersensitive}"
    
    def _construct_prompt(self, environment_state, strategic_field, memory_system, is_hypersensitive):
        """
        Construct prompt for the LLM with relevant context
        
        Parameters:
        - environment_state: Current state of the environment
        - strategic_field: Strategic field object
        - memory_system: Optional memory system for additional context
        - is_hypersensitive: Whether the agent is at a hypersensitive point
        
        Returns:
        - prompt: Formatted prompt for the LLM
        """
        # Get strategy vector at current position
        position_strategy = strategic_field.get_strategy_at_position(self.position)
        
        # Format environment state
        env_state_str = json.dumps(environment_state, default=str)
        
        # Format agent state
        agent_state = {
            "position": self.position.tolist(),
            "current_strategy": self.strategy.tolist(),
            "energy": self.energy,
            "age": self.age,
            "is_hypersensitive": is_hypersensitive
        }
        agent_state_str = json.dumps(agent_state)
        
        # Get relevant memories if memory system is provided
        memory_str = "No memory system available."
        if memory_system is not None:
            # Create a query state from current position and environment
            query_state = {
                "position": self.position.tolist(),
                "environment": environment_state
            }
            
            # Retrieve relevant memories
            memories = memory_system.retrieve_memories(query_state, k=3)
            if memories:
                memory_str = json.dumps(memories, default=str)
        
        # Construct the prompt
        prompt = f"""
        You are Agent {self.agent_id}, an intelligent entity in a multi-agent simulation operating within a fractal time architecture.

        CURRENT ENVIRONMENT STATE:
        {env_state_str}

        YOUR STATE:
        {agent_state_str}

        STRATEGIC FIELD AT YOUR POSITION:
        {position_strategy.tolist()}

        RELEVANT MEMORIES:
        {memory_str}

        STRATEGIC CONTEXT:
        - You operate across three temporal scales: immediate (dt), intermediate (t), and long-term (T)
        - Your decisions contribute to strategic wave formation across the environment
        - Each strategy represents a different approach to resource acquisition and field influence:
          * Strategy 1: Exploitative (maximizes immediate resource gain)
          * Strategy 2: Explorative (discovers new resource opportunities)
          * Strategy 3: Cooperative (strengthens field coherence)
        - Your goal is to balance survival with contribution to strategic field coherence

        TASK:
        Based on the above information, determine the optimal strategy weights for your next decision.
        {"You are at a HYPERSENSITIVE POINT where small differences in strategy weights can lead to different decisions. At hypersensitive points, your choices have greater influence on strategic field formation." if is_hypersensitive else ""}

        INSTRUCTIONS:
        1. Analyze the environment state, your current position, and the strategic field
        2. Consider your current energy level and age in relation to reproduction threshold
        3. Review relevant memories for patterns of successful strategies
        4. Determine optimal strategy weights that:
           - Ensure your survival through efficient resource acquisition
           - Contribute to strategic field coherence
           - Align with emerging Nash equilibrium patterns
        5. Provide clear reasoning for your decision, including how it balances individual needs with collective patterns

        RESPONSE FORMAT:
        Provide your response in the following format:
        ```json
        {{
            "strategy_weights": [float, float, ...],
            "reasoning": "Your detailed reasoning here"
        }}
        ```

        Ensure that strategy_weights is an array of {self.n_strategies} probabilities that sum to 1.0.
        """
        
        return prompt
    
    def _query_llm(self, prompt):
        """
        Query the LLM with the constructed prompt
        
        Parameters:
        - prompt: Formatted prompt for the LLM
        
        Returns:
        - response: LLM response text
        """
        try:
            # Prepare the request based on the model type
            if "anthropic.claude" in self.model_id:
                # Claude-specific request format
                request_body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"{self.system_prompt}\n\n{prompt}"
                        }
                    ],
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            elif "amazon.titan" in self.model_id:
                # Titan-specific request format
                request_body = {
                    "inputText": f"<system>{self.system_prompt}</system>\n\n{prompt}",
                    "textGenerationConfig": {
                        "maxTokenCount": 1000,
                        "temperature": 0.7,
                        "topP": 0.9
                    }
                }
            else:
                # Generic format for other models
                request_body = {
                    "prompt": f"{self.system_prompt}\n\n{prompt}",
                    "max_tokens": 1000,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            
            # Make the API call
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            
            # Parse the response based on the model type
            response_body = json.loads(response.get("body").read())
            
            if "anthropic.claude" in self.model_id:
                return response_body.get("content")[0].get("text")
            elif "amazon.titan" in self.model_id:
                return response_body.get("results")[0].get("outputText")
            else:
                return response_body.get("completion")
            
        except Exception as e:
            self.logger.error(f"Error querying LLM: {str(e)}")
            # Return a fallback response
            return json.dumps({
                "strategy_weights": [1.0/self.n_strategies] * self.n_strategies,
                "reasoning": "Error querying LLM. Using uniform strategy weights."
            })
    
    def _parse_llm_response(self, response):
        """
        Parse LLM response to extract strategy weights and reasoning
        
        Parameters:
        - response: LLM response text
        
        Returns:
        - strategy_weights: Array of strategy weights
        - reasoning: Reasoning text
        """
        try:
            # Extract JSON from response (it might be wrapped in markdown code blocks)
            json_match = response
            
            # If response contains markdown code blocks, extract the JSON
            if "```json" in response:
                json_match = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_match = response.split("```")[1].split("```")[0].strip()
            
            # Parse the JSON
            parsed = json.loads(json_match)
            
            # Extract strategy weights and reasoning
            strategy_weights = np.array(parsed["strategy_weights"])
            reasoning = parsed["reasoning"]
            
            # Ensure strategy weights sum to 1
            strategy_weights = strategy_weights / np.sum(strategy_weights)
            
            return strategy_weights, reasoning
            
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {str(e)}")
            self.logger.error(f"Response was: {response[:100]}...")
            
            # Return uniform strategy weights as fallback
            return np.ones(self.n_strategies) / self.n_strategies, "Error parsing LLM response"
    
    def reproduce(self):
        """
        Create a new agent through reproduction with LLM capabilities
        
        Returns:
        - child: New LLM agent with inherited properties
        - success: Boolean indicating if reproduction was successful
        """
        if not self.can_reproduce():
            return None, False
        
        # Create child with similar properties but some mutation
        child_id = self.agent_id + 1000  # Temporary ID, should be updated by simulation
        
        # Position near parent
        child_position = self.position + np.random.randint(-2, 3, size=2)
        
        # Create child with same LLM capabilities
        child = LLMAgent(
            agent_id=child_id,
            position=child_position,
            model_id=self.model_id,
            region=self.region,
            system_prompt=self.system_prompt,
            n_strategies=self.n_strategies,
            memory_depth=self.memory_depth,
            initial_energy=self.energy * 0.3,  # Child gets 30% of parent's energy
            energy_decay=self.energy_decay * (0.9 + 0.2 * np.random.random()),  # Slight mutation
            reproduction_threshold=self.reproduction_threshold,
            hypersensitive_threshold=self.hypersensitive_threshold,
            logger=self.logger
        )
        
        # Inherit strategy with some mutation
        mutation_rate = 0.1
        child.strategy = (1 - mutation_rate) * self.strategy + mutation_rate * np.random.random(self.n_strategies)
        child.strategy = child.strategy / np.sum(child.strategy)
        
        # Inherit some hypersensitive points
        if self.hypersensitive_points:
            for point in self.hypersensitive_points:
                if np.random.random() < 0.5:  # 50% chance to inherit each point
                    child.hypersensitive_points.append(point.copy())
        
        # Parent loses energy
        self.energy *= 0.7  # Parent keeps 70% of energy
        
        self.logger.info(f"LLM Agent {self.agent_id} reproduced, creating agent {child_id}")
        
        return child, True
    
    def get_performance_stats(self):
        """
        Get performance statistics for the LLM agent
        
        Returns:
        - stats: Dictionary of performance statistics
        """
        stats = {
            "llm_call_count": self.llm_call_count,
            "avg_response_time": np.mean(self.llm_response_times) if self.llm_response_times else 0,
            "cache_hit_rate": 1.0 - (self.llm_call_count / max(1, len(self.reasoning_cache) + self.llm_call_count)),
            "cache_size": len(self.reasoning_cache)
        }
        
        return stats
