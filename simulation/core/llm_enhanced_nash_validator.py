#!/usr/bin/env python3
"""
LLM-Enhanced Nash Validator for the unified MMAI system

This module implements an enhanced Nash validator that leverages Amazon Bedrock
for sophisticated game theory analysis and equilibrium detection.
"""

import numpy as np
import logging
import json
import boto3
import time
from core.nash_validator import NashValidator

class LLMEnhancedNashValidator(NashValidator):
    """
    Nash validator enhanced with LLM-based game theory analysis
    
    This validator extends the base NashValidator with LLM-powered capabilities for:
    - More sophisticated equilibrium analysis
    - Strategic pattern recognition
    - Payoff matrix interpretation
    - Prediction of equilibrium shifts
    """
    def __init__(self, n_strategies=3, equilibrium_threshold=0.1, growth_rate=0.05,
                 model_id="anthropic.claude-3-sonnet-20240229-v1:0", 
                 region="us-east-1",
                 system_prompt=None,
                 logger=None):
        """
        Initialize the LLM-enhanced Nash validator
        
        Parameters:
        - n_strategies: Number of strategies
        - equilibrium_threshold: Maximum distance to consider as equilibrium
        - growth_rate: System growth rate (g)
        - model_id: Amazon Bedrock model ID
        - region: AWS region for Bedrock
        - system_prompt: System prompt for the LLM
        - logger: Optional logger instance
        """
        super().__init__(n_strategies, equilibrium_threshold, growth_rate, logger)
        
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
                "You are an advanced game theory analyzer for a multi-agent simulation. "
                "Your role is to analyze strategic interactions, identify Nash equilibria, "
                "and provide insights into the dynamics of the system. Always provide your "
                "analysis in a structured JSON format as specified in each prompt."
            )
        else:
            self.system_prompt = system_prompt
            
        # Performance tracking
        self.llm_response_times = []
        self.llm_call_count = 0
        
        # Analysis cache
        self.analysis_cache = {}
        
        self.logger.info(f"Initialized LLM-Enhanced Nash Validator with model {model_id}")
    
    def analyze_equilibrium_dynamics(self, time_step=None):
        """
        Use LLM to analyze equilibrium dynamics
        
        Parameters:
        - time_step: Optional time step to analyze (defaults to latest)
        
        Returns:
        - analysis: Dictionary containing equilibrium analysis
        """
        # Determine data range to analyze
        if time_step is None:
            # Use the last 20 time steps or all available data if less
            start_idx = max(0, len(self.nash_distance_history) - 20)
            data_range = slice(start_idx, None)
        else:
            # Use data up to the specified time step
            data_range = slice(0, min(time_step + 1, len(self.nash_distance_history)))
        
        # Extract key data about equilibrium history
        data = {
            'nash_distance': self.nash_distance_history[data_range],
            'strategy_distribution': self.strategy_distribution_history[data_range] if self.strategy_distribution_history else None,
            'payoff_matrix': self.payoff_matrix_history[-1].tolist() if self.payoff_matrix_history else None,
            'equilibrium_history': self.equilibrium_history[data_range] if self.equilibrium_history else None,
            'growth_rate': self.growth_rate
        }
        
        # Create cache key
        cache_key = f"eq_analysis_{hash(str(data))}"
        
        # Check cache first
        if cache_key in self.analysis_cache:
            self.logger.debug(f"Using cached equilibrium analysis")
            return self.analysis_cache[cache_key]
        
        # Construct prompt
        prompt = f"""
        Analyze the following game-theoretic data and provide insights:
        
        Nash Distance History: {data['nash_distance']}
        Strategy Distribution History: {data['strategy_distribution']}
        Current Payoff Matrix: {data['payoff_matrix']}
        Equilibrium History: {data['equilibrium_history']}
        System Growth Rate: {data['growth_rate']}
        
        Perform a comprehensive game theory analysis addressing:
        
        1. Equilibrium convergence:
           - Is the system converging to equilibrium?
           - At what rate is convergence occurring?
           - How does this relate to the system growth rate?
        
        2. Equilibrium type:
           - What type of equilibrium is emerging (pure, mixed)?
           - Is it a stable or unstable equilibrium?
           - Are there multiple equilibria present?
        
        3. Strategic patterns:
           - Are there dominant or dominated strategies?
           - Is there evidence of strategic cycling?
           - Are there cooperative or competitive dynamics?
        
        4. Predictions:
           - What future equilibrium shifts might occur?
           - What strategic adaptations would accelerate convergence?
           - What perturbations might destabilize the current equilibrium?
        
        RESPONSE FORMAT:
        ```json
        {{
            "convergence_analysis": {{
                "is_converging": boolean,
                "convergence_rate": float,
                "proportional_to_growth": boolean,
                "estimated_steps_to_equilibrium": integer,
                "confidence": float  // 0.0 to 1.0
            }},
            "equilibrium_type": {{
                "pure_strategy": boolean,
                "mixed_strategy": boolean,
                "stable": boolean,
                "multiple_equilibria": boolean,
                "description": "string"
            }},
            "strategic_patterns": [
                {{
                    "pattern_type": "string",  // e.g., "dominance", "cycling", "cooperation"
                    "description": "string",
                    "strength": float  // 0.0 to 1.0
                }},
                ...
            ],
            "predictions": [
                {{
                    "prediction_type": "string",
                    "description": "string",
                    "probability": float,  // 0.0 to 1.0
                    "estimated_timeframe": integer  // time steps
                }},
                ...
            ],
            "recommended_actions": [
                "string",
                ...
            ]
        }}
        ```
        """
        
        # Query LLM
        start_time = time.time()
        response = self._query_llm(prompt)
        elapsed_time = time.time() - start_time
        self.llm_response_times.append(elapsed_time)
        self.llm_call_count += 1
        
        # Parse response
        try:
            analysis = self._extract_json_from_response(response)
            
            # Cache the result
            self.analysis_cache[cache_key] = analysis
            
            return analysis
        except Exception as e:
            self.logger.error(f"Error parsing LLM equilibrium analysis response: {str(e)}")
            # Return empty analysis as fallback
            return {
                "convergence_analysis": {"is_converging": False, "confidence": 0},
                "equilibrium_type": {"description": "Analysis failed"},
                "strategic_patterns": [],
                "predictions": [],
                "recommended_actions": ["Retry analysis with more data"]
            }
    
    def analyze_payoff_matrix(self, payoff_matrix=None):
        """
        Use LLM to analyze the payoff matrix
        
        Parameters:
        - payoff_matrix: Optional payoff matrix to analyze (defaults to latest)
        
        Returns:
        - analysis: Dictionary containing payoff matrix analysis
        """
        # Use provided payoff matrix or the latest one
        if payoff_matrix is None:
            if not self.payoff_matrix_history:
                return {"error": "No payoff matrix available"}
            payoff_matrix = self.payoff_matrix_history[-1]
        
        # Convert to list for JSON serialization
        if isinstance(payoff_matrix, np.ndarray):
            payoff_matrix = payoff_matrix.tolist()
        
        # Create cache key
        cache_key = f"payoff_analysis_{hash(str(payoff_matrix))}"
        
        # Check cache first
        if cache_key in self.analysis_cache:
            self.logger.debug(f"Using cached payoff matrix analysis")
            return self.analysis_cache[cache_key]
        
        # Construct prompt
        prompt = f"""
        Analyze the following payoff matrix and provide game theory insights:
        
        Payoff Matrix:
        {payoff_matrix}
        
        Perform a comprehensive analysis addressing:
        
        1. Nash equilibria:
           - Identify all pure strategy Nash equilibria
           - Estimate mixed strategy equilibria if applicable
           - Evaluate the stability of each equilibrium
        
        2. Strategic relationships:
           - Identify dominant and dominated strategies
           - Detect zero-sum or non-zero-sum characteristics
           - Identify cooperative or competitive dynamics
        
        3. Game classification:
           - Classify the game type (e.g., Prisoner's Dilemma, Coordination, etc.)
           - Identify key strategic tensions in the game
        
        RESPONSE FORMAT:
        ```json
        {{
            "nash_equilibria": [
                {{
                    "type": "string",  // "pure" or "mixed"
                    "strategies": [float, ...],  // Strategy probabilities
                    "stability": float,  // 0.0 to 1.0
                    "description": "string"
                }},
                ...
            ],
            "strategic_relationships": {{
                "dominant_strategies": [integer, ...],  // Strategy indices
                "dominated_strategies": [integer, ...],  // Strategy indices
                "zero_sum": boolean,
                "cooperative_potential": float,  // 0.0 to 1.0
                "competitive_intensity": float  // 0.0 to 1.0
            }},
            "game_classification": {{
                "game_type": "string",
                "strategic_tensions": [
                    "string",
                    ...
                ],
                "key_characteristics": [
                    "string",
                    ...
                ]
            }},
            "recommended_strategies": [
                {{
                    "strategy_index": integer,
                    "rationale": "string",
                    "expected_payoff": float
                }},
                ...
            ]
        }}
        ```
        """
        
        # Query LLM
        start_time = time.time()
        response = self._query_llm(prompt)
        elapsed_time = time.time() - start_time
        self.llm_response_times.append(elapsed_time)
        self.llm_call_count += 1
        
        # Parse response
        try:
            analysis = self._extract_json_from_response(response)
            
            # Cache the result
            self.analysis_cache[cache_key] = analysis
            
            return analysis
        except Exception as e:
            self.logger.error(f"Error parsing LLM payoff matrix analysis response: {str(e)}")
            # Return empty analysis as fallback
            return {
                "nash_equilibria": [],
                "strategic_relationships": {"dominant_strategies": [], "dominated_strategies": []},
                "game_classification": {"game_type": "Unknown", "strategic_tensions": []},
                "recommended_strategies": []
            }
    
    def predict_strategy_evolution(self, time_horizon=10):
        """
        Use LLM to predict how strategies will evolve over time
        
        Parameters:
        - time_horizon: Number of time steps to predict into the future
        
        Returns:
        - prediction: Dictionary containing strategy evolution prediction
        """
        # Ensure we have enough history to make predictions
        if len(self.strategy_distribution_history) < 5:
            return {"error": "Insufficient strategy history for prediction"}
        
        # Extract recent strategy distribution history
        recent_history = self.strategy_distribution_history[-20:]
        
        # Get current payoff matrix
        current_payoff = self.payoff_matrix_history[-1].tolist() if self.payoff_matrix_history else None
        
        # Create cache key
        cache_key = f"evolution_pred_{hash(str(recent_history))}_{time_horizon}"
        
        # Check cache first
        if cache_key in self.analysis_cache:
            self.logger.debug(f"Using cached strategy evolution prediction")
            return self.analysis_cache[cache_key]
        
        # Construct prompt
        prompt = f"""
        Predict how strategies will evolve over the next {time_horizon} time steps based on:
        
        Recent Strategy Distribution History:
        {recent_history}
        
        Current Payoff Matrix:
        {current_payoff}
        
        System Growth Rate: {self.growth_rate}
        
        Provide a detailed prediction addressing:
        
        1. Strategy distribution trajectory:
           - How will each strategy's prevalence change over time?
           - Will any strategies become dominant or extinct?
           - At what time steps will significant changes occur?
        
        2. Equilibrium dynamics:
           - Will the system reach equilibrium within the time horizon?
           - Will there be oscillations or cycles in strategy distributions?
           - What perturbations might disrupt the predicted trajectory?
        
        3. Key inflection points:
           - Identify critical time points where strategic shifts are likely
           - Explain the dynamics driving these shifts
        
        RESPONSE FORMAT:
        ```json
        {{
            "strategy_predictions": [
                {{
                    "time_step": integer,
                    "distribution": [float, ...],  // Predicted strategy distribution
                    "dominant_strategy": integer,  // Index of dominant strategy, if any
                    "confidence": float  // 0.0 to 1.0
                }},
                ...
            ],
            "equilibrium_prediction": {{
                "will_reach_equilibrium": boolean,
                "estimated_time_step": integer,
                "stability": float,  // 0.0 to 1.0
                "description": "string"
            }},
            "inflection_points": [
                {{
                    "time_step": integer,
                    "description": "string",
                    "driving_factors": [
                        "string",
                        ...
                    ]
                }},
                ...
            ],
            "confidence_overall": float  // 0.0 to 1.0
        }}
        ```
        """
        
        # Query LLM
        start_time = time.time()
        response = self._query_llm(prompt)
        elapsed_time = time.time() - start_time
        self.llm_response_times.append(elapsed_time)
        self.llm_call_count += 1
        
        # Parse response
        try:
            prediction = self._extract_json_from_response(response)
            
            # Cache the result
            self.analysis_cache[cache_key] = prediction
            
            return prediction
        except Exception as e:
            self.logger.error(f"Error parsing LLM strategy evolution prediction response: {str(e)}")
            # Return empty prediction as fallback
            return {
                "strategy_predictions": [],
                "equilibrium_prediction": {"will_reach_equilibrium": False, "description": "Prediction failed"},
                "inflection_points": [],
                "confidence_overall": 0.0
            }
    
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
                    "max_tokens": 1500,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"{self.system_prompt}\n\n{prompt}"
                        }
                    ],
                    "temperature": 0.2,
                    "top_p": 0.9
                }
            elif "amazon.titan" in self.model_id:
                # Titan-specific request format
                request_body = {
                    "inputText": f"<system>{self.system_prompt}</system>\n\n{prompt}",
                    "textGenerationConfig": {
                        "maxTokenCount": 1500,
                        "temperature": 0.2,
                        "topP": 0.9
                    }
                }
            else:
                # Generic format for other models
                request_body = {
                    "prompt": f"{self.system_prompt}\n\n{prompt}",
                    "max_tokens": 1500,
                    "temperature": 0.2,
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
            return json.dumps({"error": "Failed to query LLM"})
    
    def _extract_json_from_response(self, response):
        """
        Extract JSON from LLM response
        
        Parameters:
        - response: LLM response text
        
        Returns:
        - parsed: Parsed JSON object
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
            return parsed
            
        except Exception as e:
            self.logger.error(f"Error extracting JSON from LLM response: {str(e)}")
            self.logger.error(f"Response was: {response[:100]}...")
            raise e
            
    def get_best_response(self, strategy):
        """
        Get the best response to a given strategy
        
        Parameters:
        - strategy: Strategy vector
        
        Returns:
        - best_response: Best response strategy vector
        """
        if not self.payoff_matrix_history:
            # Return uniform strategy if no payoff matrix available
            best_response = np.ones(self.n_strategies) / self.n_strategies
            return best_response
            
        # Use the latest payoff matrix
        payoff_matrix = self.payoff_matrix_history[-1]
        
        # Calculate expected payoff for each pure strategy against the given strategy
        expected_payoffs = np.zeros(self.n_strategies)
        for i in range(self.n_strategies):
            for j in range(self.n_strategies):
                expected_payoffs[i] += payoff_matrix[i, j] * strategy[j]
        
        # Find best response (strategy with highest expected payoff)
        best_response_idx = np.argmax(expected_payoffs)
        best_response = np.zeros(self.n_strategies)
        best_response[best_response_idx] = 1.0
        
        return best_response
    
    def get_performance_stats(self):
        """
        Get performance statistics for the LLM-enhanced Nash validator
        
        Parameters:
        - None
        
        Returns:
        - stats: Dictionary of performance statistics
        """
        stats = {
            "llm_call_count": self.llm_call_count,
            "avg_response_time": np.mean(self.llm_response_times) if self.llm_response_times else 0,
            "analysis_cache_size": len(self.analysis_cache),
            "equilibrium_reached_count": sum(self.equilibrium_history) if self.equilibrium_history else 0,
            "current_nash_distance": self.nash_distance_history[-1] if self.nash_distance_history else None
        }
        
        return stats
    def check_equilibrium(self, nash_distance):
        """
        Check if the current nash distance indicates an equilibrium
        
        Parameters:
        - nash_distance: Current Nash distance
        
        Returns:
        - is_equilibrium: Boolean indicating if this is a Nash equilibrium
        """
        # Handle case where nash_distance is a tuple
        if isinstance(nash_distance, tuple):
            nash_distance = nash_distance[0]
            
        is_equilibrium = nash_distance <= self.equilibrium_threshold
        
        if is_equilibrium:
            self.logger.info(f"Nash equilibrium detected with distance {nash_distance:.4f}")
            
        return is_equilibrium
