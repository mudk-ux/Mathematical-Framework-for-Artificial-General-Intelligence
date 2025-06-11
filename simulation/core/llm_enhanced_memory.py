#!/usr/bin/env python3
"""
LLM-Enhanced Memory System for the unified MMAI system

This module implements an enhanced memory system that leverages Amazon Bedrock
for improved memory retrieval, synthesis, and pattern recognition.
"""

import numpy as np
import logging
import json
import boto3
import time
from core.memory_system import MemorySystem

class LLMEnhancedMemorySystem(MemorySystem):
    """
    Memory system with LLM-enhanced retrieval and synthesis capabilities
    
    This system extends the base MemorySystem with LLM-powered capabilities for:
    - More sophisticated memory retrieval based on semantic understanding
    - Memory synthesis to extract patterns and insights
    - Anomaly detection in memory patterns
    """
    def __init__(self, capacity=1000, n_frames=10, 
                 model_id="anthropic.claude-3-sonnet-20240229-v1:0", 
                 region="us-east-1",
                 system_prompt=None,
                 logger=None):
        """
        Initialize the LLM-enhanced memory system
        
        Parameters:
        - capacity: Maximum number of memories to store
        - n_frames: Number of memory frames
        - model_id: Amazon Bedrock model ID
        - region: AWS region for Bedrock
        - system_prompt: System prompt for the LLM
        - logger: Optional logger instance
        """
        super().__init__(logger=logger)
        
        # Initialize memory storage
        self.capacity = capacity
        self.n_frames = n_frames
        self.memories = []
        self.frames = {}  # For compatibility with parent class
        
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
                "You are an advanced memory system for a multi-agent simulation. "
                "Your role is to analyze memories, identify patterns, and extract insights "
                "that can help agents make better decisions. Always provide your analysis "
                "in a structured JSON format as specified in each prompt."
            )
        else:
            self.system_prompt = system_prompt
            
        # Performance tracking
        self.llm_response_times = []
        self.llm_call_count = 0
        
        # Cache for memory retrievals and syntheses
        self.retrieval_cache = {}
        self.synthesis_cache = {}
        
        self.logger.info(f"Initialized LLM-Enhanced Memory System with model {model_id}")
    
    def add_memory(self, memory):
        """
        Add a memory to the system
        
        Parameters:
        - memory: Memory object to add
        
        Returns:
        - memory_id: ID of the added memory
        """
        # Add timestamp if not present
        if 'timestamp' not in memory:
            memory['timestamp'] = self.current_time
            
        # Add unique ID if not present
        if 'id' not in memory:
            memory['id'] = len(self.memories)
            
        # Add memory to list
        self.memories.append(memory)
        
        # Keep memory list within capacity
        if len(self.memories) > self.capacity:
            self.memories = self.memories[-self.capacity:]
            
        # Also store in appropriate frame if applicable
        if 'agent_id' in memory:
            self.store_experience(memory['agent_id'], memory, memory.get('outcome', 0.5))
            
        return memory['id']
    
    def retrieve_memories(self, query_state, k=5):
        """
        Basic memory retrieval based on vector similarity
        
        Parameters:
        - query_state: State to query against
        - k: Number of memories to retrieve
        
        Returns:
        - memories: List of relevant memories
        """
        if not self.memories:
            return []
        
        # Simple implementation: return most recent memories
        # In a real implementation, this would use vector similarity
        return sorted(self.memories, key=lambda m: m.get('timestamp', 0), reverse=True)[:k]
    
    def retrieve_relevant_memories(self, query_state, k=5):
        """
        Retrieve memories relevant to the current state using LLM reasoning
        
        Parameters:
        - query_state: State to query against
        - k: Number of memories to retrieve
        
        Returns:
        - memories: List of relevant memories with reasoning
        """
        # Create cache key
        cache_key = self._create_cache_key(query_state, k)
        
        # Check cache first
        if cache_key in self.retrieval_cache:
            self.logger.debug(f"Using cached memory retrieval for {cache_key[:30]}...")
            return self.retrieval_cache[cache_key]
        
        # Get initial candidates using vector similarity
        candidates = self.retrieve_memories(query_state, k=k*2)
        
        if not candidates:
            return []
        
        # Use LLM to refine and rank candidates
        prompt = f"""
        Given the current state:
        {json.dumps(query_state, default=str)}
        
        And these candidate memories:
        {json.dumps(candidates, default=str)}
        
        Analyze these memories and:
        1. Rank them by relevance to the current state
        2. Explain why each is relevant or not relevant
        3. Return only the {k} most relevant memories
        
        RESPONSE FORMAT:
        ```json
        {{
            "relevant_memories": [
                {{
                    "memory": {{...}},  // The original memory object
                    "relevance_score": float,  // 0.0 to 1.0
                    "reasoning": "string"  // Why this memory is relevant
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
            refined_memories = self._parse_memory_response(response)
            
            # Cache the result
            self.retrieval_cache[cache_key] = refined_memories
            
            return refined_memories
        except Exception as e:
            self.logger.error(f"Error parsing LLM memory retrieval response: {str(e)}")
            # Fall back to standard retrieval
            return candidates[:k]
    
    def synthesize_memories(self, memories, context=None):
        """
        Use LLM to synthesize insights from multiple memories
        
        Parameters:
        - memories: List of memories to synthesize
        - context: Optional context to guide synthesis
        
        Returns:
        - synthesis: Dictionary containing synthesized insights
        """
        if not memories:
            return {"insights": [], "patterns": [], "anomalies": []}
        
        # Create cache key
        memory_ids = [str(m.get('id', hash(str(m)))) for m in memories]
        cache_key = f"synthesis_{'-'.join(memory_ids)}"
        if context:
            cache_key += f"_ctx_{hash(str(context))}"
        
        # Check cache first
        if cache_key in self.synthesis_cache:
            self.logger.debug(f"Using cached memory synthesis")
            return self.synthesis_cache[cache_key]
        
        # Construct prompt
        context_str = f"\nContext:\n{json.dumps(context, default=str)}" if context else ""
        
        prompt = f"""
        Synthesize the following memories into key insights:{context_str}
        
        Memories:
        {json.dumps(memories, default=str)}
        
        Analyze these memories to:
        1. Identify key insights that could inform decision-making
        2. Detect patterns across memories (temporal, spatial, strategic)
        3. Highlight any anomalies or contradictions
        4. Suggest potential strategic implications
        
        RESPONSE FORMAT:
        ```json
        {{
            "insights": [
                {{
                    "description": "string",
                    "confidence": float,  // 0.0 to 1.0
                    "supporting_evidence": ["memory reference", ...]
                }},
                ...
            ],
            "patterns": [
                {{
                    "pattern_type": "string",  // e.g., "temporal", "spatial", "strategic"
                    "description": "string",
                    "strength": float  // 0.0 to 1.0
                }},
                ...
            ],
            "anomalies": [
                {{
                    "description": "string",
                    "severity": float,  // 0.0 to 1.0
                    "affected_memories": ["memory reference", ...]
                }},
                ...
            ],
            "strategic_implications": [
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
            synthesis = self._parse_synthesis_response(response)
            
            # Cache the result
            self.synthesis_cache[cache_key] = synthesis
            
            return synthesis
        except Exception as e:
            self.logger.error(f"Error parsing LLM memory synthesis response: {str(e)}")
            # Return empty synthesis as fallback
            return {"insights": [], "patterns": [], "anomalies": [], "strategic_implications": []}
    
    def detect_memory_anomalies(self, recent_memories, historical_context=None):
        """
        Use LLM to detect anomalies in recent memories compared to historical context
        
        Parameters:
        - recent_memories: Recent memories to analyze
        - historical_context: Optional historical memories for context
        
        Returns:
        - anomalies: List of detected anomalies
        """
        if not recent_memories:
            return []
        
        # Construct prompt
        historical_str = ""
        if historical_context:
            historical_str = f"\nHistorical Context:\n{json.dumps(historical_context, default=str)}"
        
        prompt = f"""
        Analyze these recent memories for anomalies:{historical_str}
        
        Recent Memories:
        {json.dumps(recent_memories, default=str)}
        
        Detect any anomalies, which could include:
        1. Unexpected changes in agent behavior
        2. Unusual patterns in the environment
        3. Contradictions with historical patterns
        4. Statistical outliers in any metrics
        5. Sudden shifts in strategic equilibria
        
        RESPONSE FORMAT:
        ```json
        {{
            "anomalies": [
                {{
                    "type": "string",  // e.g., "behavioral", "environmental", "statistical"
                    "description": "string",
                    "severity": float,  // 0.0 to 1.0
                    "confidence": float,  // 0.0 to 1.0
                    "affected_entities": ["string", ...],
                    "potential_causes": ["string", ...],
                    "recommended_actions": ["string", ...]
                }},
                ...
            ]
        }}
        ```
        """
        
        # Query LLM
        response = self._query_llm(prompt)
        
        # Parse response
        try:
            parsed = self._extract_json_from_response(response)
            return parsed.get("anomalies", [])
        except Exception as e:
            self.logger.error(f"Error parsing LLM anomaly detection response: {str(e)}")
            return []
    
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
                            "role": "system",
                            "content": self.system_prompt
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
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
    
    def _parse_memory_response(self, response):
        """
        Parse LLM response for memory retrieval
        
        Parameters:
        - response: LLM response text
        
        Returns:
        - memories: List of relevant memories with reasoning
        """
        parsed = self._extract_json_from_response(response)
        
        # Extract relevant memories
        if "relevant_memories" in parsed:
            return parsed["relevant_memories"]
        else:
            self.logger.warning("LLM response did not contain relevant_memories field")
            return []
    
    def _parse_synthesis_response(self, response):
        """
        Parse LLM response for memory synthesis
        
        Parameters:
        - response: LLM response text
        
        Returns:
        - synthesis: Dictionary containing synthesized insights
        """
        parsed = self._extract_json_from_response(response)
        
        # Ensure all expected fields are present
        synthesis = {
            "insights": parsed.get("insights", []),
            "patterns": parsed.get("patterns", []),
            "anomalies": parsed.get("anomalies", []),
            "strategic_implications": parsed.get("strategic_implications", [])
        }
        
        return synthesis
    
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
    
    def _create_cache_key(self, query_state, k):
        """
        Create a cache key for memory retrieval
        
        Parameters:
        - query_state: State to query against
        - k: Number of memories to retrieve
        
        Returns:
        - key: Cache key string
        """
        # For simple states, use the string representation
        if isinstance(query_state, (str, int, float, bool)):
            return f"query_{str(query_state)}_{k}"
        
        # For dictionaries, use key-value pairs
        if isinstance(query_state, dict):
            key_parts = []
            for key in sorted(query_state.keys()):
                value = query_state[key]
                if isinstance(value, (list, np.ndarray)) and len(value) > 5:
                    # For long arrays, use hash of the array
                    value_str = f"array_{hash(str(value))}"
                else:
                    value_str = str(value)
                key_parts.append(f"{key}:{value_str}")
            return f"query_{'|'.join(key_parts)}_{k}"
        
        # For other types, use hash of string representation
        return f"query_{hash(str(query_state))}_{k}"
    
    def get_performance_stats(self):
        """
        Get performance statistics for the LLM-enhanced memory system
        
        Returns:
        - stats: Dictionary of performance statistics
        """
        stats = {
            "llm_call_count": self.llm_call_count,
            "avg_response_time": np.mean(self.llm_response_times) if self.llm_response_times else 0,
            "retrieval_cache_size": len(self.retrieval_cache),
            "synthesis_cache_size": len(self.synthesis_cache),
            "memory_count": len(self.memories),
            "frame_count": len(self.collective_memory) + sum(len(frames) for frames in self.individual_memory.values() if frames)
        }
        
        return stats
