"""
LLM Service for integrating Nemotron 49B with the Glid Surge dashboard.
Provides route optimization insights, surge predictions, and agentic capabilities.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover
    httpx = None  # type: ignore

# LLM Server configuration
# NIM uses port 8000, vLLM uses port 5000
# Default to NIM port for production demos
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "nemotron-49b")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120.0"))


class Message(BaseModel):
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: float = Field(0.6, ge=0.0, le=2.0)
    max_tokens: int = Field(4096, ge=1, le=32768)
    stream: bool = Field(False)
    tools: Optional[List[Dict[str, Any]]] = None


class ChatResponse(BaseModel):
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, int]] = None
    reasoning: Optional[str] = None


# System prompts for different modes
SYSTEM_PROMPTS = {
    "route_optimizer": """You are an expert logistics route optimizer for the Glid Surge platform.
You analyze multi-modal routes (road + rail) and provide actionable insights.

When given route data, analyze:
- Total distance and estimated time
- Cost efficiency comparisons
- Optimal departure windows based on surge predictions
- Mode transition points (road ↔ rail) efficiency
- Weather and disruption risk factors

Provide concise, data-driven recommendations. Use the available tools when you need real-time data.""",

    "surge_analyst": """You are a surge prediction analyst for the Glid Surge platform.
You interpret GNN model predictions and explain surge patterns at ports.

When analyzing surge data:
- Explain the predicted surge levels (low/medium/high/critical)
- Identify contributing factors (vessel arrivals, weather, holidays)
- Recommend optimal dispatch windows to avoid congestion
- Estimate cost savings from following recommendations

Be specific with numbers and timeframes. Use reasoning to show your analysis.""",

    "assistant": """You are an AI assistant for the Glid Surge logistics optimization platform.
You help users with:
- Route planning and optimization
- Understanding surge predictions
- Interpreting cost savings and recommendations
- General logistics and supply chain questions

You have access to real-time route and surge data through tools. Always reason through complex queries.""",
}


def _require_httpx() -> None:
    if httpx is None:
        raise RuntimeError(
            "LLM support requires the optional dependency 'httpx'. "
            "Install it in the API container (pip install httpx) or disable LLM routes."
        )


# Tool definitions for agentic capabilities
ROUTE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "compute_route",
            "description": "Compute an optimized multi-modal route between two points",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Origin coordinates [longitude, latitude]"
                    },
                    "destination": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Destination coordinates [longitude, latitude]"
                    },
                    "optimize_for": {
                        "type": "string",
                        "enum": ["time", "distance", "cost"],
                        "description": "Optimization priority"
                    }
                },
                "required": ["origin", "destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_surge_prediction",
            "description": "Get surge predictions for a specific port",
            "parameters": {
                "type": "object",
                "properties": {
                    "port_name": {
                        "type": "string",
                        "description": "Name of the port (e.g., 'Long Beach', 'Los Angeles')"
                    },
                    "horizon_hours": {
                        "type": "integer",
                        "enum": [24, 48, 72],
                        "description": "Prediction horizon in hours"
                    }
                },
                "required": ["port_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_routes",
            "description": "Compare multiple route options and recommend the best one",
            "parameters": {
                "type": "object",
                "properties": {
                    "route_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of route IDs to compare"
                    },
                    "criteria": {
                        "type": "string",
                        "enum": ["fastest", "cheapest", "most_reliable"],
                        "description": "Comparison criteria"
                    }
                },
                "required": ["route_ids"]
            }
        }
    }
]


class LLMService:
    """Service for interacting with the Nemotron 49B LLM."""
    
    def __init__(self, base_url: str = LLM_BASE_URL, model: str = LLM_MODEL):
        _require_httpx()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.AsyncClient(timeout=LLM_TIMEOUT)
    
    async def health_check(self) -> bool:
        """Check if the LLM server is healthy."""
        try:
            response = await self.client.get(f"{self.base_url}/models")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"LLM health check failed: {e}")
            return False
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        mode: str = "assistant",
        temperature: float = 0.6,
        max_tokens: int = 4096,
        enable_tools: bool = False,
        stream: bool = False,
    ) -> ChatResponse:
        """
        Send a chat completion request to the LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            mode: One of 'assistant', 'route_optimizer', 'surge_analyst'
            temperature: Sampling temperature (0.6 recommended for reasoning)
            max_tokens: Maximum tokens to generate
            enable_tools: Whether to enable tool calling
            stream: Whether to stream the response
        
        Returns:
            ChatResponse with content and optional tool calls
        """
        # Prepend system prompt based on mode
        system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["assistant"])
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        
        payload = {
            "model": self.model,
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "top_p": 0.95,  # Recommended for reasoning mode
        }
        
        if enable_tools:
            payload["tools"] = ROUTE_TOOLS
        
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            
            choice = data["choices"][0]
            message = choice["message"]
            
            # Extract reasoning from <think> tags if present
            content = message.get("content", "")
            reasoning = None
            if "<think>" in content and "</think>" in content:
                start = content.find("<think>") + 7
                end = content.find("</think>")
                reasoning = content[start:end].strip()
                content = content[end + 8:].strip()
            
            return ChatResponse(
                content=content,
                tool_calls=message.get("tool_calls"),
                usage=data.get("usage"),
                reasoning=reasoning,
            )
        
        except httpx.HTTPStatusError as e:
            logger.error(f"LLM request failed: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"LLM request error: {e}")
            raise
    
    async def analyze_route(
        self,
        route_data: Dict[str, Any],
        surge_predictions: Optional[Dict[str, Any]] = None,
    ) -> ChatResponse:
        """
        Analyze a route and provide optimization insights.
        
        Args:
            route_data: Route data including segments, distances, times
            surge_predictions: Optional surge prediction data for relevant ports
        
        Returns:
            ChatResponse with analysis and recommendations
        """
        context = f"""Analyze this route and provide optimization recommendations:

Route Data:
{json.dumps(route_data, indent=2)}
"""
        if surge_predictions:
            context += f"""
Surge Predictions:
{json.dumps(surge_predictions, indent=2)}
"""
        
        return await self.chat(
            messages=[{"role": "user", "content": context}],
            mode="route_optimizer",
            temperature=0.6,
        )
    
    async def explain_surge(
        self,
        port_name: str,
        predictions: Dict[str, Any],
    ) -> ChatResponse:
        """
        Explain surge predictions for a port in natural language.
        
        Args:
            port_name: Name of the port
            predictions: Surge prediction data from the GNN model
        
        Returns:
            ChatResponse with explanation and recommendations
        """
        prompt = f"""Explain the surge predictions for {port_name} and provide dispatch recommendations:

Prediction Data:
{json.dumps(predictions, indent=2)}

Please:
1. Summarize the predicted surge levels
2. Identify the best dispatch windows
3. Estimate potential cost savings
4. Note any risk factors to consider
"""
        return await self.chat(
            messages=[{"role": "user", "content": prompt}],
            mode="surge_analyst",
            temperature=0.6,
        )
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# Singleton instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create the LLM service singleton."""
    global _llm_service
    if _llm_service is None:
        _require_httpx()
        _llm_service = LLMService()
    return _llm_service

