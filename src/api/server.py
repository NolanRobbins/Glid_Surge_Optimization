from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

from src.config import MODELS_DIR, OUTPUT_DIR, PROJECT_ROOT
from src.forecasting.surge_model import SurgePredictionModel
from src.graph.route_adapter import RouteAdapter
from src.api.llm_service import get_llm_service, ChatRequest, ChatResponse, LLMService

logger = logging.getLogger(__name__)

# Configure logging to help track all agent interactions [[memory:8054604]]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

app = FastAPI(title="Glid Surge + Routing API", version="0.1.0")


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class SurgePredictRequest(BaseModel):
    rows: List[Dict[str, Any]] = Field(..., description="Pre-engineered feature rows")
    horizon: int = Field(24, description="Prediction horizon in hours")

    @validator("rows")
    def _require_rows(cls, value: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not value:
            raise ValueError("rows must not be empty")
        return value


class RouteComputeRequest(BaseModel):
    origin: List[float] = Field(..., min_items=2, max_items=2, description="[lon, lat]")
    destination: List[float] = Field(..., min_items=2, max_items=2, description="[lon, lat]")
    mode: str = Field("auto", description='Mode hint: "rail" | "road" | "auto"')
    optimize_for: str = Field("time", description='"time" or "distance"')

    @validator("origin", "destination")
    def _validate_point(cls, value: List[float]) -> List[float]:
        if len(value) != 2:
            raise ValueError("Point must be [lon, lat]")
        return value

    @property
    def origin_tuple(self) -> tuple[float, float]:
        return float(self.origin[0]), float(self.origin[1])

    @property
    def destination_tuple(self) -> tuple[float, float]:
        return float(self.destination[0]), float(self.destination[1])


# --------------------------------------------------------------------------- #
# Globals
# --------------------------------------------------------------------------- #
route_adapter: Optional[RouteAdapter] = None
surge_model: Optional[SurgePredictionModel] = None
surge_model_error: Optional[str] = None
llm_service: Optional[LLMService] = None
llm_available: bool = False


# --------------------------------------------------------------------------- #
# Startup
# --------------------------------------------------------------------------- #
def load_surge_model() -> SurgePredictionModel:
    """Load a persisted SurgePredictionModel if available."""
    model = SurgePredictionModel()
    # Prefer MODELS_DIR joblib artifact
    joblib_path = MODELS_DIR / "surge_model.joblib"
    if joblib_path.exists():
        model.load(joblib_path)
        return model

    # Fallback to latest checkpoint in output/checkpoints (best effort)
    fallback_path = OUTPUT_DIR / "checkpoints" / "surge_model_latest.pkl"
    if fallback_path.exists():
        try:
            model.load(fallback_path)
            return model
        except Exception:
            logger.warning("Failed to load fallback model at %s", fallback_path)

    raise FileNotFoundError("No surge model artifact found.")


@app.on_event("startup")
async def startup_event() -> None:
    global route_adapter, surge_model, surge_model_error, llm_service, llm_available

    route_adapter = RouteAdapter()
    try:
        route_adapter.build_graph()
        logger.info("Graph built and ready: %s nodes", route_adapter.graph.number_of_nodes())
    except Exception as exc:
        route_adapter = None
        logger.exception("Failed to build routing graph: %s", exc)

    try:
        surge_model = load_surge_model()
        logger.info("Surge model loaded.")
    except Exception as exc:
        surge_model_error = str(exc)
        surge_model = None
        logger.warning("Surge model not loaded: %s", exc)

    # Initialize LLM service (Nemotron 49B)
    try:
        llm_service = get_llm_service()
        llm_available = await llm_service.health_check()
        if llm_available:
            logger.info("LLM service (Nemotron 49B) connected and ready.")
        else:
            logger.warning("LLM service not available - chat features disabled.")
    except Exception as exc:
        llm_available = False
        logger.warning("LLM service initialization failed: %s", exc)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def resolve_route_file(route_id: str) -> Path:
    """Find a saved GeoJSON route by id (without extension)."""
    filename = route_id if route_id.endswith(".json") else f"{route_id}.json"
    search_paths = [
        PROJECT_ROOT,
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "public",
        OUTPUT_DIR,
    ]
    for base in search_paths:
        candidate = base / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Route file '{filename}' not found in known locations.")


def ensure_route_adapter_ready() -> RouteAdapter:
    if route_adapter is None or not route_adapter.is_ready():
        raise HTTPException(status_code=503, detail="Routing graph not ready.")
    return route_adapter


def ensure_surge_model_ready() -> SurgePredictionModel:
    if surge_model is None or not getattr(surge_model, "is_fitted", False):
        detail = "Surge model not loaded."
        if surge_model_error:
            detail = f"{detail} ({surge_model_error})"
        raise HTTPException(status_code=503, detail=detail)
    return surge_model


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> Dict[str, Any]:
    return {
        "graph_ready": route_adapter is not None and route_adapter.is_ready(),
        "surge_model_ready": surge_model is not None and getattr(surge_model, "is_fitted", False),
        "surge_model_error": surge_model_error,
    }


@app.post("/predict/surge")
def predict_surge(payload: SurgePredictRequest) -> Dict[str, Any]:
    model = ensure_surge_model_ready()
    df = pd.DataFrame(payload.rows)

    try:
        result = model.predict(df, horizon=payload.horizon)
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")

    return {
        "predictions": result.predictions.tolist(),
        "horizon_hours": result.prediction_horizon_hours,
        "model": result.model_name,
        "count": len(result.predictions),
    }


@app.post("/routes/compute")
def compute_route(payload: RouteComputeRequest) -> Dict[str, Any]:
    adapter = ensure_route_adapter_ready()

    try:
        result = adapter.compute_route(
            origin=payload.origin_tuple,
            destination=payload.destination_tuple,
            optimize_for=payload.optimize_for,
        )
    except Exception as exc:
        logger.exception("Route computation failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"Route computation failed: {exc}")

    return {
        "route": result["route"],
        "metrics": result["metrics"],
        "path_nodes": result["path_nodes"],
        "origin_links": result["origin_links"],
        "destination_links": result["destination_links"],
    }


@app.get("/routes/{route_id}")
def get_route(route_id: str) -> Dict[str, Any]:
    try:
        path = resolve_route_file(route_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    try:
        with open(path, "r") as f:
            content = json.load(f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read route: {exc}")

    return content


@app.get("/network/rail/lines")
def rail_lines(max_features: int = 2000) -> Dict[str, Any]:
    adapter = ensure_route_adapter_ready()
    try:
        return adapter.rail_lines_geojson(max_features=max_features)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load rail lines: {exc}")


@app.get("/network/rail/nodes")
def rail_nodes(max_features: int = 2000) -> Dict[str, Any]:
    adapter = ensure_route_adapter_ready()
    try:
        return adapter.rail_nodes_geojson(max_features=max_features)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load rail nodes: {exc}")


# --------------------------------------------------------------------------- #
# LLM Routes (Nemotron 49B Integration)
# --------------------------------------------------------------------------- #
class LLMChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    mode: str = Field("assistant", description="Chat mode: assistant, route_optimizer, surge_analyst")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional context data (route, surge predictions)")
    enable_tools: bool = Field(False, description="Enable tool calling for agentic behavior")


class LLMChatResponse(BaseModel):
    response: str
    reasoning: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, int]] = None


@app.get("/llm/status")
async def llm_status() -> Dict[str, Any]:
    """Check LLM service status."""
    is_healthy = False
    if llm_service:
        is_healthy = await llm_service.health_check()
    return {
        "available": llm_available and is_healthy,
        "model": "nemotron-49b",
        "modes": ["assistant", "route_optimizer", "surge_analyst"],
        "features": ["reasoning", "tool_calling", "128k_context"]
    }


@app.post("/llm/chat", response_model=LLMChatResponse)
async def llm_chat(request: LLMChatRequest) -> LLMChatResponse:
    """
    Chat with Nemotron 49B for route optimization and surge analysis.
    
    Modes:
    - assistant: General logistics assistant
    - route_optimizer: Specialized route analysis and optimization
    - surge_analyst: Surge prediction interpretation and recommendations
    """
    if not llm_service or not llm_available:
        raise HTTPException(
            status_code=503,
            detail="LLM service not available. Ensure Nemotron 49B is running on port 5000."
        )
    
    logger.info(f"LLM chat request - mode: {request.mode}, tools: {request.enable_tools}")
    
    messages = [{"role": "user", "content": request.message}]
    
    # Add context if provided
    if request.context:
        context_msg = f"Context data:\n```json\n{json.dumps(request.context, indent=2)}\n```\n\n{request.message}"
        messages = [{"role": "user", "content": context_msg}]
    
    try:
        result = await llm_service.chat(
            messages=messages,
            mode=request.mode,
            enable_tools=request.enable_tools,
        )
        
        logger.info(f"LLM response generated - tokens: {result.usage}")
        
        return LLMChatResponse(
            response=result.content,
            reasoning=result.reasoning,
            tool_calls=result.tool_calls,
            usage=result.usage,
        )
    except Exception as exc:
        logger.exception("LLM chat failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"LLM request failed: {exc}")


@app.post("/llm/analyze-route")
async def analyze_route_with_llm(
    route_data: Dict[str, Any],
    include_surge: bool = True,
) -> Dict[str, Any]:
    """
    Use LLM to analyze a route and provide optimization insights.
    """
    if not llm_service or not llm_available:
        raise HTTPException(status_code=503, detail="LLM service not available.")
    
    # Optionally fetch surge predictions for relevant ports
    surge_predictions = None
    if include_surge and surge_model:
        # Extract port info from route if available
        # This is a simplified example - would need actual port extraction logic
        pass
    
    try:
        result = await llm_service.analyze_route(route_data, surge_predictions)
        return {
            "analysis": result.content,
            "reasoning": result.reasoning,
            "usage": result.usage,
        }
    except Exception as exc:
        logger.exception("Route analysis failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")


# --------------------------------------------------------------------------- #
# Entrypoint for `python -m src.api.server`
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

