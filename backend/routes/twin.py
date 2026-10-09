"""
Digital Twin state routes.

Endpoints for reading and updating the per-engine digital twin state,
accessing sensor/RUL history, resetting a twin, and listing all tracked engines.
"""

from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from backend.digital_twin import registry
from backend.schemas import UpdateSensorRequest, EngineStateResponse
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/twin", tags=["Digital Twin"])


# ---------------------------------------------------------------------------
# GET /api/twin/state
# ---------------------------------------------------------------------------

@router.get("/state", summary="Get current engine twin state")
async def get_twin_state(engine_id: str = Query(..., description="Engine identifier, e.g. 'E1'")):
    """
    Return the full digital twin state snapshot for the given engine.
    Creates the twin with default state if it does not yet exist.
    """
    twin = registry.get_or_create(engine_id)
    return twin.to_dict()


# ---------------------------------------------------------------------------
# POST /api/twin/update
# ---------------------------------------------------------------------------

@router.post("/update", summary="Push new sensor data to the twin")
async def update_twin(request: UpdateSensorRequest):
    """
    Ingest a new sensor observation for the specified engine.
    Triggers RUL prediction and state update; returns the updated state.
    """
    twin = registry.get_or_create(request.engine_id)
    try:
        updated_state = twin.update_with_sensor_data(
            cycle=request.cycle,
            sensor_data=request.sensor_data,
            op_settings=request.op_settings,
        )
    except Exception as exc:
        logger.exception("Failed to update twin for engine %s", request.engine_id)
        raise HTTPException(status_code=500, detail=f"Twin update failed: {exc}")

    return updated_state


# ---------------------------------------------------------------------------
# GET /api/twin/history
# ---------------------------------------------------------------------------

@router.get("/history", summary="Get RUL and sensor history")
async def get_twin_history(
    engine_id: str = Query(..., description="Engine identifier"),
    n_cycles: Optional[int] = Query(None, description="Number of historical cycles to return"),
):
    """
    Return the full RUL history and per-sensor value trends for the given engine.
    Optionally limited to the last n_cycles observations.
    """
    twin = registry.get(engine_id)
    if twin is None:
        raise HTTPException(
            status_code=404,
            detail=f"Engine '{engine_id}' not found. Call /api/twin/state first to initialise it.",
        )
    history: Dict[str, Any] = twin.get_history()

    if n_cycles is not None and n_cycles > 0:
        if "rul_history" in history and isinstance(history["rul_history"], list):
            history["rul_history"] = history["rul_history"][-n_cycles:]

        if "sensor_trends" in history and isinstance(history["sensor_trends"], dict):
            history["sensor_trends"] = {
                sensor: values[-n_cycles:] if isinstance(values, list) else values
                for sensor, values in history["sensor_trends"].items()
            }

        if "cycles" in history and isinstance(history["cycles"], list):
            history["cycles"] = history["cycles"][-n_cycles:]

    return history


# ---------------------------------------------------------------------------
# POST /api/twin/reset
# ---------------------------------------------------------------------------

@router.post("/reset", summary="Reset (delete and re-create) the digital twin")
async def reset_twin(engine_id: str = Query(..., description="Engine identifier to reset")):
    """
    Delete the existing twin state for the specified engine and create a fresh one.
    Returns the new default state.
    """
    registry.delete(engine_id)
    twin = registry.get_or_create(engine_id)
    logger.info("Twin reset for engine %s", engine_id)
    return {
        "message": f"Twin for engine '{engine_id}' has been reset.",
        "state": twin.to_dict(),
    }


# ---------------------------------------------------------------------------
# GET /api/twin/engines
# ---------------------------------------------------------------------------

@router.get("/engines", summary="List all tracked engines")
async def list_engines():
    """Return a list of all engine IDs currently tracked in the registry."""
    engines = registry.list_engines()
    return {"engines": engines, "count": len(engines)}
