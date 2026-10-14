"""
Digital Twin state routes.

Endpoints for reading and updating the per-engine digital twin state,
accessing sensor/RUL history, resetting a twin, and listing all tracked engines.
"""

from fastapi import APIRouter, HTTPException, Query
from backend.digital_twin import registry
from backend.replay import replay_manager
from backend.schemas import UpdateSensorRequest, EngineStateResponse
from datetime import datetime
from typing import Optional
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
    n_cycles: int = Query(default=50, ge=1, le=500, description="Max cycles of history to return"),
):
    """
    Return the full RUL history and per-sensor value trends for the given engine.
    """
    twin = registry.get(engine_id)
    if twin is None:
        twin = registry.get_or_create(engine_id)
    return twin.get_history(n_cycles=n_cycles)


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


# ---------------------------------------------------------------------------
# Replay Control Endpoints
# ---------------------------------------------------------------------------

@router.post("/replay/start", summary="Start progressive replay streaming")
async def replay_start(
    engine_id: str = Query(..., description="Engine identifier, e.g. 'E1'"),
    interval_sec: Optional[float] = Query(default=None, description="Playback speed in seconds per cycle"),
):
    """
    Start continuous progressive replay of NASA test data for this engine.
    Steps forward 1 cycle every interval_sec without blocking callers.
    """
    rep = replay_manager.get_or_create(engine_id)
    status = rep.start(interval_sec=interval_sec)
    twin = registry.get_or_create(engine_id)
    return {
        "status": status,
        "twin_state": twin.to_dict(),
    }


@router.post("/replay/pause", summary="Pause progressive replay")
async def replay_pause(
    engine_id: str = Query(..., description="Engine identifier, e.g. 'E1'"),
):
    """Pause playback of NASA test data streaming."""
    rep = replay_manager.get_or_create(engine_id)
    status = rep.pause()
    twin = registry.get_or_create(engine_id)
    return {
        "status": status,
        "twin_state": twin.to_dict(),
    }


@router.post("/replay/step", summary="Advance replay by exactly 1 cycle")
async def replay_step(
    engine_id: str = Query(..., description="Engine identifier, e.g. 'E1'"),
):
    """
    Advance replay by exactly one operational cycle.
    Updates the twin state with the genuine sensor observation.
    """
    rep = replay_manager.get_or_create(engine_id)
    updated_state = rep.step()
    return {
        "status": rep.get_status(),
        "twin_state": updated_state,
    }


@router.post("/replay/reset", summary="Reset replay cursor and digital twin state")
async def replay_reset(
    engine_id: str = Query(..., description="Engine identifier, e.g. 'E1'"),
):
    """
    Reset replay back to cycle 0 and reset the twin's history,
    while keeping the trained model and preprocessor attached.
    """
    rep = replay_manager.get_or_create(engine_id)
    reset_state = rep.reset()
    return {
        "status": rep.get_status(),
        "twin_state": reset_state,
    }


@router.get("/replay/status", summary="Get current replay status")
async def replay_status(
    engine_id: str = Query(..., description="Engine identifier, e.g. 'E1'"),
):
    """
    Query current replay playback status without advancing the cursor.
    """
    rep = replay_manager.get_or_create(engine_id)
    twin = registry.get_or_create(engine_id)
    return {
        "status": rep.get_status(),
        "twin_state": twin.to_dict(),
    }


@router.post("/replay/speed", summary="Set replay playback interval")
async def replay_speed(
    engine_id: str = Query(..., description="Engine identifier, e.g. 'E1'"),
    interval_sec: float = Query(..., ge=0.1, le=10.0, description="Interval in seconds per cycle"),
):
    """Update replay speed interval."""
    rep = replay_manager.get_or_create(engine_id)
    status = rep.set_speed(interval_sec)
    return {"status": status}

