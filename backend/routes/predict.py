"""
Prediction routes.

Endpoints for querying the latest RUL prediction, running a prediction
on a provided sensor sequence, and retrieving health-state details.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastapi import APIRouter, HTTPException, Query, Body
from backend.digital_twin import registry
from backend.schemas import UpdateSensorRequest
from config import SELECTED_SENSORS, SEQUENCE_LENGTH
import numpy as np
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/predict", tags=["Prediction"])


# ---------------------------------------------------------------------------
# GET /api/predict/rul
# ---------------------------------------------------------------------------

@router.get("/rul", summary="Get latest RUL prediction for an engine")
async def get_rul(engine_id: str = Query(..., description="Engine identifier")):
    """
    Return the most recent RUL prediction and confidence interval
    from the digital twin state.
    """
    twin = registry.get(engine_id)
    if twin is None:
        raise HTTPException(
            status_code=404,
            detail=f"Engine '{engine_id}' not found. Initialise it via /api/twin/state first.",
        )

    state = twin.to_dict()
    return {
        "engine_id": engine_id,
        "current_cycle": state["current_cycle"],
        "predicted_rul": state["predicted_rul"],
        "rul_lower": state["rul_lower"],
        "rul_upper": state["rul_upper"],
        "uncertainty": state["uncertainty"],
        "health_state": state["health_state"],
        "health_score": state["health_score"],
        "risk_level": state["risk_level"],
        "last_updated": state["last_updated"],
    }


# ---------------------------------------------------------------------------
# POST /api/predict/sequence
# ---------------------------------------------------------------------------

@router.post("/sequence", summary="Run RUL prediction on a provided sensor sequence")
async def predict_from_sequence(
    payload: dict = Body(
        ...,
        example={
            "engine_id": "E1",
            "data": [
                {"s2": 641.8, "s3": 1589.7, "s4": 1400.6, "s7": 554.4, "cycle": 1}
            ],
        },
    )
):
    """
    Accept a list of sensor dictionaries representing a temporal sequence
    and return a prediction based on those readings.

    The twin's internal state is updated with each data point in order.
    Returns the state after processing the full sequence.
    """
    engine_id = payload.get("engine_id")
    data: list = payload.get("data", [])

    if not engine_id:
        raise HTTPException(status_code=422, detail="'engine_id' is required.")
    if not data:
        raise HTTPException(status_code=422, detail="'data' list must not be empty.")

    twin = registry.get_or_create(engine_id)

    try:
        last_state = None
        for i, entry in enumerate(data):
            # Determine cycle number
            cycle = int(entry.get("cycle", i + 1))

            # Split out sensor values vs. operational settings
            sensor_data = {
                k: float(v)
                for k, v in entry.items()
                if k.startswith("s") and k != "setting" and k != "sensor"
            }
            op_settings = {
                k: float(v)
                for k, v in entry.items()
                if k.startswith("setting_") or k.startswith("op_")
            }

            last_state = twin.update_with_sensor_data(
                cycle=cycle,
                sensor_data=sensor_data,
                op_settings=op_settings,
            )
    except Exception as exc:
        logger.exception("Sequence prediction failed for engine %s", engine_id)
        raise HTTPException(status_code=500, detail=f"Sequence prediction failed: {exc}")

    return {
        "engine_id": engine_id,
        "cycles_processed": len(data),
        "prediction": {
            "predicted_rul": last_state["predicted_rul"],
            "rul_lower": last_state["rul_lower"],
            "rul_upper": last_state["rul_upper"],
            "uncertainty": last_state["uncertainty"],
            "health_state": last_state["health_state"],
            "health_score": last_state["health_score"],
            "risk_level": last_state["risk_level"],
        },
        "full_state": last_state,
    }


# ---------------------------------------------------------------------------
# GET /api/predict/health-state
# ---------------------------------------------------------------------------

@router.get("/health-state", summary="Get detailed health-state information")
async def get_health_state(engine_id: str = Query(..., description="Engine identifier")):
    """
    Return extended health state details including degradation rate,
    acceleration flag, intervention history summary, and status message.
    """
    twin = registry.get(engine_id)
    if twin is None:
        raise HTTPException(
            status_code=404,
            detail=f"Engine '{engine_id}' not found.",
        )

    state = twin.to_dict()
    return {
        "engine_id": engine_id,
        "current_cycle": state["current_cycle"],
        "health_state": state["health_state"],
        "health_score": state["health_score"],
        "risk_level": state["risk_level"],
        "degradation_rate": state["degradation_rate"],
        "is_degradation_accelerating": state["is_degradation_accelerating"],
        "predicted_rul": state["predicted_rul"],
        "rul_lower": state["rul_lower"],
        "rul_upper": state["rul_upper"],
        "recent_rul_trend": state["recent_rul_trend"],
        "intervention_count": len(state.get("intervention_history", [])),
        "has_pending_recommendation": state.get("pending_recommendation") is not None,
        "status_message": state["status_message"],
        "last_updated": state["last_updated"],
    }
