"""
Simulation routes.

Endpoints for running what-if intervention simulations and comparing
all available interventions for a given engine.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastapi import APIRouter, HTTPException, Query
from backend.digital_twin import registry
from backend.schemas import SimulationRequest, SimulationResponse
from ml.simulation import simulate_intervention, simulate_all_interventions
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/simulation", tags=["Simulation"])


# ---------------------------------------------------------------------------
# POST /api/simulation/what-if
# ---------------------------------------------------------------------------

@router.post("/what-if", summary="Run a what-if intervention simulation", response_model=SimulationResponse)
async def what_if_simulation(request: SimulationRequest):
    """
    Simulate the effect of a specific maintenance intervention on the
    given engine's digital twin.

    Returns the full trajectory of RUL and health score for both the
    baseline (no-intervention) and post-intervention scenarios.
    """
    twin = registry.get_or_create(request.engine_id)
    state = twin.to_dict()

    current_rul = float(state["predicted_rul"])
    health_score = float(state["health_score"])

    try:
        result = simulate_intervention(
            current_rul=current_rul,
            current_health_score=health_score / 100.0,
            intervention_id=request.intervention_id,
            n_future_cycles=request.n_future_cycles,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception(
            "What-if simulation failed for engine %s, intervention %s",
            request.engine_id,
            request.intervention_id,
        )
        raise HTTPException(status_code=500, detail=f"Simulation failed: {exc}")

    # Inject engine_id into result
    result["engine_id"] = request.engine_id

    logger.info(
        "What-if simulation complete: engine=%s, intervention=%s, RUL gain=%.1f",
        request.engine_id,
        request.intervention_id,
        result.get("rul_gain", 0.0),
    )
    return result


# ---------------------------------------------------------------------------
# GET /api/simulation/compare
# ---------------------------------------------------------------------------

@router.get("/compare", summary="Compare all interventions for an engine")
async def compare_interventions(
    engine_id: str = Query(..., description="Engine identifier"),
    n_future_cycles: int = Query(default=50, ge=1, le=500, description="Simulation horizon in cycles"),
):
    """
    Simulate all available maintenance interventions for the given engine
    and return the full comparison data.

    Useful for building comparison charts in the frontend.
    Each entry in the returned list contains full trajectory arrays
    (baseline vs. intervention) alongside summary statistics.
    """
    twin = registry.get_or_create(engine_id)
    state = twin.to_dict()

    current_rul = float(state["predicted_rul"])
    health_score = float(state["health_score"])

    try:
        all_results = simulate_all_interventions(
            current_rul=current_rul,
            current_health_score=health_score / 100.0,
            n_future_cycles=n_future_cycles,
        )
    except Exception as exc:
        logger.exception("Comparison simulation failed for engine %s", engine_id)
        raise HTTPException(status_code=500, detail=f"Comparison simulation failed: {exc}")

    # Attach engine context to each result
    for r in all_results:
        r["engine_id"] = engine_id

    return {
        "engine_id": engine_id,
        "current_rul": current_rul,
        "health_score": health_score,
        "n_future_cycles": n_future_cycles,
        "interventions": all_results,
        "count": len(all_results),
    }


# ---------------------------------------------------------------------------
# GET /api/simulation/interventions
# ---------------------------------------------------------------------------

@router.get("/interventions", summary="Summarised intervention options for an engine")
async def list_simulation_interventions(
    engine_id: str = Query(..., description="Engine identifier"),
):
    """
    Return a compact summary of all intervention simulation results — without
    the full trajectory arrays — suitable for populating a selection list
    or summary table in the frontend.
    """
    twin = registry.get_or_create(engine_id)
    state = twin.to_dict()

    current_rul = float(state["predicted_rul"])
    health_score = float(state["health_score"])

    try:
        all_results = simulate_all_interventions(
            current_rul=current_rul,
            current_health_score=health_score / 100.0,
            n_future_cycles=50,
        )
    except Exception as exc:
        logger.exception("Intervention listing failed for engine %s", engine_id)
        raise HTTPException(status_code=500, detail=f"Simulation failed: {exc}")

    # Strip trajectory arrays to keep the response compact
    summary_keys = {
        "intervention_id",
        "intervention_name",
        "current_rul",
        "predicted_rul_after",
        "rul_gain",
        "improvement_pct",
        "risk",
        "cost",
        "recommendation_score",
        "rank",
        "is_recommended",
        "is_feasible",
        "assumptions",
        "source",
        "data_source",
    }
    summaries = [
        {k: v for k, v in r.items() if k in summary_keys}
        for r in all_results
    ]

    return {
        "engine_id": engine_id,
        "current_rul": current_rul,
        "health_score": health_score,
        "interventions": summaries,
        "count": len(summaries),
    }
