"""
Agent routes.

Endpoints for triggering maintenance analysis, retrieving ranked
intervention lists, and approving / rejecting recommendations.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastapi import APIRouter, HTTPException, Query
from backend.digital_twin import registry
from backend.agent import agent
from backend.schemas import ApprovalRequest
from ml.simulation import rank_interventions, simulate_intervention
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent", tags=["Agent"])


# ---------------------------------------------------------------------------
# GET /api/agent/analyze
# ---------------------------------------------------------------------------

@router.get("/analyze", summary="Run agent analysis for an engine")
async def analyze_engine(engine_id: str = Query(..., description="Engine identifier")):
    """
    Trigger a full maintenance analysis for the specified engine.

    The agent:
    1. Reads the current twin state.
    2. Identifies the problem and degradation indicators.
    3. Ranks all available maintenance interventions using simulation.
    4. Applies safety constraints.
    5. Generates a natural-language explanation and reasoning steps.
    6. Stores the result as the engine's pending recommendation.

    Returns the full AgentAnalysisResponse dict.
    """
    twin = registry.get(engine_id)
    if twin is None:
        # Auto-create with default state so the API is convenient to call
        twin = registry.get_or_create(engine_id)

    twin_state = twin.to_dict()

    try:
        analysis = agent.analyze(twin_state)
    except Exception as exc:
        logger.exception("Agent analysis failed for engine %s", engine_id)
        raise HTTPException(status_code=500, detail=f"Agent analysis failed: {exc}")

    # Store as pending recommendation on the twin
    twin.pending_recommendation = analysis

    logger.info(
        "Agent analysis complete for engine %s — recommendation: %s",
        engine_id,
        analysis.get("recommended_intervention_id"),
    )
    return analysis


# ---------------------------------------------------------------------------
# GET /api/agent/interventions
# ---------------------------------------------------------------------------

@router.get("/interventions", summary="Get ranked intervention list for an engine")
async def get_interventions(engine_id: str = Query(..., description="Engine identifier")):
    """
    Return all available maintenance interventions ranked by recommendation score,
    based on the current twin state. Does not store a recommendation.
    """
    twin = registry.get_or_create(engine_id)
    state = twin.to_dict()

    rul = float(state.get("predicted_rul") or state.get("reference_rul") or 100.0)
    health_score = float(state.get("health_score", 100.0))
    health_ratio = health_score / 100.0

    try:
        ranked = rank_interventions(rul, health_ratio)
    except Exception as exc:
        logger.exception("rank_interventions failed for engine %s", engine_id)
        raise HTTPException(status_code=500, detail=f"Intervention ranking failed: {exc}")

    return {
        "engine_id": engine_id,
        "current_rul": rul,
        "health_score": health_score,
        "interventions": ranked,
        "count": len(ranked),
    }


# ---------------------------------------------------------------------------
# POST /api/agent/approve
# ---------------------------------------------------------------------------

@router.post("/approve", summary="Approve a pending maintenance recommendation")
async def approve_recommendation(request: ApprovalRequest):
    """
    Engineer approves the pending recommendation for the given engine.

    This triggers the twin to apply the intervention's predicted effects to its
    internal state (restoring health score / RUL as simulated).

    Returns the updated twin state and a confirmation message.
    """
    twin = registry.get(request.engine_id)
    if twin is None:
        raise HTTPException(
            status_code=404,
            detail=f"Engine '{request.engine_id}' not found.",
        )

    # Run simulation to get the effect estimate
    state = twin.to_dict()
    rul = float(state["predicted_rul"])
    health_score = float(state["health_score"])

    try:
        sim_result = simulate_intervention(
            current_rul=rul,
            current_health_score=health_score / 100.0,
            intervention_id=request.intervention_id,
            n_future_cycles=50,
        )
    except Exception as exc:
        logger.exception(
            "Simulation failed during approval for engine %s, intervention %s",
            request.engine_id,
            request.intervention_id,
        )
        raise HTTPException(status_code=500, detail=f"Simulation failed: {exc}")

    twin.apply_intervention(
        intervention_id=request.intervention_id,
        sim_result=sim_result,
        engineer_notes=request.engineer_notes or "",
    )

    logger.info(
        "Intervention '%s' approved for engine %s by engineer.",
        request.intervention_id,
        request.engine_id,
    )

    return {
        "message": (
            f"Intervention '{request.intervention_id}' approved and applied "
            f"to engine '{request.engine_id}'."
        ),
        "rul_before": rul,
        "rul_after": sim_result.get("predicted_rul_after", rul),
        "rul_gain": sim_result.get("rul_gain", 0.0),
        "updated_state": twin.to_dict(),
    }


# ---------------------------------------------------------------------------
# POST /api/agent/reject
# ---------------------------------------------------------------------------

@router.post("/reject", summary="Reject a pending maintenance recommendation")
async def reject_recommendation(request: ApprovalRequest):
    """
    Engineer rejects the pending recommendation for the given engine.

    The recommendation is marked as rejected. No changes are made to the
    twin's health or RUL state.

    Returns the current twin state.
    """
    twin = registry.get(request.engine_id)
    if twin is None:
        raise HTTPException(
            status_code=404,
            detail=f"Engine '{request.engine_id}' not found.",
        )

    twin.reject_recommendation(engineer_notes=request.engineer_notes or "")

    logger.info(
        "Recommendation rejected for engine %s. Notes: %s",
        request.engine_id,
        request.engineer_notes,
    )

    return {
        "message": f"Recommendation rejected for engine '{request.engine_id}'.",
        "engineer_notes": request.engineer_notes,
        "current_state": twin.to_dict(),
    }
