"""
Agentic Decision Layer for the Digital Twin system.

The MaintenanceAgent analyses the current engine state, ranks available
maintenance interventions, applies safety constraints, and produces a
structured recommendation ready for engineer approval.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    EMERGENCY_RUL_THRESHOLD,
    CRITICAL_RUL_THRESHOLD,
    WARNING_RUL_THRESHOLD,
)

from ml.simulation import rank_interventions
from datetime import datetime
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MaintenanceAgent
# ---------------------------------------------------------------------------

class MaintenanceAgent:
    """
    Rule-based + heuristic maintenance recommendation agent.

    Analyses digital twin state, applies safety constraints, ranks
    interventions via the simulation layer, and returns a structured
    analysis dictionary compatible with AgentAnalysisResponse.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, twin_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a full maintenance analysis for one engine.

        Parameters
        ----------
        twin_state:
            Dict returned by DigitalTwin.to_dict().

        Returns
        -------
        Dict matching AgentAnalysisResponse schema.
        """
        engine_id = twin_state["engine_id"]
        predicted_rul = float(twin_state.get("predicted_rul") or twin_state.get("reference_rul") or 100.0)
        health_state = twin_state.get("health_state", "Healthy")
        health_score = float(twin_state.get("health_score", 100.0))
        degradation_rate = float(twin_state.get("degradation_rate", 0.0))
        is_degradation_accelerating = bool(twin_state.get("is_degradation_accelerating", False))
        risk_level = twin_state.get("risk_level", "Low")
        sensor_readings = twin_state.get("sensor_readings", {})
        current_cycle = int(twin_state.get("current_cycle", 0))

        logger.info(
            "Agent analysing engine %s — RUL=%.1f, state=%s",
            engine_id,
            predicted_rul,
            health_state,
        )

        # ── Step 1: Identify problem ─────────────────────────────────────
        problem = self._identify_problem(
            predicted_rul, health_state, is_degradation_accelerating, degradation_rate
        )

        # ── Step 2: Identify degradation indicators ──────────────────────
        indicators = self._identify_degradation_indicators(
            twin_state, predicted_rul, health_state, degradation_rate, is_degradation_accelerating
        )

        # ── Step 3: Rank interventions (with safety filtering) ───────────
        all_ranked, safety_constraints = self._get_ranked_interventions(
            predicted_rul, health_score, health_state
        )

        # ── Step 4: Best intervention ────────────────────────────────────
        best = all_ranked[0] if all_ranked else self._no_action_fallback(predicted_rul)

        # ── Step 5: Explanation & reasoning ─────────────────────────────
        explanation = self.generate_explanation(problem, best, all_ranked, twin_state)
        reasoning_steps = self.generate_reasoning_steps(
            current_cycle, predicted_rul, health_state, health_score,
            degradation_rate, is_degradation_accelerating, problem, best, all_ranked
        )

        return {
            "engine_id": engine_id,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "identified_problem": problem,
            "degradation_indicators": indicators,
            "interventions": all_ranked,
            "recommended_intervention_id": best.get("intervention_id", "no_action"),
            "recommendation_score": round(best.get("recommendation_score", 0.0), 4),
            "explanation": explanation,
            "reasoning_steps": reasoning_steps,
            "safety_constraints_applied": safety_constraints,
            "requires_engineer_approval": True,
            "approval_status": "pending",
        }

    # ------------------------------------------------------------------
    # Problem identification
    # ------------------------------------------------------------------

    def _identify_problem(
        self,
        rul: float,
        health_state: str,
        is_accelerating: bool,
        degradation_rate: float,
    ) -> str:
        """Return a one-line problem description based on engine state."""
        if rul <= EMERGENCY_RUL_THRESHOLD:
            return (
                f"CRITICAL: Imminent failure risk detected — only {rul:.0f} cycles "
                "of useful life remain. Immediate intervention required."
            )
        if health_state == "Critical":
            return (
                "Significant engine degradation detected. "
                f"RUL has dropped to {rul:.0f} cycles with critical wear indicators present."
            )
        if is_accelerating:
            return (
                f"Accelerating degradation pattern detected — degradation rate is "
                f"{degradation_rate:.2f} cycles/cycle, exceeding safe thresholds."
            )
        if health_state == "Warning":
            return (
                f"Moderate engine wear detected. "
                f"Approximately {rul:.0f} cycles of useful life remain; "
                "preventive maintenance should be planned."
            )
        return (
            f"Engine within normal operating parameters. "
            f"Estimated {rul:.0f} cycles of useful life remaining."
        )

    # ------------------------------------------------------------------
    # Degradation indicator extraction
    # ------------------------------------------------------------------

    def _identify_degradation_indicators(
        self,
        twin_state: Dict[str, Any],
        rul: float,
        health_state: str,
        degradation_rate: float,
        is_accelerating: bool,
    ) -> List[str]:
        """Build a list of human-readable degradation indicator strings."""
        indicators: List[str] = []

        if rul <= EMERGENCY_RUL_THRESHOLD:
            indicators.append(f"RUL critically low: {rul:.0f} cycles (threshold: {EMERGENCY_RUL_THRESHOLD})")
        elif rul <= CRITICAL_RUL_THRESHOLD:
            indicators.append(f"RUL below critical threshold: {rul:.0f} cycles (threshold: {CRITICAL_RUL_THRESHOLD})")
        elif rul <= WARNING_RUL_THRESHOLD:
            indicators.append(f"RUL below warning threshold: {rul:.0f} cycles (threshold: {WARNING_RUL_THRESHOLD})")

        if degradation_rate < -2.0:
            indicators.append(
                f"High degradation rate: losing {abs(degradation_rate):.2f} cycles of life per operational cycle"
            )
        elif degradation_rate < -1.0:
            indicators.append(
                f"Moderate degradation rate: {abs(degradation_rate):.2f} cycles/cycle"
            )

        if is_accelerating:
            indicators.append("Degradation is accelerating — rate exceeds –1.5 cycles/cycle")

        # Health-state based indicator
        health_score = float(twin_state.get("health_score", 100.0))
        if health_score < 30:
            indicators.append(f"Health score severely reduced: {health_score:.1f}/100")
        elif health_score < 60:
            indicators.append(f"Health score degraded: {health_score:.1f}/100")

        if not indicators:
            indicators.append("No significant degradation indicators — engine operating normally")

        return indicators

    # ------------------------------------------------------------------
    # Safety constraints & intervention ranking
    # ------------------------------------------------------------------

    def apply_safety_constraints(
        self,
        interventions: List[Dict[str, Any]],
        rul: float,
        health_state: str,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Filter the intervention list according to safety rules.

        Returns
        -------
        (filtered_interventions, list_of_constraint_descriptions)
        """
        constraints: List[str] = []

        # Emergency: must act — remove do-nothing / minor options
        if rul <= EMERGENCY_RUL_THRESHOLD:
            removed_ids = {"no_action", "reduce_load", "fluid_service"}
            before = len(interventions)
            interventions = [i for i in interventions if i["intervention_id"] not in removed_ids]
            after = len(interventions)
            if before != after:
                constraints.append(
                    f"Safety rule: RUL ≤ {EMERGENCY_RUL_THRESHOLD} — minor interventions removed; "
                    "only substantive maintenance options are permitted."
                )

        # Healthy engine with plenty of life: only no-action is appropriate
        if health_state == "Healthy" and rul > 100:
            interventions = [i for i in interventions if i["intervention_id"] == "no_action"]
            constraints.append(
                "Operational policy: engine is healthy with RUL > 100 cycles — "
                "proactive maintenance not yet warranted."
            )

        if not constraints:
            constraints.append("No safety constraints applied — all interventions are available.")

        return interventions, constraints

    def _get_ranked_interventions(
        self,
        rul: float,
        health_score: float,
        health_state: str,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Get all interventions ranked by the simulation layer, then apply
        safety constraints.
        """
        health_ratio = health_score / 100.0
        all_interventions = rank_interventions(rul, health_ratio)

        # Enforce safety rules
        filtered, constraints = self.apply_safety_constraints(
            all_interventions, rul, health_state
        )

        # If filtering removed everything, fall back to full list (safety-net)
        if not filtered:
            filtered = all_interventions
            constraints.append(
                "Safety fallback: all safety filters applied but no interventions remained — "
                "reverting to full ranked list."
            )

        return filtered, constraints

    # ------------------------------------------------------------------
    # Explanation generation
    # ------------------------------------------------------------------

    def generate_explanation(
        self,
        problem: str,
        best_intervention: Dict[str, Any],
        all_interventions: List[Dict[str, Any]],
        twin_state: Dict[str, Any],
    ) -> str:
        """
        Generate a natural-language explanation paragraph for the recommendation.
        """
        rul = float(twin_state.get("predicted_rul", 0))
        n = len(all_interventions)
        best_name = best_intervention.get("intervention_name", "the recommended intervention")
        rul_gain = float(best_intervention.get("rul_gain", 0))
        improvement_pct = float(best_intervention.get("improvement_pct", 0))
        risk = best_intervention.get("risk", "unknown")
        cost = best_intervention.get("cost", "unknown")

        return (
            f"The engine currently has approximately {rul:.0f} cycles of useful life remaining. "
            f"{problem} "
            f"I evaluated {n} possible maintenance intervention(s) using the Digital Twin simulation. "
            f"{best_name} is predicted to provide the best outcome, adding approximately "
            f"{rul_gain:.0f} cycles of useful life ({improvement_pct:.1f}% improvement) "
            f"while maintaining an acceptable risk profile (risk: {risk}, cost: {cost}). "
            "This recommendation is based on virtual simulation estimates — actual effects will "
            "depend on real engine conditions and maintenance execution quality. "
            "Engineer review and approval is required before any action is taken."
        )

    # ------------------------------------------------------------------
    # Reasoning steps
    # ------------------------------------------------------------------

    def generate_reasoning_steps(
        self,
        current_cycle: int,
        predicted_rul: float,
        health_state: str,
        health_score: float,
        degradation_rate: float,
        is_accelerating: bool,
        problem: str,
        best_intervention: Dict[str, Any],
        all_interventions: List[Dict[str, Any]],
    ) -> List[str]:
        """Return a numbered list of reasoning steps for the recommendation."""
        steps: List[str] = []

        steps.append(f"Step 1: Read current engine state — Cycle {current_cycle}, RUL = {predicted_rul:.0f} cycles.")
        steps.append(f"Step 2: Health classification = '{health_state}' (score: {health_score:.1f}/100).")
        steps.append(
            f"Step 3: Degradation rate = {degradation_rate:.2f} cycles/cycle "
            f"({'accelerating' if is_accelerating else 'stable'})."
        )
        steps.append(f"Step 4: Problem identified — {problem}")
        steps.append(
            f"Step 5: Retrieved {len(all_interventions)} candidate intervention(s) from the simulation layer."
        )

        for i, iv in enumerate(all_interventions, start=1):
            steps.append(
                f"Step 6.{i}: Evaluated '{iv.get('intervention_name', iv.get('intervention_id'))}' — "
                f"RUL gain: {iv.get('rul_gain', 0):.0f} cycles, "
                f"score: {iv.get('recommendation_score', 0):.3f}, "
                f"rank: {iv.get('rank', i)}."
            )

        best_name = best_intervention.get("intervention_name", "unknown")
        best_score = best_intervention.get("recommendation_score", 0)
        steps.append(
            f"Step 7: Selected '{best_name}' as the top-ranked intervention "
            f"(score: {best_score:.3f}) after applying safety constraints."
        )
        steps.append(
            "Step 8: Recommendation flagged for engineer review — awaiting approval before execution."
        )

        return steps

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _no_action_fallback(self, rul: float) -> Dict[str, Any]:
        """Return a minimal no-action intervention dict when no ranked results exist."""
        return {
            "intervention_id": "no_action",
            "intervention_name": "No Action",
            "current_rul": rul,
            "predicted_rul_after": rul,
            "rul_gain": 0.0,
            "improvement_pct": 0.0,
            "risk": "Low",
            "cost": "None",
            "recommendation_score": 0.0,
            "rank": 1,
            "is_recommended": True,
            "is_feasible": True,
            "assumptions": ["Engine continues to operate without intervention."],
            "source": "simulated",
            "data_source": "rule-based fallback",
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

agent = MaintenanceAgent()
