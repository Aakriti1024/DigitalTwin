"""
Digital Twin state manager.

Manages per-engine in-memory state, sensor history, RUL predictions,
degradation tracking, and intervention application.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    MAX_RUL,
    SEQUENCE_LENGTH,
    SELECTED_SENSORS,
    EMERGENCY_RUL_THRESHOLD,
    CRITICAL_RUL_THRESHOLD,
    WARNING_RUL_THRESHOLD,
)

from datetime import datetime
from collections import deque
from typing import Optional, Dict, List, Any
import threading
import logging

import numpy as np
import pandas as pd
from ml.preprocess import load_single_engine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DigitalTwin
# ---------------------------------------------------------------------------

class DigitalTwin:
    """
    In-memory digital twin for a single engine unit.

    Tracks sensor history, RUL predictions, health state, degradation rate,
    applied interventions, and pending recommendations.
    """

    def __init__(self, engine_id: str):
        self.engine_id: str = engine_id

        # ── Core state ───────────────────────────────────────────────────────
        self.current_cycle: int = 0
        self.health_state: str = "Healthy"
        self.health_score: float = 100.0
        self.predicted_rul: float = float(MAX_RUL)
        self.rul_lower: float = float(MAX_RUL) * 0.85
        self.rul_upper: float = float(MAX_RUL) * 1.15
        self.uncertainty: float = float(MAX_RUL) * 0.15
        self.degradation_rate: float = 0.0
        self.is_degradation_accelerating: bool = False
        self.risk_level: str = "Low"

        # ── History buffers ──────────────────────────────────────────────────
        # Each entry: {'cycle': int, 'sensors': dict, 'op_settings': dict}
        self.sensor_history: deque = deque(maxlen=200)
        # Each entry: (cycle, rul)
        self.rul_history: deque = deque(maxlen=200)

        # ── Intervention tracking ────────────────────────────────────────────
        self.intervention_history: List[Dict[str, Any]] = []
        self.pending_recommendation: Optional[Dict[str, Any]] = None
        self.approval_status: str = "none"

        # ── Metadata ─────────────────────────────────────────────────────────
        self.last_updated: datetime = datetime.utcnow()

        # ── ML model (loaded lazily) ─────────────────────────────────────────
        self.model = None
        self.scaler = None
        self.km = None
        self.op_scaler = None
        self.feature_cols: List[str] = list(SELECTED_SENSORS)
        self.is_demo_mode: bool = True  # True until a real model is loaded

        # ── Thread safety ────────────────────────────────────────────────────
        self._lock: threading.Lock = threading.Lock()

        # ── Demo-mode synthetic state ────────────────────────────────────────
        self._demo_base_rul: float = 120.0  # starting RUL for synthetic demo

        logger.info("DigitalTwin created for engine %s", engine_id)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, model, scaler, km=None, op_scaler=None) -> None:
        """
        Attach a trained ML model and preprocessor artifacts. Disables demo mode.

        Supports either a preprocessor bundle dict (from load_preprocessor)
        or separate scaler, km, op_scaler objects.
        """
        with self._lock:
            self.model = model
            if isinstance(scaler, dict) and "scaler" in scaler:
                self.scaler = scaler["scaler"]
                self.km = scaler.get("km", km)
                self.op_scaler = scaler.get("op_scaler", op_scaler)
                self.feature_cols = list(scaler.get("feature_cols", SELECTED_SENSORS))
            else:
                self.scaler = scaler
                self.km = km
                self.op_scaler = op_scaler
                self.feature_cols = list(SELECTED_SENSORS)
            self.is_demo_mode = False
            logger.info("Model loaded for engine %s", self.engine_id)

    # ------------------------------------------------------------------
    # Sensor update & prediction
    # ------------------------------------------------------------------

    def update_with_sensor_data(
        self,
        cycle: int,
        sensor_data: Dict[str, float],
        op_settings: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a new sensor observation, update internal state, and return
        the full state dictionary.

        Parameters
        ----------
        cycle:
            Current operational cycle number.
        sensor_data:
            Mapping of sensor names to float values.
        op_settings:
            Optional operational settings for this cycle.

        Returns
        -------
        dict matching EngineStateResponse schema.
        """
        with self._lock:
            op_settings = op_settings or {}

            # Append to sensor history
            self.sensor_history.append(
                {"cycle": cycle, "sensors": sensor_data, "op_settings": op_settings}
            )
            self.current_cycle = cycle

            # ── Prediction ───────────────────────────────────────────────
            if not self.is_demo_mode and self.model is not None:
                self._predict_from_model()
            else:
                self._synthetic_degradation(cycle)

            # ── Append RUL to history ─────────────────────────────────────
            self.rul_history.append((cycle, self.predicted_rul))

            # ── Compute degradation rate (slope over last 5 points) ───────
            self._update_degradation_rate()

            # ── Classify health state ─────────────────────────────────────
            self._classify_health()

            self.last_updated = datetime.utcnow()

            return self.to_dict()

    def _predict_from_model(self) -> None:
        """Run the trained ML model on the current sensor history window."""
        try:
            history = list(self.sensor_history)
            n = len(history)

            if n == 0:
                return

            # Build feature DataFrame from the recent history entries
            window = history[-SEQUENCE_LENGTH:] if n >= SEQUENCE_LENGTH else history
            rows = []
            for entry in window:
                row = {}
                sensors = entry.get("sensors", {})
                for s in self.feature_cols:
                    row[s] = float(sensors.get(s, 0.0))
                # Operational settings for clustering (if multi-condition)
                op = entry.get("op_settings", {})
                row["op_set1"] = float(op.get("op_set1", op.get("setting_1", op.get("setting1", 0.0))))
                row["op_set2"] = float(op.get("op_set2", op.get("setting_2", op.get("setting2", 0.0))))
                row["op_set3"] = float(op.get("op_set3", op.get("setting_3", op.get("setting3", 0.0))))
                row["cycle"] = entry.get("cycle", 1)
                rows.append(row)

            df = pd.DataFrame(rows)

            preprocessor = {
                "scaler": self.scaler,
                "km": self.km,
                "op_scaler": self.op_scaler,
                "feature_cols": self.feature_cols,
            }
            # Preprocess window into model input shape (1, SEQUENCE_LENGTH, 14)
            X = load_single_engine(
                engine_df=df,
                scaler=preprocessor,
                seq_len=SEQUENCE_LENGTH,
                feature_cols=self.feature_cols,
            )

            # Predict RUL with uncertainty
            try:
                from ml.predict import predict_rul
                pred_res = predict_rul(self.model, X)
                rul_pred = float(pred_res["predicted_rul"])
                unc = float(pred_res["uncertainty"])
                rul_lower = float(pred_res["rul_lower"])
                rul_upper = float(pred_res["rul_upper"])
            except Exception:
                pred = self.model.predict(X, verbose=0)
                if hasattr(pred, "__len__") and len(pred[0]) >= 2:
                    rul_pred = float(pred[0][0])
                    unc = float(pred[0][1])
                else:
                    rul_pred = float(np.squeeze(pred))
                    unc = max(1.0, rul_pred * 0.1)
                rul_lower = max(0.0, rul_pred - unc)
                rul_upper = rul_pred + unc

            self.predicted_rul = max(0.0, rul_pred)
            self.uncertainty = max(0.0, unc)
            self.rul_lower = max(0.0, rul_lower)
            self.rul_upper = max(0.0, rul_upper)

        except Exception as exc:
            logger.warning("Model prediction failed for engine %s: %s", self.engine_id, exc)

    def _synthetic_degradation(self, cycle: int) -> None:
        """
        Generate synthetic RUL degradation for demo / no-model mode.

        RUL starts at ~120 and decreases by roughly 1–1.5 cycles per cycle
        with small Gaussian noise.
        """
        rng = np.random.default_rng(seed=cycle + hash(self.engine_id) % 1000)
        decrement = rng.uniform(0.8, 1.5)
        noise = rng.normal(0, 0.5)

        # On first call initialise from the base
        if self.current_cycle <= 1:
            self.predicted_rul = self._demo_base_rul
        else:
            self.predicted_rul = max(0.0, self.predicted_rul - decrement + noise)

        uncertainty = max(2.0, self.predicted_rul * 0.08)
        self.uncertainty = uncertainty
        self.rul_lower = max(0.0, self.predicted_rul - uncertainty)
        self.rul_upper = self.predicted_rul + uncertainty

    def _update_degradation_rate(self) -> None:
        """
        Estimate RUL degradation rate (cycles of life lost per operational cycle)
        using linear regression over the last 5 RUL observations.
        """
        history = list(self.rul_history)
        if len(history) < 2:
            self.degradation_rate = 0.0
            self.is_degradation_accelerating = False
            return

        window = history[-5:]
        cycles = np.array([c for c, _ in window], dtype=float)
        ruls = np.array([r for _, r in window], dtype=float)

        if len(cycles) >= 2:
            coeffs = np.polyfit(cycles, ruls, 1)
            self.degradation_rate = float(coeffs[0])  # slope (negative means degrading)
        else:
            self.degradation_rate = 0.0

        # Acceleration: slope worse than –1.5 cycles/cycle
        self.is_degradation_accelerating = self.degradation_rate < -1.5

    def _classify_health(self) -> None:
        """Update health_state, health_score, and risk_level from predicted_rul."""
        rul = self.predicted_rul

        if rul <= 0:
            self.health_state = "Failure"
            self.health_score = 0.0
            self.risk_level = "Critical"
        elif rul <= EMERGENCY_RUL_THRESHOLD:
            self.health_state = "Critical"
            self.health_score = max(0.0, (rul / EMERGENCY_RUL_THRESHOLD) * 30.0)
            self.risk_level = "Critical"
        elif rul <= CRITICAL_RUL_THRESHOLD:
            self.health_state = "Critical"
            span = CRITICAL_RUL_THRESHOLD - EMERGENCY_RUL_THRESHOLD
            frac = (rul - EMERGENCY_RUL_THRESHOLD) / span
            self.health_score = 30.0 + frac * 20.0
            self.risk_level = "High"
        elif rul <= WARNING_RUL_THRESHOLD:
            self.health_state = "Warning"
            span = WARNING_RUL_THRESHOLD - CRITICAL_RUL_THRESHOLD
            frac = (rul - CRITICAL_RUL_THRESHOLD) / span
            self.health_score = 50.0 + frac * 30.0
            self.risk_level = "Medium"
        else:
            self.health_state = "Healthy"
            frac = min(1.0, rul / float(MAX_RUL))
            self.health_score = 80.0 + frac * 20.0
            self.risk_level = "Low"

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @property
    def status_message(self) -> str:
        """Human-readable status string derived from current state."""
        rul = self.predicted_rul
        hs = self.health_state
        rl = self.risk_level

        if hs == "Failure":
            return "⛔ Engine has reached end-of-life. Immediate replacement required."
        if hs == "Critical" and rl == "Critical":
            return (
                f"🔴 CRITICAL: Only {rul:.0f} cycles remaining. "
                "Emergency maintenance required immediately."
            )
        if hs == "Critical":
            return (
                f"🔴 Critical degradation detected. Approximately {rul:.0f} cycles "
                "of useful life remain. Schedule maintenance urgently."
            )
        if hs == "Warning":
            return (
                f"🟡 Warning: Engine showing signs of wear. "
                f"Estimated {rul:.0f} cycles of useful life remain. "
                "Monitor closely and plan maintenance."
            )
        return (
            f"🟢 Engine operating normally. "
            f"Estimated {rul:.0f} cycles of useful life remaining."
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise full twin state to a dict matching EngineStateResponse schema."""
        # Latest sensor readings
        latest_sensors: Dict[str, float] = {}
        if self.sensor_history:
            latest_sensors = dict(self.sensor_history[-1].get("sensors", {}))

        # Recent RUL trend (last 20 points)
        recent_rul_trend = [r for _, r in list(self.rul_history)[-20:]]

        return {
            "engine_id": self.engine_id,
            "current_cycle": self.current_cycle,
            "health_state": self.health_state,
            "health_score": round(self.health_score, 2),
            "predicted_rul": round(self.predicted_rul, 2),
            "rul_lower": round(self.rul_lower, 2),
            "rul_upper": round(self.rul_upper, 2),
            "uncertainty": round(self.uncertainty, 2),
            "degradation_rate": round(self.degradation_rate, 4),
            "is_degradation_accelerating": self.is_degradation_accelerating,
            "risk_level": self.risk_level,
            "sensor_readings": latest_sensors,
            "recent_rul_trend": [round(r, 2) for r in recent_rul_trend],
            "last_updated": self.last_updated.isoformat(),
            "intervention_history": self.intervention_history,
            "pending_recommendation": self.pending_recommendation,
            "status_message": self.status_message,
        }

    # ------------------------------------------------------------------
    # Intervention management
    # ------------------------------------------------------------------

    def apply_intervention(
        self,
        intervention_id: str,
        sim_result: Dict[str, Any],
        engineer_notes: str = "",
    ) -> None:
        """
        Record an approved intervention and adjust twin state to reflect
        the expected health restoration.

        Parameters
        ----------
        intervention_id:
            The chosen intervention (e.g. 'schedule_maintenance').
        sim_result:
            The simulation result dict (from ml.simulation.simulate_intervention).
        engineer_notes:
            Free-text notes from the approving engineer.
        """
        with self._lock:
            rul_before = self.predicted_rul
            rul_after = sim_result.get("predicted_rul_after", rul_before)
            intervention_name = sim_result.get("intervention_name", intervention_id)

            record = {
                "timestamp": datetime.utcnow().isoformat(),
                "intervention_id": intervention_id,
                "intervention_name": intervention_name,
                "rul_before": round(rul_before, 2),
                "rul_after_predicted": round(rul_after, 2),
                "rul_gain": round(rul_after - rul_before, 2),
                "engineer_notes": engineer_notes,
                "status": "approved",
            }
            self.intervention_history.append(record)

            # Apply health restoration (if the intervention does real work)
            if intervention_id != "no_action":
                self.predicted_rul = max(rul_before, rul_after)
                self.rul_lower = max(0.0, self.predicted_rul - self.uncertainty)
                self.rul_upper = self.predicted_rul + self.uncertainty
                # Re-classify health with updated RUL
                self._classify_health()
                # Append to history so the RUL trend reflects the improvement
                self.rul_history.append((self.current_cycle, self.predicted_rul))

            # Clear pending recommendation
            self.pending_recommendation = None
            self.approval_status = "approved"
            self.last_updated = datetime.utcnow()

            logger.info(
                "Engine %s: intervention '%s' applied. RUL %s → %s",
                self.engine_id,
                intervention_id,
                rul_before,
                self.predicted_rul,
            )

    def reject_recommendation(self, engineer_notes: str = "") -> None:
        """Mark the pending recommendation as rejected."""
        with self._lock:
            if self.pending_recommendation:
                self.pending_recommendation["_rejected"] = True
                self.pending_recommendation["_engineer_notes"] = engineer_notes

            self.approval_status = "rejected"
            self.last_updated = datetime.utcnow()

            logger.info("Engine %s: recommendation rejected. Notes: %s", self.engine_id, engineer_notes)

    # ------------------------------------------------------------------
    # History retrieval
    # ------------------------------------------------------------------

    def get_history(self, n_cycles: int = 50) -> Dict[str, Any]:
        """
        Return RUL history and per-sensor trend data.

        Parameters
        ----------
        n_cycles : int
            Maximum number of recent cycles to return. Defaults to 50.

        Returns
        -------
        dict with keys:
            'rul_history': list of {cycle, rul} dicts
            'sensor_trends': dict of sensor_name → list of {cycle, value} dicts
        """
        with self._lock:
            recent_rul = list(self.rul_history)[-n_cycles:]
            rul_records = [
                {"cycle": c, "rul": round(r, 2)} for c, r in recent_rul
            ]

            recent_sensors = list(self.sensor_history)[-n_cycles:]
            sensor_trends: Dict[str, List[Dict[str, Any]]] = {}
            for entry in recent_sensors:
                cycle = entry["cycle"]
                for sensor_name, value in entry.get("sensors", {}).items():
                    if sensor_name not in sensor_trends:
                        sensor_trends[sensor_name] = []
                    sensor_trends[sensor_name].append(
                        {"cycle": cycle, "value": round(float(value), 4)}
                    )

            return {
                "engine_id": self.engine_id,
                "rul_history": rul_records,
                "sensor_trends": sensor_trends,
            }


# ---------------------------------------------------------------------------
# DigitalTwinRegistry
# ---------------------------------------------------------------------------

class DigitalTwinRegistry:
    """
    Singleton registry that manages multiple DigitalTwin instances,
    keyed by engine_id.
    """

    def __init__(self):
        self._twins: Dict[str, DigitalTwin] = {}
        self._lock: threading.Lock = threading.Lock()

    def get_or_create(self, engine_id: str) -> DigitalTwin:
        """Return existing twin or create a new one."""
        with self._lock:
            if engine_id not in self._twins:
                self._twins[engine_id] = DigitalTwin(engine_id)
                logger.info("Registry: created DigitalTwin for engine %s", engine_id)
            return self._twins[engine_id]

    def get(self, engine_id: str) -> Optional[DigitalTwin]:
        """Return existing twin or None."""
        with self._lock:
            return self._twins.get(engine_id)

    def list_engines(self) -> List[str]:
        """Return sorted list of all tracked engine IDs."""
        with self._lock:
            return sorted(self._twins.keys())

    def delete(self, engine_id: str) -> bool:
        """Remove a twin from the registry. Returns True if it existed."""
        with self._lock:
            if engine_id in self._twins:
                del self._twins[engine_id]
                logger.info("Registry: deleted DigitalTwin for engine %s", engine_id)
                return True
            return False


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

registry = DigitalTwinRegistry()
