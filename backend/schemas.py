"""
Pydantic v2 schemas for the Digital Twin FastAPI backend.

Defines all request/response models used across the API.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class HealthState(str, Enum):
    """Categorical health classification of an engine."""
    Healthy = "Healthy"
    Warning = "Warning"
    Critical = "Critical"
    Failure = "Failure"


class RiskLevel(str, Enum):
    """Operational risk level associated with continuing to run an engine."""
    Low = "Low"
    Medium = "Medium"
    High = "High"
    Critical = "Critical"


class InterventionType(str, Enum):
    """Available maintenance intervention actions."""
    no_action = "no_action"
    reduce_load = "reduce_load"
    fluid_service = "fluid_service"
    schedule_maintenance = "schedule_maintenance"
    component_replacement = "component_replacement"


# ---------------------------------------------------------------------------
# Sensor / Input Models
# ---------------------------------------------------------------------------

class SensorReading(BaseModel):
    """A single timestep of sensor and operational data for an engine."""
    engine_id: str = Field(..., description="Unique identifier for the engine unit.")
    cycle: int = Field(..., ge=1, description="Current operational cycle number.")
    sensors: Dict[str, float] = Field(
        ...,
        description="Sensor measurement values keyed by sensor name (e.g. s2–s21)."
    )
    op_settings: Dict[str, float] = Field(
        ...,
        description="Operational setting values (e.g. setting_1, setting_2, setting_3)."
    )


class UpdateSensorRequest(BaseModel):
    """Request body for pushing a new sensor observation to the digital twin."""
    engine_id: str = Field(..., description="Engine to update.")
    cycle: int = Field(..., ge=1, description="Cycle number of this observation.")
    sensor_data: Dict[str, float] = Field(..., description="New sensor readings.")
    op_settings: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional operational settings for this cycle."
    )


# ---------------------------------------------------------------------------
# Engine State
# ---------------------------------------------------------------------------

class EngineStateResponse(BaseModel):
    """Full snapshot of the digital twin state for one engine."""
    engine_id: str
    current_cycle: int
    health_state: HealthState
    health_score: float = Field(..., ge=0.0, le=100.0, description="0 (failed) – 100 (perfect).")
    predicted_rul: Optional[float] = Field(default=None, description="Predicted remaining useful life in cycles.")
    rul_lower: Optional[float] = Field(default=None, description="Lower bound of RUL confidence interval.")
    rul_upper: Optional[float] = Field(default=None, description="Upper bound of RUL confidence interval.")
    uncertainty: Optional[float] = Field(default=None, description="Model uncertainty / half-width of CI.")
    reference_rul: Optional[float] = Field(default=None, description="Dataset ground truth reference RUL if known.")
    is_warmed_up: bool = Field(default=False, description="True once SEQUENCE_LENGTH readings are collected.")
    warmup_progress: str = Field(default="0/30", description="Warm-up progress string.")
    degradation_rate: float = Field(default=0.0, description="Estimated cycles of RUL lost per cycle (negative = degrading).")
    is_degradation_accelerating: bool = False
    risk_level: RiskLevel = RiskLevel.Low
    sensor_readings: Dict[str, float] = Field(default_factory=dict)
    recent_rul_trend: List[Optional[float]] = Field(default_factory=list, description="Last N RUL predictions for sparkline.")
    last_updated: datetime
    intervention_history: List[Dict[str, Any]] = Field(default_factory=list)
    pending_recommendation: Optional[Dict[str, Any]] = None
    status_message: str
    model_metrics: Optional[List[Dict[str, Any]]] = None


# ---------------------------------------------------------------------------
# Intervention / Agent Models
# ---------------------------------------------------------------------------

class InterventionResult(BaseModel):
    """Outcome of simulating a single maintenance intervention."""
    intervention_id: str
    intervention_name: str
    current_rul: float
    predicted_rul_after: float
    rul_gain: float
    improvement_pct: float
    risk: str
    cost: str
    recommendation_score: float
    rank: int
    is_recommended: bool
    is_feasible: bool
    assumptions: List[str] = Field(default_factory=list)
    source: str = "simulated"
    data_source: str


class AgentAnalysisResponse(BaseModel):
    """Full output of the maintenance agent analysis for one engine."""
    engine_id: str
    analysis_timestamp: datetime
    identified_problem: str
    degradation_indicators: List[str]
    interventions: List[InterventionResult]
    recommended_intervention_id: str
    recommendation_score: float
    explanation: str
    reasoning_steps: List[str]
    safety_constraints_applied: List[str]
    requires_engineer_approval: bool = True
    approval_status: Literal["pending", "approved", "rejected"] = "pending"


class ApprovalRequest(BaseModel):
    """Engineer approval or rejection of an agent recommendation."""
    engine_id: str
    intervention_id: str
    engineer_notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Simulation Models
# ---------------------------------------------------------------------------

class SimulationRequest(BaseModel):
    """Request parameters for a what-if intervention simulation."""
    engine_id: str
    intervention_id: str
    n_future_cycles: int = Field(default=50, ge=1, le=500)


class SimulationResponse(BaseModel):
    """Result of a single intervention simulation run."""
    engine_id: str
    intervention_id: str
    intervention_name: str
    current_rul: float
    predicted_rul_after: float
    rul_gain: float
    rul_trajectory_baseline: List[float]
    rul_trajectory_intervention: List[float]
    health_trajectory: List[float]
    improvement_pct: float
    risk: str
    cost: str
    assumptions: List[str]
    source: Literal["simulated"]
    data_source: str
