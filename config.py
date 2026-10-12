"""
Shared configuration for the Autonomous Agentic Digital Twin project.
All modules import from this file to ensure consistency.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "saved_models"
LOGS_DIR = BASE_DIR / "logs"

for _d in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# C-MAPSS Dataset
# ---------------------------------------------------------------------------
CMAPSS_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

# Raw column names as per NASA dataset readme
RAW_COLUMNS = (
    ["engine_id", "cycle"]
    + [f"op_set{i}" for i in range(1, 4)]
    + [f"s{i}" for i in range(1, 22)]
)

# Sensors that are constant / zero-variance across all FD subsets — removed during preprocessing
CONSTANT_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]

# Operational setting columns
OP_COLS = ["op_set1", "op_set2", "op_set3"]

# Sensor columns kept after removing constants
SELECTED_SENSORS = [
    "s2", "s3", "s4", "s7", "s8", "s9",
    "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21",
]

# Named groups used by the multi-branch LSTM. Their members must remain in
# SELECTED_SENSORS, whose order defines the model input feature order.
SENSOR_GROUPS = {
    "temperature": ["s2", "s3", "s4"],
    "pressure": ["s7", "s11"],
    "speed": ["s8", "s9", "s13", "s14"],
    "flow_other": ["s12", "s15", "s17", "s20", "s21"],
}

N_FEATURES = len(SELECTED_SENSORS)  # 14 features

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 30        # sliding-window length (cycles)
MAX_RUL = 125               # piecewise-linear RUL cap
TEST_SIZE = 0.20            # fraction of training engines for validation
VALIDATION_SIZE = 0.10      # fraction of remaining for val (from train split)
N_OPERATING_CLUSTERS = 6   # KMeans clusters for FD002/FD004 (multi-condition)
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.30         # used in all models + MC-Dropout at inference

# ---------------------------------------------------------------------------
# Health State Thresholds  (based on predicted RUL)
# ---------------------------------------------------------------------------
WARNING_RUL_THRESHOLD = 75
CRITICAL_RUL_THRESHOLD = 40

HEALTH_THRESHOLDS = {
    "Healthy":   (WARNING_RUL_THRESHOLD, float("inf")),
    "Warning":   (CRITICAL_RUL_THRESHOLD, WARNING_RUL_THRESHOLD),
    "Critical":  (15, CRITICAL_RUL_THRESHOLD),
    "Failure":   (0,  15),
}

RISK_LEVELS = {
    "Healthy":  "Low",
    "Warning":  "Medium",
    "Critical": "High",
    "Failure":  "Critical",
}

# ---------------------------------------------------------------------------
# MC-Dropout Uncertainty Estimation
# ---------------------------------------------------------------------------
MC_DROPOUT_SAMPLES = 50  # forward passes for uncertainty estimation

# ---------------------------------------------------------------------------
# Intervention Simulation
# (Virtual model — NOT real hardware validation.
#  Effects modelled as degradation-rate multipliers applied to the ML trajectory.)
# ---------------------------------------------------------------------------
INTERVENTIONS = {
    "no_action": {
        "id":                        "no_action",
        "name":                      "Continue Operation (No Action)",
        "degradation_multiplier":    1.00,   # no change to degradation rate
        "health_restoration":        0.00,   # no health point restoration
        "cost":                      "None",
        "source":                    "simulated",
        "assumption":                "Engine continues on current degradation trajectory.",
    },
    "reduce_load": {
        "id":                        "reduce_load",
        "name":                      "Reduce Operating Load",
        "degradation_multiplier":    0.80,
        "health_restoration":        0.05,
        "cost":                      "Low",
        "source":                    "simulated",
        "assumption":                (
            "Reducing engine thrust/load slows thermal degradation by ~20%. "
            "Health restoration of 5% represents minor recovery in thermal stress."
        ),
    },
    "fluid_service": {
        "id":                        "fluid_service",
        "name":                      "Oil / Fluid Service",
        "degradation_multiplier":    0.90,
        "health_restoration":        0.10,
        "cost":                      "Low",
        "source":                    "simulated",
        "assumption":                (
            "Lubrication service reduces friction-related degradation by ~10%. "
            "Small health restoration assumed from reduced bearing wear."
        ),
    },
    "schedule_maintenance": {
        "id":                        "schedule_maintenance",
        "name":                      "Schedule Preventive Maintenance",
        "degradation_multiplier":    0.70,
        "health_restoration":        0.25,
        "cost":                      "Medium",
        "source":                    "simulated",
        "assumption":                (
            "Planned maintenance includes minor component inspection and re-calibration, "
            "slowing degradation by ~30% and restoring 25% of lost health margin."
        ),
    },
    "component_replacement": {
        "id":                        "component_replacement",
        "name":                      "Replace / Restore Degraded Component",
        "degradation_multiplier":    0.55,
        "health_restoration":        0.50,
        "cost":                      "High",
        "source":                    "simulated",
        "assumption":                (
            "Major component replacement significantly resets degradation trajectory. "
            "50% health restoration and 45% degradation rate reduction assumed. "
            "Actual values depend on component type — NOT experimentally validated."
        ),
    },
}

# ---------------------------------------------------------------------------
# Agentic Decision Layer — Objective Function Weights
# (RUL gain, inverse risk, inverse cost, feasibility)
# ---------------------------------------------------------------------------
AGENT_WEIGHTS = {
    "rul_gain":    0.45,
    "safety":      0.30,
    "cost":        0.15,
    "feasibility": 0.10,
}

# Cost score mapping (lower cost → higher score)
COST_SCORES = {"None": 1.0, "Low": 0.80, "Medium": 0.50, "High": 0.25}

# Safety / risk constraint: if RUL < this threshold, only ≥ schedule_maintenance allowed
EMERGENCY_RUL_THRESHOLD = 10

# ---------------------------------------------------------------------------
# API Server
# ---------------------------------------------------------------------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
FRONTEND_DIR = BASE_DIR / "frontend"
