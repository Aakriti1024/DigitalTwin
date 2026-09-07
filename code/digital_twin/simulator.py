import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

COLUMN_NAMES = (
    ["unit", "cycle", "op_1", "op_2", "op_3"] +
    [f"sensor_{i}" for i in range(1, 22)]
)

CONSTANT_SENSORS = ['op_3', 'sensor_1', 'sensor_5', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']

def load_cmapss_data(data_path="."):
    """Loads and preprocesses NASA C-MAPSS FD001 dataset."""
    import os
    if not os.path.exists(f"{data_path}/train_FD001.txt") and os.path.exists(f"{data_path}/data/train_FD001.txt"):
        data_path = f"{data_path}/data"

    train_file = f"{data_path}/train_FD001.txt"
    test_file = f"{data_path}/test_FD001.txt"
    rul_file = f"{data_path}/RUL_FD001.txt"

    train_df = pd.read_csv(train_file, sep=r"\s+", header=None, names=COLUMN_NAMES)
    test_df = pd.read_csv(test_file, sep=r"\s+", header=None, names=COLUMN_NAMES)
    rul_df = pd.read_csv(rul_file, sep=r"\s+", header=None, names=["RUL"])

    feature_cols = [c for c in train_df.columns if c not in ["unit", "cycle"] + CONSTANT_SENSORS]

    max_cycles = train_df.groupby("unit")["cycle"].max().to_dict()
    train_df["RUL"] = train_df.apply(lambda row: min(125.0, max_cycles[row["unit"]] - row["cycle"]), axis=1)

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    return train_df, test_df, rul_df, feature_cols, scaler

class EngineDigitalTwin:
    def __init__(self, unit_df, feature_cols, scaler):
        self.unit_df = unit_df.sort_values("cycle").reset_index(drop=True)
        self.feature_cols = feature_cols
        self.scaler = scaler
        self.total_recorded_cycles = len(self.unit_df)
        self.current_cycle = 0
        self.accumulated_wear = 0.0
        self.derate_factor = 1.0

    def apply_action(self, derate_factor):
        self.derate_factor = float(derate_factor)

    def step(self):
        if self.current_cycle >= self.total_recorded_cycles - 1:
            return None
        
        wear_step = 1.0 * (self.derate_factor ** 2)
        self.accumulated_wear += wear_step
        self.current_cycle += 1
        current_row = self.unit_df.iloc[self.current_cycle].copy()
        
        return {
            "unit": int(current_row["unit"]),
            "cycle": self.current_cycle,
            "derate_factor": self.derate_factor,
            "accumulated_wear": self.accumulated_wear,
            "features": current_row[self.feature_cols].values,
            "true_rul": max(0.0, self.total_recorded_cycles - self.current_cycle)
        }

def run_comparative_simulation(unit_df, model, scaler, feature_cols, agent, temp_offset=0.0, pressure_offset=0.0):
    """
    Simulates the physical parameter feedback loop:
    User modifies physical sensors (T30 Temp, P30 Pressure) -> Model predicts shift in RUL -> Agent recommends action!
    """
    seq_len = 30
    total_recorded = len(unit_df)
    
    # Create physically modified sensor matrix
    mod_unit_df = unit_df.copy()
    if "sensor_3" in mod_unit_df.columns:
        mod_unit_df["sensor_3"] += temp_offset  # T30 HPC Outlet Temp (°R)
    if "sensor_7" in mod_unit_df.columns:
        mod_unit_df["sensor_7"] += pressure_offset  # P30 HPC Outlet Pressure (psia)

    scaled_features = scaler.transform(mod_unit_df[feature_cols].values)

    # 1. Baseline Simulation (No Agent) under user's physical parameter conditions
    stress_multiplier = 1.0 + (temp_offset * 0.015) + (pressure_offset * 0.01)
    effective_baseline_length = max(10, int(total_recorded / stress_multiplier))

    baseline_records = []
    for cycle in range(1, effective_baseline_length + 1):
        health_pct = max(0.0, 100.0 * (effective_baseline_length - cycle) / effective_baseline_length)
        baseline_records.append({
            "cycle": cycle,
            "health_percent": round(health_pct, 1),
            "thrust_rating": 1.0,
            "status": "RUNNING" if cycle < effective_baseline_length else "CATASTROPHIC_FAILURE"
        })

    # 2. Agentic Closed-Loop Simulation
    agent_records = []
    accumulated_damage = 0.0
    window_buffer = []

    import torch

    for cycle in range(1, effective_baseline_length + 50):
        idx = min(cycle - 1, total_recorded - 1)
        current_feat = scaled_features[idx]
        window_buffer.append(current_feat)

        if len(window_buffer) > seq_len:
            window_buffer.pop(0)

        avg_t30 = float(mod_unit_df.iloc[idx]["sensor_3"]) if "sensor_3" in mod_unit_df.columns else 1500.0
        avg_p30 = float(mod_unit_df.iloc[idx]["sensor_7"]) if "sensor_7" in mod_unit_df.columns else 540.0
        eff_t30 = round(avg_t30 + temp_offset, 1)
        eff_p30 = round(avg_p30 + pressure_offset, 1)

        if len(window_buffer) < seq_len:
            accumulated_damage += 1.0 * stress_multiplier
            health_pct = max(0.0, 100.0 * (effective_baseline_length - accumulated_damage) / effective_baseline_length)
            agent_records.append({
                "cycle": cycle,
                "predicted_rul": float(effective_baseline_length - cycle),
                "effective_t30": eff_t30,
                "effective_p30": eff_p30,
                "temp_offset": temp_offset,
                "pressure_offset": pressure_offset,
                "health_percent": round(health_pct, 1),
                "thrust_rating": 1.0,
                "action_type": "NOMINAL",
                "action_headline": "CONTINUE NOMINAL OPERATION (100% RATING)",
                "derate_pct": 0.0,
                "gained_cycles": 0.0,
                "egt_temp_drop": 0.0,
                "rationale": "Telemetry buffering: Nominal operational state.",
                "counterfactual_matrix": [],
                "status": "IN_SERVICE"
            })
            continue

        # Predict RUL using Deep Learning Digital Twin under modified physical parameters
        input_window = np.array(window_buffer)
        input_tensor = torch.tensor(input_window, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred_rul = float(model(input_tensor).item()) / stress_multiplier

        # Physical Parameter Agent Evaluation
        decision = agent.evaluate_physical_telemetry(
            predicted_rul=pred_rul,
            current_cycle=cycle,
            avg_t30=avg_t30,
            avg_p30=avg_p30,
            temp_offset=temp_offset,
            pressure_offset=pressure_offset
        )
        derate = decision["derate_factor"]

        wear_rate = (derate ** 2.5) * stress_multiplier if derate > 0 else 0.0
        accumulated_damage += wear_rate

        remaining_life = max(0.0, effective_baseline_length - accumulated_damage)
        health_pct = max(0.0, 100.0 * remaining_life / effective_baseline_length)

        agent_records.append({
            "cycle": cycle,
            "predicted_rul": round(pred_rul, 1),
            "effective_t30": decision["effective_t30"],
            "effective_p30": decision["effective_p30"],
            "temp_offset": temp_offset,
            "pressure_offset": pressure_offset,
            "health_percent": round(health_pct, 1),
            "thrust_rating": derate,
            "action_type": decision["action_type"],
            "action_headline": decision["action_headline"],
            "derate_pct": decision["derate_pct"],
            "gained_cycles": decision["gained_cycles"],
            "egt_temp_drop": decision["egt_temp_drop"],
            "rationale": decision["rationale"],
            "counterfactual_matrix": decision["counterfactual_matrix"],
            "status": decision["status"]
        })

        if decision["status"] == "MAINTENANCE_TRIGGERED" or health_pct <= 0:
            break

    return pd.DataFrame(baseline_records), pd.DataFrame(agent_records)
