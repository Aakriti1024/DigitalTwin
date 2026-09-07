import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from prognostics.model import CMAPSSDataset, LSTMRULPredictor
from digital_twin.simulator import load_cmapss_data, EngineDigitalTwin
from agent.policy import AutonomousMaintenanceAgent

def train_rul_model(train_df, feature_cols, scaler, epochs=10, batch_size=64, seq_len=30):
    print("[PROGNOSTICS] Preparing PyTorch dataset...")
    
    # Scale features
    scaled_features = scaler.transform(train_df[feature_cols].values)
    targets = train_df["RUL"].values

    dataset = CMAPSSDataset(scaled_features, targets, sequence_length=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = len(feature_cols)
    model = LSTMRULPredictor(input_dim=input_dim, hidden_dim=64, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"[PROGNOSTICS] Training PyTorch LSTM Model for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        
        avg_loss = total_loss / len(dataset)
        rmse = np.sqrt(avg_loss)
        print(f"  Epoch {epoch+1}/{epochs} | Train Loss (MSE): {avg_loss:.2f} | RMSE: {rmse:.2f}")

    return model

def run_agentic_closed_loop_demo():
    print("[INIT] Starting Agentic Digital Twin Pipeline Execution...")
    
    # 1. Load Data
    train_df, test_df, rul_df, feature_cols, scaler = load_cmapss_data(data_path=".")
    
    # 2. Train Model
    model = train_rul_model(train_df, feature_cols, scaler, epochs=8, batch_size=128, seq_len=30)
    model.eval()

    # 3. Select a Test Engine (e.g. Unit 24 from test dataset)
    test_unit_id = 24
    unit_df = test_df[test_df["unit"] == test_unit_id].copy()
    print(f"\n[DIGITAL TWIN] Initializing Twin for Engine Unit #{test_unit_id} ({len(unit_df)} recorded cycles)...")

    # 4. Instantiate Digital Twin & Agent
    twin = EngineDigitalTwin(unit_df, feature_cols, scaler)
    agent = AutonomousMaintenanceAgent(critical_threshold=15.0, warning_threshold=35.0, caution_threshold=60.0)

    # 5. Run Closed-Loop Simulation
    simulation_log = []
    window_buffer = []
    
    scaled_unit_features = scaler.transform(unit_df[feature_cols].values)
    seq_len = 30

    for step_idx in range(len(unit_df)):
        current_feat = scaled_unit_features[step_idx]
        window_buffer.append(current_feat)
        
        if len(window_buffer) < seq_len:
            continue  # Waiting for initial sequence window buffer
        
        # Extract sliding window
        input_window = np.array(window_buffer[-seq_len:])
        input_tensor = torch.tensor(input_window, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            predicted_rul = float(model(input_tensor).item())
        
        # Calculate true RUL for comparison
        true_rul = float(len(unit_df) - step_idx - 1)
        
        # Agent evaluates state & decides closed-loop policy action
        decision = agent.evaluate_telemetry_and_decide(
            predicted_rul=predicted_rul,
            current_cycle=step_idx + 1
        )
        
        # Apply action to Digital Twin
        twin.apply_action(decision["derate_factor"])
        
        decision["true_rul"] = true_rul
        simulation_log.append(decision)
        
        if decision["status"] == "MAINTENANCE_TRIGGERED":
            print(f"\n[CLOSED-LOOP ACTION] Cycle {step_idx+1}: {decision['rationale']}")
            print(f"[STOP] Engine Unit #{test_unit_id} taken offline safely for maintenance!")
            break

    sim_df = pd.DataFrame(simulation_log)
    print("\n[SUCCESS] Simulation Complete! First 5 cycles:")
    print(sim_df[["cycle", "predicted_rul", "true_rul", "action", "derate_factor"]].head())
    print("\nLast 5 cycles before Agent Intervention:")
    print(sim_df[["cycle", "predicted_rul", "true_rul", "action", "derate_factor"]].tail())

    # Save outputs
    os.makedirs("results", exist_ok=True)
    sim_df.to_csv("results/agentic_digital_twin_simulation.csv", index=False)
    print("\n[SAVED] Results saved to results/agentic_digital_twin_simulation.csv")

if __name__ == "__main__":
    run_agentic_closed_loop_demo()
