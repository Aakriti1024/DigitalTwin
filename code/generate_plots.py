import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_simulation_plots():
    csv_path = "results/agentic_digital_twin_simulation.csv"
    if not os.path.exists(csv_path):
        print("[ERROR] Simulation results CSV not found. Run train_and_simulate.py first.")
        return

    df = pd.read_csv(csv_path)
    os.makedirs("results/plots", exist_ok=True)

    # -------------------------------------------------------------
    # Plot 1: RUL Trajectory (True RUL vs Predicted RUL)
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(df["cycle"], df["true_rul"], label="Ground Truth RUL", color="navy", linewidth=2.5)
    plt.plot(df["cycle"], df["predicted_rul"], label="LSTM Digital Twin Predicted RUL", color="darkorange", linestyle="--", linewidth=2)
    
    # Add risk threshold lines
    plt.axhline(y=60, color="gold", linestyle=":", label="Caution Threshold (Derate 5%)")
    plt.axhline(y=35, color="orange", linestyle=":", label="Warning Threshold (Derate 10%)")
    plt.axhline(y=15, color="red", linestyle=":", label="Critical Threshold (Schedule Maintenance)")

    plt.title("Engine Digital Twin: Remaining Useful Life (RUL) Trajectory & Risk Thresholds", fontsize=12, fontweight="bold")
    plt.xlabel("Operational Cycle", fontsize=10)
    plt.ylabel("Remaining Useful Life (Cycles)", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("results/plots/rul_prediction_trajectory.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # Plot 2: Closed-Loop Agent Actions & Derate Factor over time
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.step(df["cycle"], df["derate_factor"], where="post", color="crimson", linewidth=2.5, label="Agent Control Action (Thrust Rating)")
    plt.title("Autonomous Agent Closed-Loop Control Signal (Thrust Derating)", fontsize=12, fontweight="bold")
    plt.xlabel("Operational Cycle", fontsize=10)
    plt.ylabel("Thrust Derate Factor (1.0 = 100%, 0.9 = 90%)", fontsize=10)
    plt.ylim(0.85, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig("results/plots/agent_control_actions.png", dpi=300)
    plt.close()

    print("[SUCCESS] Plots generated successfully in results/plots/")
    print("  - results/plots/rul_prediction_trajectory.png")
    print("  - results/plots/agent_control_actions.png")

if __name__ == "__main__":
    generate_simulation_plots()
