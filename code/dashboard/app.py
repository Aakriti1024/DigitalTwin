import sys
import os
import time
import torch
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prognostics.model import LSTMRULPredictor
from digital_twin.simulator import load_cmapss_data, run_comparative_simulation
from agent.policy import PhysicalParameterAgentEngine

st.set_page_config(page_title="Physical Digital Twin Control Center", page_icon="✈️", layout="wide")

st.title("✈️ Physical Agentic Digital Twin: C-MAPSS Sensor Telemetry & Control")
st.markdown("""
<div style="background-color: #0F172A; padding: 15px; border-radius: 10px; border-left: 5px solid #3B82F6; margin-bottom: 20px;">
    <h4 style="margin:0; color: #60A5FA;">🌡️ Physical Parameter Control & Proactive AI Response</h4>
    <p style="margin:5px 0 0 0; color: #94A3B8; font-size: 14px;">
        Adjust real physical engine parameters (HPC Temp T30, Compressor Pressure P30) in the sidebar. 
        The PyTorch LSTM Twin predicts the resulting RUL shift, and the AI Agent proactively commands physical derating actions!
    </p>
</div>
""", unsafe_allow_html=True)

data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
train_df, test_df, rul_df, feature_cols, scaler = load_cmapss_data(data_path=data_path)

# Sidebar - Physical Parameter Controls (C-MAPSS Dataset Parameters)
st.sidebar.header("⚙️ C-MAPSS Physical Telemetry Controls")

available_units = sorted(test_df["unit"].unique())
selected_unit = st.sidebar.selectbox("Select Turbofan Unit", available_units, index=2)

st.sidebar.subheader("🌡️ Injected Operational Conditions")
temp_offset = st.sidebar.slider("HPC Outlet Temp T30 Offset (°R)", 0.0, 35.0, 0.0, step=1.0, help="Simulates thermal stress / high ambient temperature.")
pressure_offset = st.sidebar.slider("HPC Outlet Pressure P30 Offset (psia)", 0.0, 20.0, 0.0, step=1.0, help="Simulates compressor load / climbing flight profile.")

st.sidebar.subheader("🎯 Safety Thresholds")
egt_limit = st.sidebar.slider("EGT Safety Limit T30 (°R)", 1500, 1650, 1580, step=10)

# Load PyTorch model
model = LSTMRULPredictor(input_dim=len(feature_cols), hidden_dim=64, num_layers=2)
model.eval()

agent = PhysicalParameterAgentEngine(egt_limit_degR=egt_limit)

unit_df = test_df[test_df["unit"] == selected_unit].copy().reset_index(drop=True)

# Run Simulation with Physical Parameter Offsets
base_df, agent_df = run_comparative_simulation(
    unit_df=unit_df,
    model=model,
    scaler=scaler,
    feature_cols=feature_cols,
    agent=agent,
    temp_offset=temp_offset,
    pressure_offset=pressure_offset
)

last_agent = agent_df.iloc[-1]
last_base = base_df.iloc[-1]

# -------------------------------------------------------------
# PROACTIVE AGENT RECOMMENDATION BOX (RESPONDS TO PHYSICAL PARAMETERS)
# -------------------------------------------------------------
st.markdown("### 🤖 Proactive Agent Recommendation (Responding to Physical Parameters)")

recommendation_color = "#EF4444" if "SCHEDULE" in last_agent["action_type"] else ("#F59E0B" if "REDUCE" in last_agent["action_type"] else "#10B981")

st.markdown(f"""
<div style="background-color: #1E293B; padding: 20px; border-radius: 12px; border: 2px solid {recommendation_color};">
    <span style="background-color: {recommendation_color}; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 12px;">
        PROACTIVE AGENT ACTION COMMAND
    </span>
    <h2 style="color: white; margin-top: 10px; font-weight: 800;">
        👉 {last_agent['action_headline']}
    </h2>
    <p style="color: #E2E8F0; font-size: 16px; line-height: 1.5;">
        <b>Physics & RUL Rationale:</b> {last_agent['rationale']}
    </p>
    <div style="display: flex; gap: 20px; margin-top: 15px; padding-top: 15px; border-top: 1px solid #334155;">
        <div><small style="color: #94A3B8;">Physical T30 Temp</small><br><b style="color: #F59E0B; font-size: 18px;">{last_agent['effective_t30']} °R</b></div>
        <div><small style="color: #94A3B8;">Predicted RUL</small><br><b style="color: white; font-size: 18px;">{last_agent['predicted_rul']} Cycles</b></div>
        <div><small style="color: #94A3B8;">EGT Temp Reduction</small><br><b style="color: #34D399; font-size: 18px;">-{last_agent['egt_temp_drop']} °R</b></div>
        <div><small style="color: #94A3B8;">Gained Time-on-Wing</small><br><b style="color: #60A5FA; font-size: 18px;">+{last_agent['gained_cycles']} Cycles</b></div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# HUMAN-IN-THE-LOOP (HITL) APPROVAL GATE
# -------------------------------------------------------------
if "approval_status" not in st.session_state:
    st.session_state["approval_status"] = "Approved"

col_hitl1, col_hitl2, col_hitl3 = st.columns([2, 2, 4])
with col_hitl1:
    if st.button("✅ Approve Action", key="btn_approve", use_container_width=True):
        st.session_state["approval_status"] = "Approved"
with col_hitl2:
    if st.button("❌ Reject (Passive)", key="btn_reject", use_container_width=True):
        st.session_state["approval_status"] = "Rejected"

with col_hitl3:
    if st.session_state["approval_status"] == "Approved":
        st.success("🟢 **Action Authorized by Human Engineer:** Closed-loop derate committed to Digital Twin & ERP.")
    else:
        st.warning("⚠️ **Action Rejected by Human Engineer:** Digital Twin operating on passive degradation path.")

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# COUNTERFACTUAL WHAT-IF SIMULATION MATRIX
# -------------------------------------------------------------
st.markdown("### 🔮 Counterfactual What-If Simulation Matrix")
st.markdown("Background simulation evaluating candidate physical operating modes:")

if len(last_agent["counterfactual_matrix"]) > 0:
    st.table(pd.DataFrame(last_agent["counterfactual_matrix"]))

st.markdown("---")

# Comparative Trajectory Plot
st.subheader("📉 Physical Degradation Curve Shift under Telemetry Modifications")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

# Health Curve
ax1.plot(base_df["cycle"], base_df["health_percent"], label=f"Standard Engine (No Agent - Failed @ Cycle {int(last_base['cycle'])})", color="red", linestyle="--", linewidth=2)
ax1.plot(agent_df["cycle"], agent_df["health_percent"], label=f"Agentic Digital Twin (Safe @ Cycle {int(last_agent['cycle'])})", color="green", linewidth=2.5)
ax1.set_ylabel("Engine Health (%)")
ax1.set_title(f"Engine Degradation Curve (T30 Offset = +{temp_offset}°R, P30 Offset = +{pressure_offset} psia)")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-5, 105)

# Thrust Signal Curve
ax2.step(agent_df["cycle"], agent_df["thrust_rating"], where="post", color="darkgreen", linewidth=2.5, label="Agent Closed-Loop Action (Thrust Rating)")
ax2.set_xlabel("Operational Cycle")
ax2.set_ylabel("Thrust Rating (1.0 = 100%, 0.9 = 90%)")
ax2.set_ylim(0.85, 1.05)
ax2.set_title("Closed-Loop Thrust Rating Signal Sent to Engine Controller")
ax2.legend(loc="lower left")
ax2.grid(True, alpha=0.3)

st.pyplot(fig)

st.markdown("---")

# Dispatch Work Order Payload
st.markdown("### 📋 Autonomous Dispatch Command Payload")
st.json({
    "protocol": "PHYSICAL_AGENT_DISPATCH_v3.0",
    "timestamp_cycle": int(last_agent["cycle"]),
    "asset_id": f"TURBOFAN_ENGINE_UNIT_{selected_unit}",
    "physical_sensor_readings": {
        "HPC_Outlet_Temp_T30": f"{last_agent['effective_t30']} degR",
        "HPC_Outlet_Pressure_P30": f"{last_agent['effective_p30']} psia",
        "applied_temp_offset": f"+{temp_offset} degR",
        "applied_pressure_offset": f"+{pressure_offset} psia"
    },
    "digital_twin_prediction": {
        "predicted_rul": float(last_agent["predicted_rul"]),
        "gained_cycles_via_action": float(last_agent["gained_cycles"])
    },
    "agent_command_issued": {
        "action_headline": last_agent["action_headline"],
        "recommended_derate_percentage": f"{last_agent['derate_pct']}%",
        "rationale": last_agent["rationale"]
    }
})
