# System Architecture & Technical Specifications

## 1. Domain Overview
The system implements Prognostics and Health Management (PHM) for commercial turbofan engines using the NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) benchmark dataset (FD001 sub-dataset: 100 run-to-failure engine units under sea-level conditions).

## 2. High-Level Subsystems
1. **Telemetry Preprocessing & Buffering:**
   - Drops 7 constant non-informative sensors (`op_3`, `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`).
   - Normalizes 14 informative sensors using `StandardScaler`.
   - Constructs $30$-cycle temporal sliding windows ($X \in \mathbb{R}^{\text{batch} \times 30 \times 14}$).
2. **PyTorch LSTM Prognostics Engine:**
   - 2-layer stacked LSTM (`hidden_dim=64`, `dropout=0.2`).
   - Fully connected head mapping hidden state to scalar Remaining Useful Life (RUL).
   - Trained using MSE loss and Adam optimizer (`lr=0.001`).
3. **Prescriptive Policy Agent:**
   - Multi-tier risk threshold evaluation:
     - $\text{RUL} > 60$: Normal operation (derate factor $1.0$).
     - $35 < \text{RUL} \le 60$: Caution stage (recommended derate $0.95$).
     - $15 < \text{RUL} \le 35$: Warning stage (recommended derate $0.90$).
     - $\text{RUL} \le 15$: Critical stage (immediate maintenance trigger).
   - Generates auditable, natural-language rationale for every recommendation.
4. **Digital Twin Simulation Engine:**
   - Models physical wear reduction under derated operational regimes:
     $$\Delta \text{Wear} = 1.0 \times (\text{derate\_factor})^2$$
   - Simulates extended time-on-wing when an approved derate is applied.
5. **Streamlit Control Center:**
   - Human-in-the-Loop review and approval interface.
   - Interactive parameter injection sliders and JSON work-order dispatch.
