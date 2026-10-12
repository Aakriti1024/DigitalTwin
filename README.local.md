# Autonomous Agentic Digital Twin
## Predictive Maintenance & Closed-Loop Control of Turbofan Engines

> **A complete ML + Backend + Frontend system** for intelligent predictive maintenance using the NASA C-MAPSS turbofan engine dataset.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Project Structure](#project-structure)
5. [Step-by-Step Setup](#step-by-step-setup)
6. [Running the Project](#running-the-project)
7. [Dashboard Guide](#dashboard-guide)
8. [ML Models](#ml-models)
9. [Simulation Assumptions](#simulation-assumptions)
10. [API Reference](#api-reference)
11. [Evaluation Metrics](#evaluation-metrics)

---

## Project Overview

This system builds an **Autonomous Agentic Digital Twin** of a turbofan engine that:

1. **Predicts RUL** — Remaining Useful Life from sensor time-series using deep learning
2. **Identifies Degradation** — Analyzes sensor patterns and health state
3. **Recommends Interventions** — Agentic AI layer generates and ranks maintenance actions
4. **Simulates Actions Virtually** — Tests each intervention on the Digital Twin before real application
5. **Requires Engineer Approval** — Human-in-the-loop before any action is applied

### Closed-Loop Flow

```
Sensor Data → Preprocess → Predict RUL → Update Digital Twin
     ↓
Agent: Analyze → Generate Interventions → Simulate Each → Rank
     ↓
Engineer Reviews → Approve / Reject
     ↓
Apply to Digital Twin → Observe New State → Compare vs Baseline
```

---

## Architecture

```
digital-twin-turbofan/
├── config.py                   ← Shared constants (modify here)
├── requirements.txt
├── data/                       ← NASA C-MAPSS dataset (download separately)
├── saved_models/               ← Auto-saved trained models
├── logs/                       ← Training logs, model comparison CSV
├── ml/
│   ├── preprocess.py          ← C-MAPSS preprocessing pipeline
│   ├── models.py              ← 5 model architectures
│   ├── train.py               ← Training + model selection
│   ├── evaluate.py            ← Metrics (MAE, RMSE, R², NASA score)
│   ├── predict.py             ← Inference + MC-Dropout uncertainty
│   └── simulation.py         ← Virtual intervention simulation
├── backend/
│   ├── app.py                 ← FastAPI server
│   ├── digital_twin.py        ← DT state manager
│   ├── agent.py               ← Agentic decision layer
│   └── routes/                ← API route handlers
└── frontend/
    ├── index.html             ← Dashboard
    ├── css/style.css
    └── js/                    ← Dashboard JavaScript
```

---

## Quick Start

```bash
# 1. Clone / open project in VS Code
cd digital-twin-turbofan

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NASA C-MAPSS data
python scripts/download_data.py

# 5. Train models (takes ~10-20 min)
python scripts/run_training.py --subset FD001

# 6. Start the backend server
python backend/app.py

# 7. Open dashboard in browser
# http://localhost:8000
```

---

## Step-by-Step Setup

### Step 1: Install VS Code Extensions (Recommended)
- **Python** (Microsoft)
- **Pylance**
- **REST Client** (for testing API endpoints)

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate    # Mac/Linux
# OR
venv\Scripts\activate       # Windows
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note**: TensorFlow installation may take a few minutes. If you're on Apple Silicon (M1/M2/M3), use:
> ```bash
> pip install tensorflow-macos tensorflow-metal
> ```

### Step 4: Download NASA C-MAPSS Dataset

The dataset is **not included** due to size. Download it:

**Option A: Kaggle (Easiest)**
1. Go to: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
2. Download and extract
3. Place files in `data/` folder

**Option B: NASA Direct**
1. Go to: https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/
2. Download the zip
3. Extract to `data/`

**Expected files in `data/`:**
```
data/
├── train_FD001.txt
├── train_FD002.txt
├── train_FD003.txt
├── train_FD004.txt
├── test_FD001.txt
├── test_FD002.txt
├── test_FD003.txt
├── test_FD004.txt
├── RUL_FD001.txt
├── RUL_FD002.txt
├── RUL_FD003.txt
└── RUL_FD004.txt
```

Verify with:
```bash
python scripts/download_data.py
```

### Step 5: Train Models
```bash
# Train on FD001 (single operating condition — recommended to start)
python scripts/run_training.py --subset FD001

# Train on all subsets
python scripts/run_training.py --all-subsets

# View comparison of already-trained models
python scripts/run_training.py --compare-only
```

Training will:
- Preprocess data (normalize, window sequences, generate RUL labels)
- Train 5 model architectures: LSTM, GRU, CNN-LSTM, TCN, Transformer
- Compare using MAE, RMSE, R², NASA scoring function
- Auto-select the best model
- Save results to `logs/model_comparison.csv`

### Step 6: Start the Backend
```bash
python backend/app.py
```

Server starts at: `http://localhost:8000`

### Step 7: Open Dashboard
Open browser and go to: **`http://localhost:8000`**

---

## Running the Project

### VS Code Launch Configurations

Create `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Train Models",
            "type": "python",
            "request": "launch",
            "program": "scripts/run_training.py",
            "args": ["--subset", "FD001"],
            "console": "integratedTerminal"
        },
        {
            "name": "Start Backend Server",
            "type": "python",
            "request": "launch",
            "program": "backend/app.py",
            "console": "integratedTerminal"
        }
    ]
}
```

---

## Dashboard Guide

### Top Stats Row
| Card | Description |
|------|-------------|
| Health State | Healthy / Warning / Critical / Failure |
| Predicted RUL | Remaining cycles ± uncertainty (95% CI) |
| Risk Level | Low / Medium / High / Critical |
| Health Score | 0-100% composite score |

### Charts
- **RUL Trend**: Historical RUL predictions over time with threshold lines
- **Health Gauge**: Current health score as a doughnut chart
- **Sensor Trends**: Multi-line chart of normalized sensor values

### AI Agent Panel
1. Click **"Analyze Engine"** to trigger agent analysis
2. Agent identifies the problem and generates candidate interventions
3. Each intervention is **virtually simulated** — results appear in the table

### Intervention Table
Shows all candidate interventions ranked by objective score:
- **Predicted RUL**: Expected RUL if this action is taken *(simulated)*
- **RUL Gain**: Improvement over no-action baseline
- **Risk/Cost**: Qualitative assessment
- **Score**: Composite objective function score

> ⚠️ **Important**: RUL values in the simulation table are generated by a **virtual model** with documented assumptions. They are NOT experimentally validated hardware effects.

### What-If Simulation
- Select any intervention from the dropdown
- Click **"Run Simulation"**
- See baseline vs intervention RUL trajectories side-by-side

### Engineer Approval
- Review AI recommendation
- Add optional notes
- Click **Approve** or **Reject**
- The Digital Twin updates accordingly

---

## ML Models

| Model | Architecture | Notes |
|-------|-------------|-------|
| **LSTM** | 2-layer LSTM (128→64) + Dense | Strong baseline for time-series |
| **GRU** | 2-layer GRU (128→64) + Dense | Faster than LSTM, similar accuracy |
| **CNN-LSTM** | Conv1D blocks + LSTM(64) | Feature extraction + temporal modeling |
| **TCN** | Dilated causal Conv1D residual blocks | Parallelizable, long-range dependencies |
| **Transformer** | Multi-head attention + FFN | Best for capturing complex temporal patterns |

**Model Selection Criterion:**
```
composite_score = 0.4 × norm_RMSE + 0.3 × norm_MAE + 0.3 × norm_NASA_score
```
Lower is better. Best model auto-selected and saved.

**Uncertainty Estimation:**
MC-Dropout with 50 forward passes. 95% confidence interval reported as:
`[mean - 1.65×std, mean + 1.65×std]`

---

## Simulation Assumptions

> **⚠️ Important Scientific Disclaimer**
>
> The intervention simulation layer is a **virtual model**. It does NOT represent real hardware experimentation. Intervention effects are modeled as mathematical multipliers on the degradation trajectory.
>
> This simulation is clearly labeled throughout the codebase and dashboard with `"source": "simulated"` and `"data_source": "virtual_simulation"`.

| Intervention | Degradation Multiplier | Health Restoration | Basis |
|-------------|----------------------|-------------------|-------|
| No Action | 1.00× | 0% | Observed dataset trajectory |
| Reduce Load | 0.80× | 5% | Engineering assumption |
| Fluid Service | 0.90× | 10% | Engineering assumption |
| Preventive Maintenance | 0.70× | 25% | Engineering assumption |
| Component Replacement | 0.55× | 50% | Engineering assumption |

---

## API Reference

### Digital Twin
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/twin/state?engine_id=E1` | Get full DT state |
| POST | `/api/twin/update` | Feed new sensor cycle |
| GET | `/api/twin/history?engine_id=E1` | Get history |
| POST | `/api/twin/reset?engine_id=E1` | Reset engine state |

### Agent
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/agent/analyze?engine_id=E1` | Run agent analysis |
| GET | `/api/agent/interventions?engine_id=E1` | Get ranked interventions |
| POST | `/api/agent/approve` | Approve recommendation |
| POST | `/api/agent/reject` | Reject recommendation |

### Simulation
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/simulation/what-if` | Run what-if simulation |
| GET | `/api/simulation/compare?engine_id=E1` | All interventions comparison |

### Interactive Docs
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Evaluation Metrics

### RUL Prediction
| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error (cycles) |
| **RMSE** | Root Mean Square Error (penalizes large errors) |
| **R²** | Coefficient of determination |
| **NASA Score** | Asymmetric: penalizes early predictions more than late |

**NASA Scoring Function:**
```
s = predicted - actual
score = Σ exp(-s/13) - 1  for s < 0  (early prediction)
      + Σ exp(s/10) - 1   for s ≥ 0  (late prediction — more penalized)
```

### Agent/Decision Layer
- RUL improvement over no-action baseline
- Safety constraint violations (should be 0)
- Recommendation consistency across similar states

### Closed-Loop System
- Compare passive baseline vs AI-assisted intervention
- Measure RUL extension, degradation rate reduction

---

## Troubleshooting

**TensorFlow not found:**
```bash
pip install tensorflow
```

**Port 8000 already in use:**
```bash
# Change port in config.py: API_PORT = 8001
```

**No model loaded (demo mode):**
The dashboard runs in demo mode with synthetic data if no trained model is found.
Train a model first: `python scripts/run_training.py --subset FD001`

**CORS errors in browser:**
Make sure you're accessing via `http://localhost:8000` (the FastAPI server), not by opening `index.html` directly as a file.
