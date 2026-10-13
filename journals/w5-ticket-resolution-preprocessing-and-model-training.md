# Ticket 05: C-MAPSS FD001 Preprocessing Pipeline & RUL Model Training

- **Owner:** Shruti Bansal (1024030391)
- **Module:** `ml/preprocess.py`, `ml/models.py`, `ml/train.py`, `ml/evaluate.py`, `scripts/run_training.py`
- **Milestone:** Sprint 2 / Implementation

> **DRAFT:** sections marked **[FILL]** need your own details. Delete this note when finished.

---

## 1. Problem Description & Difficulties Encountered

Raw C-MAPSS telemetry cannot be fed directly into a sequence model. It needs cleaning, scaling, windowing and RUL labelling, and then a model has to be trained and compared.

**[FILL]** Add the real difficulties you faced (for example: choosing the RUL cap, handling uninformative sensors, model not converging, poor evaluation scores) and how you noticed them.

---

## 2. Technical Context

Implemented in the project (see `config.py`):
- **Constant sensors removed** (7 of them), leaving **14 informative sensors**
- **Normalization** of sensor values
- **Sliding windows** of **30 cycles**
- **Piecewise-linear RUL cap** at **125**
- **Models defined** in `ml/models.py`: LSTM, GRU, CNN-LSTM, TCN, Transformer, all with **MC-Dropout** for uncertainty estimates
- **Evaluation** in `ml/evaluate.py`: MAE, RMSE, R2 and the NASA scoring function
- **CLI:** `scripts/run_training.py` (options for subset, model, epochs, quick mode)

---

## 3. Resolution & Engineering Implementation

### A. Preprocessing Pipeline
**[FILL]** Describe in your own words what you built in `ml/preprocess.py` and why you made each choice.

### B. Model Training
**[FILL]** State which model(s) you trained, the settings used (epochs, batch size, dropout) and the best checkpoint produced (`saved_models/lstm_best.h5`).

### C. Evaluation
**[FILL]** Add your final metrics after retraining and what they mean.

---

## 4. Verification & Outcome

**[FILL]** Add the final metrics table, how the best model was selected, and how it connects to the digital twin layer.
