"""
ml/predict.py
-------------
Real-time inference with MC-Dropout uncertainty quantification.

MC-Dropout works by keeping Dropout layers active at inference time
(training=True).  Running N forward passes produces a distribution of
predictions; the mean is the point estimate and the standard deviation
captures epistemic uncertainty.
"""

import numpy as np

from config import (
    MC_DROPOUT_SAMPLES,
    MAX_RUL,
    HEALTH_THRESHOLDS,
    RISK_LEVELS,
    SELECTED_SENSORS,
    SEQUENCE_LENGTH,
)
from ml.preprocess import load_single_engine


# ---------------------------------------------------------------------------
# MC-Dropout inference
# ---------------------------------------------------------------------------

def mc_dropout_predict(
    model,
    X: np.ndarray,
    n_samples: int = MC_DROPOUT_SAMPLES,
):
    """
    Run Monte-Carlo Dropout inference.

    Executes n_samples stochastic forward passes with training=True so that
    Dropout is active, producing a distribution of RUL predictions per sample.

    Parameters
    ----------
    model   : compiled Keras Model (must contain Dropout layers)
    X       : np.ndarray, shape (N, seq_len, n_features)
    n_samples : int  — number of MC samples

    Returns
    -------
    mean_pred : np.ndarray, shape (N,)  — mean prediction across samples
    std_pred  : np.ndarray, shape (N,)  — std deviation (uncertainty)
    """
    preds = np.stack(
        [model(X, training=True).numpy().ravel() for _ in range(n_samples)],
        axis=0,
    )  # shape: (n_samples, N)

    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)
    return mean_pred, std_pred


# ---------------------------------------------------------------------------
# Single-sample RUL prediction
# ---------------------------------------------------------------------------

def predict_rul(
    model,
    X: np.ndarray,
    n_samples: int = MC_DROPOUT_SAMPLES,
) -> dict:
    """
    Predict RUL with uncertainty bounds for a single window.

    Calls mc_dropout_predict and packages the output into a dict with
    90 % confidence bounds (z = 1.65 for one-tailed 95 %).

    Parameters
    ----------
    model : Keras Model
    X     : np.ndarray, shape (1, seq_len, n_features)
    n_samples : int

    Returns
    -------
    dict with keys:
        predicted_rul  (float)  — point estimate (mean of MC samples)
        uncertainty    (float)  — 1-σ spread of MC samples
        rul_lower      (float)  — mean − 1.65 * std, clipped to ≥ 0
        rul_upper      (float)  — mean + 1.65 * std, clipped to ≥ 0
    """
    mean_pred, std_pred = mc_dropout_predict(model, X, n_samples)

    predicted_rul = float(np.clip(mean_pred[0], 0, None))
    uncertainty = float(std_pred[0])
    rul_lower = float(np.clip(mean_pred[0] - 1.65 * std_pred[0], 0, None))
    rul_upper = float(np.clip(mean_pred[0] + 1.65 * std_pred[0], 0, None))

    return {
        "predicted_rul": predicted_rul,
        "uncertainty": uncertainty,
        "rul_lower": rul_lower,
        "rul_upper": rul_upper,
    }


# ---------------------------------------------------------------------------
# Health-state classification
# ---------------------------------------------------------------------------

def get_health_state(predicted_rul: float) -> dict:
    """
    Map a predicted RUL to a health-state classification.

    Thresholds are read from config.HEALTH_THRESHOLDS, which should be a
    dict like:
        {
          'Healthy':   {'min_rul': 120, 'risk': 'Low'},
          'Degraded':  {'min_rul': 50,  'risk': 'Medium'},
          'Critical':  {'min_rul': 0,   'risk': 'High'},
        }

    The health_score is a 0–100 linear mapping from 0 (RUL=0) to 100 (RUL=MAX_RUL).

    Parameters
    ----------
    predicted_rul : float  — clipped-to-zero RUL value

    Returns
    -------
    dict with keys: health_state (str), risk_level (str), health_score (float)
    """
    health_score = float(np.clip(predicted_rul / MAX_RUL * 100, 0, 100))

    health_state = "Healthy"
    risk_level = "Low"

    for state_name, bounds in HEALTH_THRESHOLDS.items():
        if isinstance(bounds, (list, tuple)):
            low, high = bounds
            if low <= predicted_rul <= high or (low <= predicted_rul and np.isinf(high)):
                health_state = state_name
                risk_level = RISK_LEVELS.get(state_name, "Low")
                break
        elif isinstance(bounds, dict):
            if predicted_rul >= bounds.get("min_rul", 0):
                health_state = state_name
                risk_level = bounds.get("risk", "Low")
                break

    return {
        "health_state": health_state,
        "risk_level": risk_level,
        "health_score": round(health_score, 2),
    }


# ---------------------------------------------------------------------------
# Full engine-state prediction
# ---------------------------------------------------------------------------

def predict_engine_state(
    model,
    engine_df,
    scaler,
    feature_cols=None,
) -> dict:
    """
    End-to-end prediction for a single engine.

    Combines preprocessing (load_single_engine), MC-Dropout RUL prediction,
    and health-state classification into one call.

    Parameters
    ----------
    model      : Keras Model
    engine_df  : pd.DataFrame  — raw sensor readings for ONE engine
    scaler     : fitted MinMaxScaler or dict of scalers
    feature_cols : list[str] or None  — defaults to SELECTED_SENSORS

    Returns
    -------
    dict merging fields from predict_rul and get_health_state, plus:
        engine_id      — inferred from engine_df if column present
        n_cycles_seen  — number of rows in engine_df
    """
    if feature_cols is None:
        feature_cols = SELECTED_SENSORS

    X = load_single_engine(engine_df, scaler, SEQUENCE_LENGTH, feature_cols)

    rul_info = predict_rul(model, X)
    health_info = get_health_state(rul_info["predicted_rul"])

    engine_id = None
    if "engine_id" in engine_df.columns:
        engine_id = int(engine_df["engine_id"].iloc[0])

    result = {
        "engine_id": engine_id,
        "n_cycles_seen": len(engine_df),
        **rul_info,
        **health_info,
    }
    return result


# ---------------------------------------------------------------------------
# Degradation trend analysis
# ---------------------------------------------------------------------------

def get_degradation_trend(
    engine_history: list,
    model,
    scaler,
    window: int = 10,
    feature_cols=None,
) -> dict:
    """
    Compute the degradation trend from the most recent RUL predictions.

    Fits a linear trend (np.polyfit, degree=1) to the last `window` predicted
    RUL values.  A negative slope means the RUL is decreasing (expected); an
    increasing magnitude indicates acceleration.

    Parameters
    ----------
    engine_history : list of pd.DataFrame
        Each entry is a snapshot of the engine's sensor readings up to that
        cycle.  The function predicts RUL for each snapshot.
    model  : Keras Model
    scaler : fitted scaler
    window : int  — number of most recent snapshots to use
    feature_cols : list[str] or None

    Returns
    -------
    dict with keys:
        rul_trend         (list[float])  — predicted RUL for each snapshot
        degradation_rate  (float)        — slope of linear fit (cycles / cycle)
        is_accelerating   (bool)         — True if |slope| > threshold (0.5)
    """
    if feature_cols is None:
        feature_cols = SELECTED_SENSORS

    # Use the last `window` snapshots
    recent = engine_history[-window:]

    rul_trend = []
    for snapshot_df in recent:
        state = predict_engine_state(model, snapshot_df, scaler, feature_cols)
        rul_trend.append(state["predicted_rul"])

    x = np.arange(len(rul_trend), dtype=np.float64)
    if len(rul_trend) >= 2:
        coeffs = np.polyfit(x, rul_trend, deg=1)
        degradation_rate = float(coeffs[0])   # slope (cycles RUL / cycle time)
    else:
        degradation_rate = 0.0

    # Acceleration threshold: if the engine is losing more than 0.5 RUL/cycle
    is_accelerating = degradation_rate < -0.5

    return {
        "rul_trend": rul_trend,
        "degradation_rate": round(degradation_rate, 4),
        "is_accelerating": is_accelerating,
    }
