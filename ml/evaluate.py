"""
ml/evaluate.py
--------------
Evaluation metrics and visualisation utilities for C-MAPSS RUL models.

Metrics
-------
* MAE    — Mean Absolute Error
* RMSE   — Root Mean Squared Error
* R²     — Coefficient of determination
* NASA score — Asymmetric exponential penalty (lower is better)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend safe for headless envs
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def nasa_scoring_function(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    NASA C-MAPSS asymmetric scoring function.

    For each sample:
        s = y_pred − y_true
        penalty = exp(−s / 13) − 1   if s < 0  (late prediction)
                  exp( s / 10) − 1   if s ≥ 0  (early prediction)

    Total score = Σ penalties.  **Lower is better.**

    The function penalises late predictions more heavily than early ones.

    Parameters
    ----------
    y_true : array-like  — true RUL values
    y_pred : array-like  — predicted RUL values

    Returns
    -------
    float  — NASA score (non-negative sum of exponential penalties)
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    s = y_pred - y_true
    penalties = np.where(s < 0, np.expm1(-s / 13), np.expm1(s / 10))
    return float(np.sum(penalties))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute all evaluation metrics for a set of RUL predictions.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like

    Returns
    -------
    dict with keys: mae, rmse, r2, nasa_score
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    nasa = nasa_scoring_function(y_true, y_pred)

    return {"mae": mae, "rmse": rmse, "r2": r2, "nasa_score": nasa}


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: str = None,
):
    """
    Plot actual versus predicted RUL, sorted by true RUL (ascending).

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    model_name : str  — used in title and filename
    save_path : str or None  — if given, saves the figure here
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    sort_idx = np.argsort(y_true)
    y_true_s = y_true[sort_idx]
    y_pred_s = y_pred[sort_idx]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_true_s, label="Actual RUL", color="steelblue", linewidth=1.5)
    ax.plot(y_pred_s, label="Predicted RUL", color="coral",
            linewidth=1.5, alpha=0.85)
    ax.set_title(f"{model_name} — Actual vs. Predicted RUL")
    ax.set_xlabel("Sample (sorted by true RUL)")
    ax.set_ylabel("Remaining Useful Life (cycles)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)

    plt.close(fig)


def plot_training_history(
    history,
    model_name: str,
    save_path: str = None,
):
    """
    Plot training and validation loss / MAE curves from a Keras History object.

    Parameters
    ----------
    history : keras History  — returned by model.fit()
    model_name : str
    save_path : str or None
    """
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, hist["loss"], label="Train Loss", color="steelblue")
    if "val_loss" in hist:
        axes[0].plot(epochs, hist["val_loss"], label="Val Loss", color="coral")
    axes[0].set_title(f"{model_name} — Loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE
    mae_key = "mae" if "mae" in hist else "mean_absolute_error"
    val_mae_key = "val_mae" if "val_mae" in hist else "val_mean_absolute_error"
    if mae_key in hist:
        axes[1].plot(epochs, hist[mae_key], label="Train MAE", color="steelblue")
    if val_mae_key in hist:
        axes[1].plot(epochs, hist[val_mae_key], label="Val MAE", color="coral")
    axes[1].set_title(f"{model_name} — MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (cycles)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Training History — {model_name}", fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)

    plt.close(fig)


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_all_models(
    models_dict: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_dir: str = None,
) -> pd.DataFrame:
    """
    Run compute_metrics on every model in models_dict and return a summary
    DataFrame.

    Parameters
    ----------
    models_dict : dict[str, keras.Model]
        Keys are model names, values are compiled Keras models.
    X_test : np.ndarray  shape (N, seq_len, n_features)
    y_test : np.ndarray  shape (N,)
    save_dir : str or None
        If provided, prediction plots are saved to save_dir/<model_name>_pred.png

    Returns
    -------
    pd.DataFrame with columns: model, mae, rmse, r2, nasa_score
    """
    rows = []
    for name, model in models_dict.items():
        y_pred = model.predict(X_test, verbose=0).ravel()
        metrics = compute_metrics(y_test, y_pred)
        metrics["model"] = name
        rows.append(metrics)

        if save_dir:
            pred_path = os.path.join(save_dir, f"{name}_predictions.png")
            plot_predictions(y_test, y_pred, model_name=name, save_path=pred_path)

    df = pd.DataFrame(rows, columns=["model", "mae", "rmse", "r2", "nasa_score"])
    return df
