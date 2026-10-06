"""
ml/train.py
-----------
Training orchestration and model comparison for the C-MAPSS Digital Twin
ML pipeline.

Trains all five architectures (LSTM, GRU, CNN-LSTM, TCN, Transformer) on a
given C-MAPSS subset, evaluates on the held-out test set, and selects the
best model using a composite score.
"""

import os
import time
import numpy as np
import pandas as pd

from config import (
    MODELS_DIR,
    LOGS_DIR,
    EPOCHS,
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    SEQUENCE_LENGTH,
    N_FEATURES,
    RANDOM_STATE,
)
from ml.preprocess import preprocess_pipeline
from ml.models import get_all_models
from ml.evaluate import compute_metrics, nasa_scoring_function

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)


# ---------------------------------------------------------------------------
# Single-model training
# ---------------------------------------------------------------------------

def train_model(
    model,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
):
    """
    Train a single Keras model with standard callbacks.

    Callbacks used
    --------------
    * EarlyStopping         — stops when val_loss stops improving
    * ModelCheckpoint       — saves best weights to MODELS_DIR
    * ReduceLROnPlateau     — halves LR when val_loss plateaus

    Parameters
    ----------
    model : compiled Keras Model
    model_name : str  — used for checkpoint filename
    X_train, y_train : training arrays
    X_val, y_val     : validation arrays

    Returns
    -------
    model : Keras Model (best weights restored)
    history : Keras History object
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    checkpoint_path = os.path.join(MODELS_DIR, f"{model_name}_best.h5")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(3, EARLY_STOPPING_PATIENCE // 3),
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    return model, history


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compare_models(
    subset: str = "FD001",
    data_dir: str = None,
):
    """
    Train and compare all five model architectures on a C-MAPSS subset.

    Steps
    -----
    1. Preprocess data via preprocess_pipeline.
    2. Build all models via get_all_models.
    3. Train each model; record training time.
    4. Evaluate on test set; collect metrics.
    5. Compute composite score and select the best model.
    6. Persist comparison CSV and best-model name.

    Composite score (lower is better)
    ----------------------------------
    score = 0.4 * norm_RMSE + 0.3 * norm_MAE + 0.3 * norm_NASA_score

    Parameters
    ----------
    subset : str   one of 'FD001'–'FD004'
    data_dir : str or None

    Returns
    -------
    comparison_df : pd.DataFrame
    best_model_name : str
    """
    print(f"\n{'='*60}")
    print(f"  Model Comparison — Subset: {subset}")
    print(f"{'='*60}\n")

    # 1. Preprocess
    print("» Preprocessing data …")
    data = preprocess_pipeline(subset=subset, data_dir=data_dir)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    input_shape = (X_train.shape[1], X_train.shape[2])
    print(
        f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}\n"
    )

    # 2. Build models
    models = get_all_models(input_shape=input_shape)

    # 3+4. Train & evaluate
    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\n{'─'*50}")
        print(f"  Training: {name.upper()}")
        print(f"{'─'*50}")

        t_start = time.time()
        trained_model, _history = train_model(
            model, name, X_train, y_train, X_val, y_val
        )
        training_time = time.time() - t_start

        # Deterministic prediction for comparison metrics
        y_pred = trained_model.predict(X_test, verbose=0).ravel()
        metrics = compute_metrics(y_test, y_pred)
        metrics["model"] = name
        metrics["training_time"] = round(training_time, 2)
        results.append(metrics)
        trained_models[name] = trained_model

        print(
            f"  ✓ {name}: MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}"
            f"  R²={metrics['r2']:.4f}  NASA={metrics['nasa_score']:.1f}"
            f"  Time={training_time:.1f}s"
        )

    # 5. Build comparison DataFrame
    comparison_df = pd.DataFrame(
        results,
        columns=["model", "mae", "rmse", "r2", "nasa_score", "training_time"],
    )

    # Normalise metrics for composite score (min-max normalisation)
    for col in ["rmse", "mae", "nasa_score"]:
        col_min = comparison_df[col].min()
        col_max = comparison_df[col].max()
        denom = col_max - col_min if col_max != col_min else 1.0
        comparison_df[f"norm_{col}"] = (comparison_df[col] - col_min) / denom

    comparison_df["composite_score"] = (
        0.4 * comparison_df["norm_rmse"]
        + 0.3 * comparison_df["norm_mae"]
        + 0.3 * comparison_df["norm_nasa_score"]
    )
    comparison_df = comparison_df.sort_values("composite_score").reset_index(drop=True)

    best_model_name = comparison_df.iloc[0]["model"]

    # 6. Persist results
    os.makedirs(LOGS_DIR, exist_ok=True)
    csv_path = os.path.join(LOGS_DIR, "model_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)

    best_path = os.path.join(LOGS_DIR, "best_model.txt")
    with open(best_path, "w") as f:
        f.write(best_model_name)

    # Formatted table
    print(f"\n{'='*60}")
    print("  Model Comparison Results")
    print(f"{'='*60}")
    display_cols = ["model", "mae", "rmse", "r2", "nasa_score",
                    "training_time", "composite_score"]
    print(comparison_df[display_cols].to_string(index=False, float_format="{:.4f}".format))
    print(f"\n  ★ Best model: {best_model_name.upper()}")
    print(f"  Comparison saved to: {csv_path}")
    print(f"  Best model name saved to: {best_path}\n")

    return comparison_df, best_model_name


# ---------------------------------------------------------------------------
# Load persisted best model
# ---------------------------------------------------------------------------

def load_best_model(subset: str = "FD001"):
    """
    Load the best model identified by a previous compare_models() run.

    Reads the model name from LOGS_DIR/best_model.txt, then loads the
    corresponding weights from MODELS_DIR/<name>_best.h5.

    Parameters
    ----------
    subset : str  — currently unused but reserved for multi-subset workflows

    Returns
    -------
    model : compiled Keras Model
    model_name : str
    """
    best_path = os.path.join(LOGS_DIR, "best_model.txt")
    if not os.path.exists(best_path):
        raise FileNotFoundError(
            f"Best-model record not found at '{best_path}'. "
            "Run compare_models() first."
        )

    with open(best_path, "r") as f:
        model_name = f.read().strip()

    checkpoint_path = os.path.join(MODELS_DIR, f"{model_name}_best.h5")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at '{checkpoint_path}'."
        )

    model = tf.keras.models.load_model(checkpoint_path)
    print(f"Loaded best model '{model_name}' from '{checkpoint_path}'.")
    return model, model_name


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    comparison_df, best_name = compare_models(subset="FD001")
    print("\nFinal comparison table:")
    print(comparison_df[["model", "mae", "rmse", "r2", "nasa_score",
                          "composite_score"]].to_string(index=False))
    print(f"\nBest model: {best_name}")
