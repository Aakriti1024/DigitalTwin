#!/usr/bin/env python3
"""
scripts/run_training.py
-----------------------
CLI entry point for training, evaluating, and comparing deep-learning time-series
models on the NASA C-MAPSS dataset for the Autonomous Agentic Digital Twin.
"""

import sys
import os
import argparse
import time
import random
from datetime import datetime, timezone

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    DATA_DIR,
    MODELS_DIR,
    LOGS_DIR,
    CMAPSS_SUBSETS,
    SEQUENCE_LENGTH,
    N_FEATURES,
    EPOCHS,
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    SENSOR_GROUPS,
)
import numpy as np
import pandas as pd


COMPARISON_LOG_COLUMNS = [
    "timestamp",
    "subset",
    "model",
    "seed",
    "epochs_trained",
    "training_seconds",
    "last_window_mae",
    "last_window_rmse",
    "last_window_r2",
    "last_window_nasa_score",
    "all_window_mae",
    "all_window_rmse",
    "all_window_r2",
    "all_window_nasa_score",
]


def set_random_seeds(seed: int, tf) -> None:
    """Set Python, NumPy, and TensorFlow random seeds for a training run."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def append_comparison_rows(csv_path, rows) -> None:
    """Append completed model runs while preserving any existing log rows."""
    records = []
    for row in rows:
        record = {column: row.get(column) for column in COMPARISON_LOG_COLUMNS}
        record["timestamp"] = record["timestamp"] or datetime.now(timezone.utc).isoformat()
        records.append(record)
    new_rows = pd.DataFrame(records, columns=COMPARISON_LOG_COLUMNS)

    if not os.path.isfile(csv_path):
        new_rows.to_csv(csv_path, index=False)
        return

    existing_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    if existing_columns == COMPARISON_LOG_COLUMNS:
        new_rows.to_csv(csv_path, mode="a", header=False, index=False)
        return

    # Preserve pre-existing legacy rows, then use the append-only schema going
    # forward. Legacy fields that did not exist are left empty.
    legacy_rows = pd.read_csv(csv_path).rename(
        columns={"training_time": "training_seconds"}
    )
    legacy_rows = legacy_rows.reindex(columns=COMPARISON_LOG_COLUMNS)
    pd.concat([legacy_rows, new_rows], ignore_index=True).to_csv(csv_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train & Evaluate Turbofan RUL Prediction Models"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="FD001",
        choices=CMAPSS_SUBSETS,
        help="C-MAPSS subset to train on (default: FD001)",
    )
    parser.add_argument(
        "--all-subsets",
        action="store_true",
        help="Train across all subsets (FD001 to FD004)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=[
            "all", "lstm", "multibranch_lstm", "multibranch_noattn",
            "gru", "cnn_lstm", "tcn", "transformer",
        ],
        help="Specific model architecture to train (default: all)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Number of training epochs (default: {EPOCHS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Python, NumPy, and TensorFlow (default: 42)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick verification mode: 5 epochs on single model (LSTM)",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Display existing comparison results from logs without retraining",
    )
    return parser.parse_args()


def display_comparison_results():
    csv_path = os.path.join(LOGS_DIR, "model_comparison.csv")
    best_path = os.path.join(LOGS_DIR, "best_model.txt")

    if not os.path.isfile(csv_path):
        print(f"❌ No existing comparison results found at: {csv_path}")
        print("Run training first: python scripts/run_training.py --quick")
        return False

    df = pd.read_csv(csv_path)
    best_model = "Unknown"
    if os.path.isfile(best_path):
        with open(best_path, "r") as f:
            best_model = f.read().strip()

    print("\n" + "=" * 70)
    print(" EXISTING MODEL EVALUATION & COMPARISON RESULTS")
    print("=" * 70)
    print(df.to_string(index=False))
    print("-" * 70)
    print(f"★ Currently Selected Best Model: {best_model.upper()}")
    print("=" * 70 + "\n")
    return True


def run_training_workflow(
    subset="FD001", model_choice="all", epochs=None, quick=False, seed=42
):
    import tensorflow as tf
    from ml.preprocess import preprocess_pipeline
    from ml.models import (
        build_lstm,
        build_multibranch_lstm,
        build_multibranch_noattn,
        build_gru,
        build_cnn_lstm,
        build_tcn,
        build_transformer,
        get_all_models,
    )
    from ml.evaluate import compute_metrics
    from ml.models import get_group_attention_model
    from ml.predict import mc_dropout_predict, predict_rul, get_health_state

    set_random_seeds(seed, tf)

    if quick:
        effective_epochs = 5
        print("\n⚡ QUICK VERIFICATION MODE ACTIVATED (5 epochs, baseline LSTM model)")
    else:
        effective_epochs = epochs or EPOCHS

    print("=" * 70)
    print(f" NASA C-MAPSS {subset} Turbofan RUL Prediction Training")
    print("=" * 70)
    print(f"• Dataset Subset:   {subset}")
    print(f"• Sequence Window:  {SEQUENCE_LENGTH} cycles")
    print(f"• Sensor Features:  {N_FEATURES}")
    print(f"• Epochs:           {effective_epochs}")
    print(f"• Random Seed:      {seed}")
    print(f"• Batch Size:       {BATCH_SIZE}")
    print("-" * 70)

    # 1. Preprocess data
    print("Step 1: Running C-MAPSS Preprocessing Pipeline...")
    t0 = time.time()
    data = preprocess_pipeline(subset=subset, data_dir=DATA_DIR)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    X_test_all_windows = data["X_test_all_windows"]
    y_test_all_windows = data["y_test_all_windows"]
    scaler = data["scaler"]
    feature_cols = data["feature_cols"]

    print(f"  ✓ Preprocessing complete in {time.time() - t0:.2f}s")
    print(f"  ✓ X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
    print(f"  ✓ X_val shape:   {X_val.shape}   | y_val shape:   {y_val.shape}")
    print(f"  ✓ X_test shape:  {X_test.shape}  | y_test shape:  {y_test.shape}")

    input_shape = (SEQUENCE_LENGTH, N_FEATURES)

    # 2. Select models
    if quick or model_choice == "lstm":
        models_to_train = {"lstm": build_lstm(input_shape)}
        models_to_train["lstm"].compile(
            optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"]
        )
    elif model_choice == "multibranch_lstm":
        models_to_train = {"multibranch_lstm": build_multibranch_lstm(input_shape)}
        models_to_train["multibranch_lstm"].compile(
            optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"]
        )
    elif model_choice == "multibranch_noattn":
        models_to_train = {"multibranch_noattn": build_multibranch_noattn(input_shape)}
        models_to_train["multibranch_noattn"].compile(
            optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"]
        )
    elif model_choice == "gru":
        models_to_train = {"gru": build_gru(input_shape)}
        models_to_train["gru"].compile(
            optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"]
        )
    elif model_choice == "cnn_lstm":
        models_to_train = {"cnn_lstm": build_cnn_lstm(input_shape)}
        models_to_train["cnn_lstm"].compile(
            optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"]
        )
    elif model_choice == "tcn":
        models_to_train = {"tcn": build_tcn(input_shape)}
        models_to_train["tcn"].compile(
            optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"]
        )
    elif model_choice == "transformer":
        models_to_train = {"transformer": build_transformer(input_shape)}
        models_to_train["transformer"].compile(
            optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"]
        )
    else:
        models_to_train = get_all_models(input_shape)

    # 3. Train models
    print("\nStep 2: Training Model(s)...")
    results = []

    for name, model in models_to_train.items():
        print(f"\n---> Training: {name.upper()}")
        checkpoint_path = os.path.join(MODELS_DIR, f"{name}_best.h5")

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=max(3, EARLY_STOPPING_PATIENCE // 3),
                min_lr=1e-6,
                verbose=0,
            ),
        ]

        t_start = time.time()
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=effective_epochs,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
        )
        t_elapsed = time.time() - t_start
        epochs_trained = len(history.history["loss"])
        print(f"---> Epochs actually trained: {epochs_trained}")

        # Primary C-MAPSS protocol: final window from each test engine.
        print(f"---> Evaluating {name.upper()} on final test windows (official protocol)...")
        y_pred = model.predict(X_test, verbose=0).ravel()
        last_window_metrics = compute_metrics(y_test, y_pred)
        y_pred_all_windows = model.predict(X_test_all_windows, verbose=0).ravel()
        all_window_metrics = compute_metrics(y_test_all_windows, y_pred_all_windows)
        print(
            f"     Last-window RMSE: {last_window_metrics['rmse']:.4f} | "
            f"NASA score: {last_window_metrics['nasa_score']:.4f}"
        )
        if name in ("multibranch_lstm", "multibranch_noattn"):
            attention_model = get_group_attention_model(model)
            mean_weights = attention_model.predict(X_test, verbose=0).mean(axis=0)
            weights_text = ", ".join(
                f"{group}={weight:.3f}"
                for group, weight in zip(SENSOR_GROUPS, mean_weights)
            )
            print(f"     Mean group attention: {weights_text}")

        results.append({
            "subset": subset,
            "model": name,
            "seed": seed,
            "last_window_mae": last_window_metrics["mae"],
            "last_window_rmse": last_window_metrics["rmse"],
            "last_window_r2": last_window_metrics["r2"],
            "last_window_nasa_score": last_window_metrics["nasa_score"],
            "all_window_mae": all_window_metrics["mae"],
            "all_window_rmse": all_window_metrics["rmse"],
            "all_window_r2": all_window_metrics["r2"],
            "all_window_nasa_score": all_window_metrics["nasa_score"],
            "epochs_trained": epochs_trained,
            "training_seconds": round(t_elapsed, 2),
        })

    # 4. Compare and select best
    df_results = pd.DataFrame(results)

    # Normalized composite score (lower is better)
    for col in ["last_window_rmse", "last_window_mae", "last_window_nasa_score"]:
        col_min = df_results[col].min()
        col_max = df_results[col].max()
        denom = col_max - col_min if col_max != col_min else 1.0
        df_results[f"norm_{col}"] = (df_results[col] - col_min) / denom

    df_results["composite_score"] = (
        0.4 * df_results["norm_last_window_rmse"]
        + 0.3 * df_results["norm_last_window_mae"]
        + 0.3 * df_results["norm_last_window_nasa_score"]
    )
    df_results = df_results.sort_values("composite_score").reset_index(drop=True)
    best_model_name = df_results.iloc[0]["model"]

    # 5. Append completed runs and persist the best model from this workflow.
    os.makedirs(LOGS_DIR, exist_ok=True)
    csv_path = os.path.join(LOGS_DIR, "model_comparison.csv")
    best_path = os.path.join(LOGS_DIR, "best_model.txt")
    append_comparison_rows(csv_path, df_results.to_dict("records"))
    with open(best_path, "w") as f:
        f.write(best_model_name)

    print("\n" + "=" * 70)
    print(" MODEL EVALUATION SUMMARY")
    print("=" * 70)
    cols = [
        "model", "last_window_rmse", "last_window_nasa_score",
        "all_window_rmse", "all_window_nasa_score", "epochs_trained",
        "training_seconds", "composite_score",
    ]
    print(df_results[cols].to_string(index=False, float_format="{:.4f}".format))
    print("-" * 70)
    print(f"★ Selected Best Model: {best_model_name.upper()}")
    print(f"• Checkpoint saved:    {os.path.join(MODELS_DIR, f'{best_model_name}_best.h5')}")
    print(f"• Results saved:       {csv_path}")

    # 6. Verify MC-Dropout uncertainty on test sample
    print("\nStep 3: Verifying MC-Dropout Uncertainty Estimation...")
    sample_X = X_test[:1]
    best_model = models_to_train[best_model_name]
    pred_res = predict_rul(best_model, sample_X)
    health = get_health_state(pred_res["predicted_rul"])

    print(f"  ✓ Sample Predicted RUL:  {pred_res['predicted_rul']:.2f} cycles")
    print(f"  ✓ Uncertainty (std):     ±{pred_res['uncertainty']:.2f} cycles")
    print(f"  ✓ 95% Confidence Bounds: [{pred_res['rul_lower']:.2f}, {pred_res['rul_upper']:.2f}] cycles")
    print(f"  ✓ Engine Health State:   {health['health_state']} (Risk: {health['risk_level']}, Score: {health['health_score']:.1f}%)")
    print("=" * 70)
    print("\n✅ Training and verification pipeline successfully completed!\n")
    return True


def main():
    args = parse_args()
    if args.compare_only:
        success = display_comparison_results()
        sys.exit(0 if success else 1)

    subsets = CMAPSS_SUBSETS if args.all_subsets else [args.subset]
    for s in subsets:
        success = run_training_workflow(
            subset=s,
            model_choice=args.model,
            epochs=args.epochs,
            quick=args.quick,
            seed=args.seed,
        )
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
