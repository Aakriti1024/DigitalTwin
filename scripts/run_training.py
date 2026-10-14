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
)
import numpy as np
import pandas as pd


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
        choices=["all", "lstm", "gru", "cnn_lstm", "tcn", "transformer", "multibranch_lstm"],
        help="Specific model architecture to train (default: all)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Number of training epochs (default: {EPOCHS})",
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


def run_training_workflow(subset="FD001", model_choice="all", epochs=None, quick=False):
    import tensorflow as tf
    from ml.preprocess import preprocess_pipeline, save_preprocessor
    from ml.models import (
        build_lstm,
        build_gru,
        build_cnn_lstm,
        build_tcn,
        build_transformer,
        get_all_models,
    )
    from ml.multibranch_lstm import build_multibranch_lstm
    from ml.evaluate import compute_metrics
    from ml.predict import mc_dropout_predict, predict_rul, get_health_state

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
    print(f"• Batch Size:       {BATCH_SIZE}")
    print("-" * 70)

    # 1. Preprocess data
    print("Step 1: Running C-MAPSS Preprocessing Pipeline...")
    t0 = time.time()
    data = preprocess_pipeline(subset=subset, data_dir=DATA_DIR)
    save_preprocessor(data, subset=subset)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
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
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="mse",
            metrics=["mae"],
        )
    elif model_choice == "multibranch_lstm":
        models_to_train = {
            "multibranch_lstm": build_multibranch_lstm(
                input_shape=input_shape,
                feature_cols=feature_cols,
            )
        }
    else:
        models_to_train = get_all_models(input_shape)

    # 3. Train models
    print("\nStep 2: Training Model(s)...")
    results = []

    for name, model in models_to_train.items():
        print(f"\n---> Training: {name.upper()}")
        if name == "multibranch_lstm":
            checkpoint_path = os.path.join(
                MODELS_DIR, "multibranch_lstm_lr0003_best.keras"
            )
        else:
            checkpoint_path = os.path.join(
               MODELS_DIR, f"{name}_best.keras"
            )

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

        # Also save legacy .h5 copy if supported
        if name != "transformer":
            try:
                if name == "multibranch_lstm":
                    h5_path = os.path.join(
                       MODELS_DIR, "multibranch_lstm_lr0003_best.h5"
                    )
                else:
                    h5_path = os.path.join(MODELS_DIR, f"{name}_best.h5")
                model.save(h5_path)
            except Exception:
                pass

        # Evaluate on test set & measure inference latency
        print(f"---> Evaluating {name.upper()} on benchmark test data...")
        t_inf_start = time.time()
        y_pred = model.predict(X_test, verbose=0).ravel()
        inf_latency_ms = round(((time.time() - t_inf_start) * 1000.0) / max(1, len(X_test)), 2)

        metrics = compute_metrics(y_test, y_pred)

        results.append({
            "model": name,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "nasa_score": metrics["nasa_score"],
            "training_time": round(t_elapsed, 2),
            "inference_time_ms": inf_latency_ms,
        })

    
    # 4. Compare results without overwriting the existing benchmark.
    os.makedirs(LOGS_DIR, exist_ok=True)

    if model_choice == "multibranch_lstm":
        # This run evaluates only the new model. Do not calculate a
        # misleading single-model composite score or select a new
        # production model from a one-model comparison.
        df_results = pd.DataFrame(results)

        metrics_path = os.path.join(
            LOGS_DIR,
            f"multibranch_lstm_lr0003_{subset.lower()}_metrics.csv",
        )
        df_results.to_csv(metrics_path, index=False)

        best_model_name = "multibranch_lstm"

        print("\n" + "=" * 70)
        print(" MULTIBRANCH LSTM EVALUATION")
        print("=" * 70)
        print(df_results.to_string(index=False))
        print("-" * 70)
        print(f"Checkpoint saved: {os.path.join(MODELS_DIR, 'multibranch_lstm_lr0003_best.keras')}")
        print(f"Metrics saved:    {metrics_path}")
        print("Existing best_model.txt and model_comparison.csv were preserved.")

    else:
        # Original comparison and best-model selection for existing models.
        df_results = pd.DataFrame(results)

        for col in ["rmse", "mae", "nasa_score"]:
            col_min = df_results[col].min()
            col_max = df_results[col].max()
            denom = col_max - col_min if col_max != col_min else 1.0
            df_results[f"norm_{col}"] = (df_results[col] - col_min) / denom

        df_results["composite_score"] = (
            0.4 * df_results["norm_rmse"]
            + 0.3 * df_results["norm_mae"]
            + 0.3 * df_results["norm_nasa_score"]
        )
        df_results = df_results.sort_values(
            "composite_score"
        ).reset_index(drop=True)

        best_model_name = df_results.iloc[0]["model"]

        csv_path = os.path.join(LOGS_DIR, "model_comparison.csv")
        best_path = os.path.join(LOGS_DIR, "best_model.txt")

        df_results.to_csv(csv_path, index=False)
        with open(best_path, "w") as f:
            f.write(best_model_name)

    
    print("\n" + "=" * 70)

    if model_choice == "multibranch_lstm":
        print(" MULTIBRANCH LSTM EXPERIMENT SUMMARY")
        print("=" * 70)

        cols = [
            "model",
            "mae",
            "rmse",
            "r2",
            "nasa_score",
            "training_time",
            "inference_time_ms",
        ]

        print(
            df_results[cols].to_string(
                index=False,
                float_format="{:.4f}".format,
            )
        )

        print("-" * 70)
        print("Experiment: learning rate 0.0003")
        print("Original production-model selection remains unchanged.")

        print(
            "Checkpoint saved: "
            + os.path.join(
                MODELS_DIR,
                "multibranch_lstm_lr0003_best.keras",
            )
        )

        print(
            "Metrics saved: "
            + os.path.join(
                LOGS_DIR,
                f"multibranch_lstm_lr0003_{subset.lower()}_metrics.csv",
            )
        )

    else:
        print(" MODEL EVALUATION SUMMARY")
        print("=" * 70)

        cols = [
            "model",
            "mae",
            "rmse",
            "r2",
            "nasa_score",
            "training_time",
            "composite_score",
        ]

        print(
            df_results[cols].to_string(
                index=False,
                float_format="{:.4f}".format,
            )
        )

        print("-" * 70)
        print(f"Selected Best Model: {best_model_name.upper()}")
        print(
            "Checkpoint saved: "
            + os.path.join(
                MODELS_DIR,
                f"{best_model_name}_best.h5",
            )
        )
        print(f"Results saved: {csv_path}")


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
        )
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
