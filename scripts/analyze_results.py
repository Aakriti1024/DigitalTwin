#!/usr/bin/env python3
"""Analyze saved FD001 LSTM models without retraining."""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    DATA_DIR,
    MAX_RUL,
    MODELS_DIR,
    RANDOM_STATE,
    SELECTED_SENSORS,
    SENSOR_GROUPS,
    SEQUENCE_LENGTH,
    TEST_SIZE,
)
from ml.models import build_multibranch_lstm, get_group_attention_model
from ml.predict import mc_dropout_predict
from ml.preprocess import load_cmapss, preprocess_pipeline, remove_constant_sensors
from sklearn.model_selection import train_test_split


INTERVAL_Z_SCORES = {"80": 1.2815515655446004, "90": 1.6448536269514722}
RESEARCH_RESULTS_DIR = os.path.join(PROJECT_ROOT, "research_results")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze saved FD001 RUL models")
    parser.add_argument("--mc-samples", type=int, default=50)
    parser.add_argument("--engines", type=int, default=5)
    parser.add_argument("--cycles", type=int, default=60)
    parser.add_argument("--conformal-seed", type=int, default=42)
    parser.add_argument("--truncations-per-engine", type=int, default=20)
    parser.add_argument(
        "--skip-mc",
        action="store_true",
        help="Skip MC-dropout analysis and run deterministic/attention analysis only",
    )
    return parser.parse_args()


def load_saved_model(filename):
    """Load a Keras H5 model, including Lambda layers in the multi-branch model."""
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Saved model not found: {path}")
    try:
        return tf.keras.models.load_model(path, compile=False, safe_mode=False)
    except TypeError:
        if filename != "multibranch_lstm_best.h5":
            raise
        # Older H5 Lambda serialization cannot be deserialized by Keras 3.
        # Rebuild the current architecture and restore the same saved weights.
        model = build_multibranch_lstm()
        model.load_weights(path)
        return model


def make_last_cycle_windows(test_df, scaler, engine_ids, cycles):
    """Create rolling 30-cycle inputs ending at each engine's last ``cycles`` rows."""
    windows = []
    window_engine_ids = []
    window_cycles = []

    for engine_id in engine_ids:
        engine_df = test_df[test_df["engine_id"] == engine_id].sort_values("cycle")
        if len(engine_df) < cycles:
            raise ValueError(
                f"Engine {engine_id} has {len(engine_df)} cycles; expected at least {cycles}."
            )
        for end_index in range(len(engine_df) - cycles + 1, len(engine_df) + 1):
            prefix = engine_df.iloc[:end_index].copy()
            prefix[SELECTED_SENSORS] = scaler.transform(prefix[SELECTED_SENSORS])
            features = prefix[SELECTED_SENSORS].to_numpy(dtype=np.float32)
            windows.append(features[-30:])
            window_engine_ids.append(int(engine_id))
            window_cycles.append(int(prefix["cycle"].iloc[-1]))

    return np.asarray(windows, dtype=np.float32), window_engine_ids, window_cycles


def save_attention_plot(
    weights, selected_engine_ids, window_engine_ids, window_cycles, output_path
):
    """Plot four group weights over each selected engine's last cycles."""
    group_names = list(SENSOR_GROUPS)
    fig, axes = plt.subplots(
        len(selected_engine_ids), 1,
        figsize=(12, 2.4 * len(selected_engine_ids)), sharey=True,
    )
    axes = np.atleast_1d(axes)
    for axis, engine_id in zip(axes, selected_engine_ids):
        mask = np.asarray(window_engine_ids) == engine_id
        for index, group_name in enumerate(group_names):
            axis.plot(
                np.asarray(window_cycles)[mask], weights[mask, index], label=group_name
            )
        axis.set_title(f"Engine {engine_id}: final {mask.sum()} cycles")
        axis.set_ylabel("Group weight")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.3)
    axes[-1].set_xlabel("Cycle")
    axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.35))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def interval_metrics(y_true, mean_pred, std_pred):
    """Calculate empirical coverage and average interval width for two intervals."""
    metrics = {}
    for level, z_score in INTERVAL_Z_SCORES.items():
        lower = mean_pred - z_score * std_pred
        upper = mean_pred + z_score * std_pred
        metrics[level] = {
            "empirical_coverage": float(np.mean((y_true >= lower) & (y_true <= upper))),
            "mean_width": float(np.mean(upper - lower)),
        }
    return metrics


def conformal_half_widths(y_true, y_pred):
    """Return finite-sample-corrected split-conformal error quantiles."""
    absolute_errors = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    sorted_errors = np.sort(absolute_errors)
    half_widths = {}
    for level in (0.80, 0.90):
        rank = min(int(np.ceil((len(sorted_errors) + 1) * level)), len(sorted_errors))
        half_widths[str(int(level * 100))] = {
            "half_width": float(sorted_errors[rank - 1]),
            "quantile_rank": rank,
            "quantile_level": rank / len(sorted_errors),
        }
    return half_widths


def conformal_metrics(y_true, y_pred, half_width):
    """Evaluate a clipped-lower split-conformal interval on test predictions."""
    lower = np.maximum(y_pred - half_width, 0)
    upper = y_pred + half_width
    return {
        "empirical_coverage": float(np.mean((y_true >= lower) & (y_true <= upper))),
        "mean_width": float(np.mean(upper - lower)),
    }


def build_test_like_validation_set(train_df, scaler, samples_per_engine, seed):
    """Sample fixed-count random truncation windows from training validation engines."""
    train_df = remove_constant_sensors(train_df).copy()
    max_cycles = train_df.groupby("engine_id")["cycle"].max()
    train_df["RUL"] = train_df.apply(
        lambda row: min(max_cycles[row["engine_id"]] - row["cycle"], MAX_RUL), axis=1
    )
    engine_ids = train_df["engine_id"].unique()
    _, validation_engine_ids = train_test_split(
        engine_ids, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )

    rng = np.random.default_rng(seed)
    windows, targets = [], []
    for engine_id in validation_engine_ids:
        engine_df = train_df[train_df["engine_id"] == engine_id].sort_values("cycle")
        features = scaler.transform(engine_df[SELECTED_SENSORS]).astype(np.float32)
        eligible_end_indices = np.arange(SEQUENCE_LENGTH - 1, len(engine_df))
        selected_end_indices = rng.choice(
            eligible_end_indices,
            size=min(samples_per_engine, len(eligible_end_indices)),
            replace=False,
        )
        for end_index in selected_end_indices:
            windows.append(features[end_index - SEQUENCE_LENGTH + 1:end_index + 1])
            targets.append(engine_df["RUL"].iloc[end_index])
    return np.asarray(windows, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def evaluate_conformal_models(models, calibration_sets, X_test, target_sets):
    """Evaluate all calibration-set and target variants for saved models."""
    report = {
        "calibration_sample_sizes": {
            name: int(len(y_calibration))
            for name, (_, y_calibration) in calibration_sets.items()
        },
        "models": {},
    }
    rows = []
    for name, model in models:
        test_predictions = model.predict(X_test, verbose=0).ravel()
        model_report = {}
        for calibration_name, (X_calibration, y_calibration) in calibration_sets.items():
            calibration_predictions = model.predict(X_calibration, verbose=0).ravel()
            half_widths = conformal_half_widths(y_calibration, calibration_predictions)
            calibration_report = {}
            for target_name, y_target in target_sets.items():
                target_report = {}
                for level, details in half_widths.items():
                    metrics = conformal_metrics(
                        y_target, test_predictions, details["half_width"]
                    )
                    target_report[level] = {**details, **metrics}
                    rows.append({
                        "model": name,
                        "calibration_set": calibration_name,
                        "target": target_name,
                        "level": f"{level}%",
                        **target_report[level],
                    })
                calibration_report[target_name] = target_report
            model_report[calibration_name] = calibration_report
        report["models"][name] = model_report
    return report, rows


def main():
    args = parse_args()
    if min(args.engines, args.cycles, args.mc_samples, args.truncations_per_engine) <= 0:
        raise ValueError("Analysis counts and --truncations-per-engine must be positive.")

    os.makedirs(RESEARCH_RESULTS_DIR, exist_ok=True)

    # Official one-final-window-per-engine FD001 evaluation set.
    data = preprocess_pipeline(subset="FD001", data_dir=DATA_DIR)
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    lstm = load_saved_model("lstm_best.h5")
    multibranch = load_saved_model("multibranch_lstm_best.h5")

    # Choose test engines with enough history to plot a full final 60 cycles.
    raw_train_df, raw_test_df, _ = load_cmapss("FD001", DATA_DIR)
    raw_test_df = remove_constant_sensors(raw_test_df)
    eligible_engines = (
        raw_test_df.groupby("engine_id").size().loc[lambda counts: counts >= args.cycles]
    )
    selected_engines = eligible_engines.index.to_list()[:args.engines]
    if len(selected_engines) != args.engines:
        raise ValueError(f"Only {len(selected_engines)} test engines have {args.cycles} cycles.")

    attention_windows, attention_engine_ids, attention_cycles = make_last_cycle_windows(
        raw_test_df, data["scaler"], selected_engines, args.cycles
    )
    attention_model = get_group_attention_model(multibranch)
    group_weights = attention_model.predict(attention_windows, verbose=0)
    attention_path = os.path.join(RESEARCH_RESULTS_DIR, "fd001_group_attention_last60.png")
    save_attention_plot(
        group_weights,
        selected_engines,
        attention_engine_ids,
        attention_cycles,
        attention_path,
    )

    if not args.skip_mc:
        report = {"subset": "FD001", "mc_samples": args.mc_samples, "models": {}}
        for name, model in (("lstm", lstm), ("multibranch_lstm", multibranch)):
            mean_pred, std_pred = mc_dropout_predict(model, X_test, args.mc_samples)
            report["models"][name] = interval_metrics(y_test, mean_pred, std_pred)
            print(f"{name}:")
            for level, values in report["models"][name].items():
                print(
                    f"  {level}% interval | coverage={values['empirical_coverage']:.3f} "
                    f"| mean_width={values['mean_width']:.3f}"
                )

        report_path = os.path.join(RESEARCH_RESULTS_DIR, "fd001_interval_calibration.json")
        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2)
        print(f"Saved interval report: {report_path}")
    print(f"Saved attention plot: {attention_path}")

    X_test_like, y_test_like = build_test_like_validation_set(
        raw_train_df,
        data["scaler"],
        args.truncations_per_engine,
        args.conformal_seed,
    )
    calibration_sets = {
        "all": (X_val, y_val),
        "test_like": (X_test_like, y_test_like),
    }
    target_sets = {
        "raw": y_test,
        "capped": np.minimum(y_test, MAX_RUL),
    }
    conformal_report, conformal_rows = evaluate_conformal_models(
        (("lstm", lstm), ("multibranch_lstm", multibranch)),
        calibration_sets,
        X_test,
        target_sets,
    )
    targets_capped = bool(np.max(y_test) <= MAX_RUL)
    conformal_report["subset"] = "FD001"
    conformal_report["test_like_calibration"] = {
        "seed": args.conformal_seed,
        "truncations_per_validation_engine": args.truncations_per_engine,
        "method": (
            "Each engine from the training validation split contributes a fixed "
            "number of seeded random truncation windows."
        ),
    }
    conformal_report["test_target_capping"] = {
        "max_rul": MAX_RUL,
        "targets_capped_at_max_rul": targets_capped,
        "maximum_test_target": float(np.max(y_test)),
        "method": (
            "Test targets are raw RUL_FD001.txt last-cycle values; "
            "preprocess_pipeline does not cap test RUL."
        ),
    }
    print("\nSplit-conformal calibration sample sizes: " + ", ".join(
        f"{name}={size}" for name, size in conformal_report["calibration_sample_sizes"].items()
    ))
    print(
        "Test targets capped at MAX_RUL "
        f"({MAX_RUL}): {targets_capped}; maximum target={np.max(y_test):.1f}."
    )
    print("Test targets are raw RUL_FD001.txt last-cycle values; lower bounds are clipped at 0.")
    print("\nSplit-conformal intervals")
    print("model                 calibration  target  level  coverage  mean-width")
    for row in conformal_rows:
        print(
            f"{row['model']:<21} {row['calibration_set']:<12} "
            f"{row['target']:<7} {row['level']:<5} "
            f"{row['empirical_coverage']:>9.3f} {row['mean_width']:>11.3f}"
        )
    conformal_path = os.path.join(RESEARCH_RESULTS_DIR, "fd001_conformal_v2.json")
    with open(conformal_path, "w", encoding="utf-8") as conformal_file:
        json.dump(conformal_report, conformal_file, indent=2)
    print(f"Saved conformal report: {conformal_path}")


if __name__ == "__main__":
    main()
