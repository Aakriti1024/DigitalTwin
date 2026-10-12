"""
tests/test_inference_preprocessing.py
-------------------------------------
Unit and integration tests for C-MAPSS preprocessing-to-inference pipeline.

Verifies:
1. Preprocessor artifact serialization (save/load) for all 4 subsets (FD001-FD004).
2. load_single_engine produces strictly finite arrays with shape (1, 30, 14).
3. Operating settings are not passed into the model as extra sensor features.
4. DigitalTwin backend integration runs inference with identical feature order,
   clustering, and normalization without refitting or NaN values.
5. Model predictions on preprocessed inputs are finite and bounded.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from config import (
    CMAPSS_SUBSETS,
    SELECTED_SENSORS,
    SEQUENCE_LENGTH,
    MAX_RUL,
    MODELS_DIR,
)
from ml.preprocess import (
    load_cmapss,
    preprocess_pipeline,
    save_preprocessor,
    load_preprocessor,
    load_single_engine,
)
from backend.digital_twin import DigitalTwin
from ml.predict import predict_rul, predict_engine_state


class TestInferencePreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Fit and save preprocessors for all 4 subsets to verify persistence."""
        import tensorflow as tf
        cls.subsets = ["FD001", "FD002", "FD003", "FD004"]
        cls.preprocessors = {}

        for subset in cls.subsets:
            # Run pipeline and save artifacts
            data = preprocess_pipeline(subset=subset)
            save_path = save_preprocessor(data, subset=subset)
            cls.assertTrue(os.path.exists(save_path), f"Failed to save preprocessor for {subset}")

            # Reload to verify consistent serialization
            loaded = load_preprocessor(subset=subset)
            cls.preprocessors[subset] = loaded

        # Load or create a lightweight model for inference verification
        model_keras_path = os.path.join(MODELS_DIR, "lstm_best.keras")
        model_h5_path = os.path.join(MODELS_DIR, "lstm_best.h5")
        if os.path.exists(model_keras_path):
            cls.model = tf.keras.models.load_model(model_keras_path, compile=False)
        elif os.path.exists(model_h5_path):
            cls.model = tf.keras.models.load_model(model_h5_path, compile=False)
        else:
            from ml.models import build_lstm
            cls.model = build_lstm((SEQUENCE_LENGTH, len(SELECTED_SENSORS)))
            cls.model.compile(optimizer="adam", loss="mse")

    def test_preprocessor_structure_all_subsets(self):
        """Verify preprocessor artifacts structure for all four subsets."""
        for subset in self.subsets:
            prep = self.preprocessors[subset]
            self.assertEqual(prep["subset"], subset)
            self.assertEqual(len(prep["feature_cols"]), 14)
            self.assertEqual(prep["feature_cols"], SELECTED_SENSORS)

            if subset in ("FD001", "FD003"):
                # Single scaler approach
                self.assertFalse(prep["multi_condition"])
                self.assertIsNone(prep["km"])
                self.assertIsNone(prep["op_scaler"])
                self.assertTrue(hasattr(prep["scaler"], "transform"))
            else:
                # Multi-condition: KMeans + cluster scalers
                self.assertTrue(prep["multi_condition"])
                self.assertIsNotNone(prep["km"])
                self.assertIsNotNone(prep["op_scaler"])
                self.assertIsInstance(prep["scaler"], dict)
                self.assertEqual(len(prep["scaler"]), 6)
                self.assertEqual(prep["km"].n_clusters, 6)

    def test_load_single_engine_shape_and_finite(self):
        """Verify load_single_engine produces finite (1, 30, 14) inputs for all subsets."""
        for subset in self.subsets:
            prep = self.preprocessors[subset]
            _, test_df, _ = load_cmapss(subset)
            engine_1 = test_df[test_df["engine_id"] == 1].copy()

            # 1. Full sequence (>= 30 cycles)
            X_full = load_single_engine(engine_1, scaler=prep, seq_len=SEQUENCE_LENGTH)
            self.assertEqual(
                X_full.shape,
                (1, SEQUENCE_LENGTH, 14),
                f"Subset {subset}: Expected shape (1, 30, 14), got {X_full.shape}",
            )
            self.assertTrue(
                np.isfinite(X_full).all(),
                f"Subset {subset}: X_full contains non-finite values (NaN or Inf)",
            )

            # 2. Short sequence (< 30 cycles) testing padding
            short_df = engine_1.iloc[:5].copy()
            X_short = load_single_engine(short_df, scaler=prep, seq_len=SEQUENCE_LENGTH)
            self.assertEqual(
                X_short.shape,
                (1, SEQUENCE_LENGTH, 14),
                f"Subset {subset}: Expected padded shape (1, 30, 14), got {X_short.shape}",
            )
            self.assertTrue(
                np.isfinite(X_short).all(),
                f"Subset {subset}: X_short contains non-finite values (NaN or Inf)",
            )

            # 3. Alternate op_setting names (setting_1, setting_2, setting_3)
            renamed_df = engine_1.rename(
                columns={"op_set1": "setting_1", "op_set2": "setting_2", "op_set3": "setting_3"}
            )
            X_renamed = load_single_engine(renamed_df, scaler=prep, seq_len=SEQUENCE_LENGTH)
            self.assertEqual(X_renamed.shape, (1, SEQUENCE_LENGTH, 14))
            self.assertTrue(np.isfinite(X_renamed).all())

    def test_digital_twin_inference_all_subsets(self):
        """Verify DigitalTwin backend inference works end-to-end for all subsets."""
        for subset in self.subsets:
            prep = self.preprocessors[subset]
            twin = DigitalTwin(f"test_twin_{subset}")
            twin.load_model(model=self.model, scaler=prep)
            self.assertFalse(twin.is_demo_mode)

            _, test_df, _ = load_cmapss(subset)
            engine_rows = test_df[test_df["engine_id"] == 1].sort_values("cycle")

            # Push 35 cycles of sensor data
            last_state = None
            for _, r in engine_rows.iloc[:35].iterrows():
                sensor_data = {s: float(r[s]) for s in SELECTED_SENSORS if s in r}
                op_settings = {
                    "setting_1": float(r["op_set1"]),
                    "setting_2": float(r["op_set2"]),
                    "setting_3": float(r["op_set3"]),
                }
                last_state = twin.update_with_sensor_data(
                    cycle=int(r["cycle"]),
                    sensor_data=sensor_data,
                    op_settings=op_settings,
                )

            self.assertIsNotNone(last_state)
            self.assertTrue(np.isfinite(last_state["predicted_rul"]))
            self.assertTrue(np.isfinite(last_state["uncertainty"]))
            self.assertTrue(np.isfinite(last_state["rul_lower"]))
            self.assertTrue(np.isfinite(last_state["rul_upper"]))
            self.assertGreaterEqual(last_state["predicted_rul"], 0.0)
            self.assertLessEqual(last_state["predicted_rul"], MAX_RUL + 50.0)
            self.assertIn(
                last_state["health_state"],
                ["Healthy", "Warning", "Critical", "Degraded", "Failure"],
            )

            # Check history retrieval with n_cycles parameter
            history = twin.get_history(n_cycles=20)
            self.assertEqual(len(history["rul_history"]), 20)
            self.assertIn("s2", history["sensor_trends"])

    def test_predict_engine_state_and_rul_all_subsets(self):
        """Verify ml.predict functions return finite values on all subsets."""
        for subset in self.subsets:
            prep = self.preprocessors[subset]
            _, test_df, _ = load_cmapss(subset)
            engine_1 = test_df[test_df["engine_id"] == 1].iloc[:40].copy()

            state = predict_engine_state(self.model, engine_1, scaler=prep)
            self.assertTrue(np.isfinite(state["predicted_rul"]))
            self.assertTrue(np.isfinite(state["uncertainty"]))
            self.assertTrue(np.isfinite(state["rul_lower"]))
            self.assertTrue(np.isfinite(state["rul_upper"]))
            self.assertGreaterEqual(state["predicted_rul"], 0.0)
            self.assertIn(
                state["health_state"],
                ["Healthy", "Warning", "Critical", "Degraded", "Failure"],
            )


if __name__ == "__main__":
    unittest.main()
