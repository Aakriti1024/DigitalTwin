
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from config import SELECTED_SENSORS
from ml.preprocess import preprocess_pipeline
from config import SEQUENCE_LENGTH, MAX_RUL


SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

ARRAY_KEYS = [
    "X_train",
    "X_val",
    "X_test",
    "X_test_all",
]


def verify_subset(subset):
    print(f"\n{'=' * 50}")
    print(f"Checking {subset}")
    print("=" * 50)

    data = preprocess_pipeline(subset)

    # 1. Check the selected sensor features
    feature_cols = data["feature_cols"]
    assert list(feature_cols) == list(SELECTED_SENSORS), (
        f"{subset}: feature order does not match SELECTED_SENSORS"
    )
    assert len(feature_cols) == 14, (
        f"{subset}: expected 14 sensor features"
    )
    print("PASS: 14 sensor features and correct feature order")

    # 2. Check the input sequence arrays
    for key in ARRAY_KEYS:
        X = np.asarray(data[key])

        assert X.ndim == 3, (
            f"{subset}/{key}: expected a 3D array, got {X.shape}"
        )
        assert X.shape[1:] == (SEQUENCE_LENGTH, 14), (
            f"{subset}/{key}: unexpected shape {X.shape}"
        )
        assert X.shape[0] > 0, f"{subset}/{key}: array is empty"
        assert np.isfinite(X).all(), (
            f"{subset}/{key}: contains NaN or infinity"
        )

        print(f"PASS: {key} shape={X.shape}, all values finite")

    # 3. Check target labels
    for X_key, y_key in [
        ("X_train", "y_train"),
        ("X_val", "y_val"),
        ("X_test", "y_test"),
        ("X_test_all", "y_test_all"),
    ]:
        y = np.asarray(data[y_key]).reshape(-1)
        X = np.asarray(data[X_key])

        assert len(X) == len(y), (
            f"{subset}: {X_key} and {y_key} lengths differ"
        )
        assert np.isfinite(y).all(), (
            f"{subset}/{y_key}: contains NaN or infinity"
        )
        assert (y >= 0).all() and (y <= MAX_RUL).all(), (
            f"{subset}/{y_key}: RUL outside [0, {MAX_RUL}]"
        )

        print(f"PASS: {y_key} labels valid, range={y.min():.1f}–{y.max():.1f}")

    # 4. Check that preprocessing produced a scaler
    assert data.get("scaler") is not None, (
        f"{subset}: scaler is missing"
    )
    print("PASS: sensor scaler is available")

    # Show available outputs for inspection
    print("Returned fields:", sorted(data.keys()))
    print(f"RESULT: {subset} PASSED")


def main():
    print("TURBOFAN PREPROCESSING VERIFICATION")
    print("This test does not train any models.")

    for subset in SUBSETS:
        verify_subset(subset)

    print("\nALL FOUR PREPROCESSING PIPELINES PASSED.")


if __name__ == "__main__":
    main()