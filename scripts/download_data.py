#!/usr/bin/env python3
"""
scripts/download_data.py
------------------------
Validates that the NASA C-MAPSS dataset files (FD001–FD004) exist, are non-empty,
and correctly formatted for the Digital Twin ML pipeline.
"""

import sys
import os

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import DATA_DIR, CMAPSS_SUBSETS, RAW_COLUMNS
import pandas as pd


def check_dataset() -> bool:
    print("=" * 70)
    print(" NASA C-MAPSS Turbofan Engine Dataset Verification")
    print("=" * 70)
    print(f"Data Directory: {DATA_DIR}\n")

    expected_files = []
    for subset in CMAPSS_SUBSETS:
        expected_files.append((f"train_{subset}.txt", "train", subset))
        expected_files.append((f"test_{subset}.txt", "test", subset))
        expected_files.append((f"RUL_{subset}.txt", "rul", subset))

    all_present = True
    summary = []

    for filename, file_type, subset in expected_files:
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.isfile(filepath):
            all_present = False
            summary.append({
                "File": filename,
                "Status": "MISSING",
                "Size (KB)": 0,
                "Rows": 0,
                "Engines": 0,
            })
            continue

        size_kb = os.path.getsize(filepath) / 1024.0
        if size_kb < 0.1:
            all_present = False
            summary.append({
                "File": filename,
                "Status": "EMPTY",
                "Size (KB)": round(size_kb, 1),
                "Rows": 0,
                "Engines": 0,
            })
            continue

        try:
            if file_type in ("train", "test"):
                df = pd.read_csv(filepath, sep=r"\s+", header=None, names=RAW_COLUMNS)
                n_rows = len(df)
                n_engines = df["engine_id"].nunique()
            else:
                df = pd.read_csv(filepath, sep=r"\s+", header=None)
                n_rows = len(df)
                n_engines = len(df)

            summary.append({
                "File": filename,
                "Status": "OK",
                "Size (KB)": round(size_kb, 1),
                "Rows": n_rows,
                "Engines": n_engines,
            })
        except Exception as e:
            all_present = False
            summary.append({
                "File": filename,
                "Status": f"ERR: {str(e)[:15]}",
                "Size (KB)": round(size_kb, 1),
                "Rows": 0,
                "Engines": 0,
            })

    # Print summary table
    df_sum = pd.DataFrame(summary)
    print(df_sum.to_string(index=False))
    print("-" * 70)

    if all_present:
        print("\n✅ All 12 C-MAPSS dataset files are present, verified, and ready!")
        print("You can run training with:")
        print("   python scripts/run_training.py --subset FD001 --quick\n")
        return True
    else:
        print("\n⚠️ Some dataset files are missing or incomplete.")
        print("Please obtain the NASA C-MAPSS dataset:")
        print("1. NASA Open Data Portal: https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/")
        print("2. Kaggle Mirror: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps")
        print(f"Place the extracted .txt files into: {DATA_DIR}\n")
        return False


if __name__ == "__main__":
    success = check_dataset()
    sys.exit(0 if success else 1)
