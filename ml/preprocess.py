"""
ml/preprocess.py
----------------
C-MAPSS preprocessing pipeline for the NASA Digital Twin turbofan project.

Handles data loading, feature engineering, normalization, sequence generation,
and end-to-end pipeline execution for all four C-MAPSS subsets (FD001–FD004).
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from config import (
    RAW_COLUMNS,
    CONSTANT_SENSORS,
    SELECTED_SENSORS,
    SEQUENCE_LENGTH,
    MAX_RUL,
    N_OPERATING_CLUSTERS,
    TEST_SIZE,
    RANDOM_STATE,
    DATA_DIR,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cmapss(subset: str = "FD001", data_dir: str = None):
    """
    Load a C-MAPSS subset from disk.

    Parameters
    ----------
    subset : str
        One of 'FD001', 'FD002', 'FD003', 'FD004'.
    data_dir : str or None
        Path to the directory containing the txt files.
        Falls back to DATA_DIR from config if None.

    Returns
    -------
    train_df : pd.DataFrame
    test_df  : pd.DataFrame
    rul_series : pd.Series  — ground-truth RUL for the last cycle of each
                              test engine (integer-indexed from 1).
    """
    if data_dir is None:
        data_dir = DATA_DIR

    def _read(filename):
        path = os.path.join(data_dir, filename)
        df = pd.read_csv(path, sep=r"\s+", header=None, names=RAW_COLUMNS)
        return df

    train_df = _read(f"train_{subset}.txt")
    test_df = _read(f"test_{subset}.txt")

    rul_path = os.path.join(data_dir, f"RUL_{subset}.txt")
    rul_series = pd.read_csv(
        rul_path, sep=r"\s+", header=None
    ).iloc[:, 0].reset_index(drop=True)
    # 1-indexed to match engine_id column
    rul_series.index = rul_series.index + 1

    return train_df, test_df, rul_series


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def remove_constant_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop sensor columns that carry no degradation signal (constant across all
    operating conditions).  The list is defined in config.CONSTANT_SENSORS.
    """
    cols_to_drop = [c for c in CONSTANT_SENSORS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def clip_rul(df: pd.DataFrame, max_rul: int = MAX_RUL) -> pd.DataFrame:
    """
    Apply piece-wise linear RUL capping.

    The true RUL is kept as-is for all cycles where RUL ≤ max_rul.
    For earlier cycles the RUL is clamped to max_rul, implementing the
    piece-wise linear (degradation starts only after a burn-in) assumption.
    """
    df = df.copy()
    df["RUL"] = df["RUL"].clip(upper=max_rul)
    return df


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_sensors(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols=None,
):
    """
    Fit MinMaxScaler on the training set (per sensor column) and transform
    both train and test.

    For multi-condition subsets (FD002/FD004) it is recommended to cluster
    operating conditions first and normalize within each cluster
    (see cluster_operating_conditions + _normalize_by_cluster).

    Parameters
    ----------
    train_df : pd.DataFrame
    test_df  : pd.DataFrame
    feature_cols : list[str] or None
        Sensor columns to normalize; defaults to SELECTED_SENSORS.

    Returns
    -------
    train_df, test_df : pd.DataFrame  (normalized in-place copies)
    scaler : fitted MinMaxScaler
    """
    if feature_cols is None:
        feature_cols = SELECTED_SENSORS

    scaler = MinMaxScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_df, test_df, scaler


def _normalize_by_cluster(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols,
):
    """
    Normalize sensor readings within each operating-condition cluster.
    Used for FD002 / FD004 which span multiple operating regimes.

    Returns (train_df, test_df, dict[cluster_id → scaler]).
    """
    scalers = {}
    train_df = train_df.copy()
    test_df = test_df.copy()

    for cluster in train_df["op_cluster"].unique():
        mask_train = train_df["op_cluster"] == cluster
        mask_test = test_df["op_cluster"] == cluster

        scaler = MinMaxScaler()
        train_df.loc[mask_train, feature_cols] = scaler.fit_transform(
            train_df.loc[mask_train, feature_cols]
        )
        if mask_test.any():
            test_df.loc[mask_test, feature_cols] = scaler.transform(
                test_df.loc[mask_test, feature_cols]
            )
        scalers[cluster] = scaler

    return train_df, test_df, scalers


# ---------------------------------------------------------------------------
# Operating-condition clustering (FD002 / FD004)
# ---------------------------------------------------------------------------

def cluster_operating_conditions(
    df: pd.DataFrame,
    n_clusters: int = N_OPERATING_CLUSTERS,
) -> pd.DataFrame:
    """
    Cluster rows by (op_set1, op_set2, op_set3) using KMeans and append an
    'op_cluster' integer column.  Only needed for multi-condition subsets
    FD002 and FD004.

    Parameters
    ----------
    df : pd.DataFrame  — must contain op_set1, op_set2, op_set3 columns
    n_clusters : int

    Returns
    -------
    df with 'op_cluster' column added (in-place copy).
    """
    df = df.copy()
    op_cols = ["op_set1", "op_set2", "op_set3"]
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    df["op_cluster"] = km.fit_predict(df[op_cols])
    return df


# ---------------------------------------------------------------------------
# Sequence generation
# ---------------------------------------------------------------------------

def make_sequences(
    df: pd.DataFrame,
    seq_len: int = SEQUENCE_LENGTH,
    feature_cols=None,
    last_only: bool = False,
):
    """
    Build sliding-window sequences over each engine's time series.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: engine_id, cycle, RUL, and all feature_cols.
    seq_len : int
        Number of consecutive cycles per sequence.
    feature_cols : list[str] or None
        Feature columns to include; defaults to SELECTED_SENSORS.

    Returns
    -------
    X : np.ndarray, shape (n_sequences, seq_len, n_features)
    y : np.ndarray, shape (n_sequences,)  — RUL at the last time-step
    """
    if feature_cols is None:
        feature_cols = SELECTED_SENSORS

    X_list, y_list = [], []

    for engine_id, group in df.groupby("engine_id"):
        group = group.sort_values("cycle")
        features = group[feature_cols].values  # (T, F)
        rul_vals = group["RUL"].values         # (T,)

        n_cycles = len(group)
        if n_cycles < seq_len:
            # Pad with the first row repeated
            pad_size = seq_len - n_cycles
            pad = np.repeat(features[:1], pad_size, axis=0)
            features = np.vstack([pad, features])
            rul_pad = np.full(pad_size, rul_vals[0])
            rul_vals = np.concatenate([rul_pad, rul_vals])
            n_cycles = seq_len

        starts = [n_cycles - seq_len] if last_only else range(n_cycles - seq_len + 1)
        for start in starts:
            end = start + seq_len
            X_list.append(features[start:end])
            y_list.append(rul_vals[end - 1])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


# ---------------------------------------------------------------------------
# Full end-to-end pipeline
# ---------------------------------------------------------------------------

def preprocess_pipeline(subset: str = "FD001", data_dir: str = None) -> dict:
    """
    Execute the complete preprocessing pipeline and return arrays ready for
    model training.

    Steps
    -----
    1. Load raw txt files.
    2. Remove constant sensors.
    3. Compute train RUL (max_cycle − current_cycle), clip at MAX_RUL.
    4. Cluster operating conditions for multi-condition subsets.
    5. Normalize sensors (per-cluster for FD002/FD004, global otherwise).
    6. Build test ground-truth RUL from RUL file.
    7. Make sliding-window sequences.
    8. Split train into train / val.

    Parameters
    ----------
    subset : str   one of 'FD001'–'FD004'
    data_dir : str or None

    Returns
    -------
    dict with keys:
        X_train, y_train, X_val, y_val, X_test, y_test,
        scaler, feature_cols, subset
    """
    # 1. Load
    train_df, test_df, rul_series = load_cmapss(subset, data_dir)

    # 2. Remove constant / low-variance sensors
    train_df = remove_constant_sensors(train_df)
    test_df = remove_constant_sensors(test_df)

    feature_cols = [c for c in SELECTED_SENSORS if c in train_df.columns]

    # 3. Compute train RUL and clip
    max_cycles = train_df.groupby("engine_id")["cycle"].max()
    train_df["RUL"] = train_df.apply(
        lambda r: max_cycles[r["engine_id"]] - r["cycle"], axis=1
    )
    train_df = clip_rul(train_df, MAX_RUL)

    # 4. Operating-condition clustering (multi-condition subsets)
    multi_condition = subset in ("FD002", "FD004")
    if multi_condition:
        train_df = cluster_operating_conditions(train_df, N_OPERATING_CLUSTERS)
        test_df = cluster_operating_conditions(test_df, N_OPERATING_CLUSTERS)

    # 5. Normalize
    if multi_condition:
        train_df, test_df, scaler = _normalize_by_cluster(
            train_df, test_df, feature_cols
        )
    else:
        train_df, test_df, scaler = normalize_sensors(
            train_df, test_df, feature_cols
        )

    # 6. Build test RUL column (ground truth = RUL at last cycle + cycles
    #    remaining, but here the RUL file gives remaining life at last cycle)
    last_cycles = test_df.groupby("engine_id")["cycle"].max().reset_index()
    last_cycles.columns = ["engine_id", "last_cycle"]
    test_df = test_df.merge(last_cycles, on="engine_id")

    def _test_rul(row):
        rul_at_last = rul_series.loc[row["engine_id"]]
        # RUL at current cycle = rul_at_last + (last_cycle − current_cycle)
        return rul_at_last + (row["last_cycle"] - row["cycle"])

    # Preserve raw RUL-file targets for official final-window evaluation.
    test_df["RUL"] = test_df.apply(_test_rul, axis=1)
    test_df = test_df.drop(columns=["last_cycle"])

    # 7. Split engines before making windows to prevent engine leakage.
    engine_ids = train_df["engine_id"].unique()
    train_engine_ids, val_engine_ids = train_test_split(
        engine_ids, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    train_split_df = train_df[train_df["engine_id"].isin(train_engine_ids)]
    val_split_df = train_df[train_df["engine_id"].isin(val_engine_ids)]
    X_train, y_train = make_sequences(train_split_df, SEQUENCE_LENGTH, feature_cols)
    X_val, y_val = make_sequences(val_split_df, SEQUENCE_LENGTH, feature_cols)
    X_test, y_test = make_sequences(
        test_df, SEQUENCE_LENGTH, feature_cols, last_only=True
    )
    X_test_all_windows, y_test_all_windows = make_sequences(
        test_df, SEQUENCE_LENGTH, feature_cols
    )

    # 8. Verify one final test window for every RUL-file entry.
    # Final-window evaluation must have one target per test engine.
    if len(y_test) != len(rul_series):
        raise ValueError(
            "Expected one final test window per RUL-file entry; "
            f"got {len(y_test)} windows for {len(rul_series)} RUL values."
        )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "X_test_all_windows": X_test_all_windows,
        "y_test_all_windows": y_test_all_windows,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "subset": subset,
    }


# ---------------------------------------------------------------------------
# Single-engine inference helper
# ---------------------------------------------------------------------------

def load_single_engine(
    engine_df: pd.DataFrame,
    scaler,
    seq_len: int = SEQUENCE_LENGTH,
    feature_cols=None,
) -> np.ndarray:
    """
    Prepare a single engine's sensor data for real-time inference.

    Takes the most recent `seq_len` cycles, normalizes them using the
    pre-fitted scaler, and returns a batched array ready for model.predict().

    Parameters
    ----------
    engine_df : pd.DataFrame
        Rows for ONE engine, sorted ascending by cycle.
        Must contain all columns in feature_cols.
    scaler : fitted MinMaxScaler (or dict of scalers for multi-condition)
    seq_len : int
    feature_cols : list[str] or None

    Returns
    -------
    X : np.ndarray, shape (1, seq_len, n_features)
    """
    if feature_cols is None:
        feature_cols = SELECTED_SENSORS

    engine_df = engine_df.sort_values("cycle").copy()

    # Normalize — handle both a single scaler and a cluster-dict of scalers
    if isinstance(scaler, dict):
        # Multi-condition: apply per-cluster scaler
        for cluster_id, sc in scaler.items():
            mask = engine_df["op_cluster"] == cluster_id
            if mask.any():
                engine_df.loc[mask, feature_cols] = sc.transform(
                    engine_df.loc[mask, feature_cols]
                )
    else:
        engine_df[feature_cols] = scaler.transform(engine_df[feature_cols])

    features = engine_df[feature_cols].values  # (T, F)
    T = len(features)

    if T >= seq_len:
        # Take the most recent window
        window = features[-seq_len:]
    else:
        # Pad from the left with the first available reading
        pad_size = seq_len - T
        pad = np.repeat(features[:1], pad_size, axis=0)
        window = np.vstack([pad, features])

    X = window[np.newaxis, :, :].astype(np.float32)  # (1, seq_len, F)
    return X
