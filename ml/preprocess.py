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
import joblib

from config import (
    RAW_COLUMNS,
    CONSTANT_SENSORS,
    SELECTED_SENSORS,
    SEQUENCE_LENGTH,
    MAX_RUL,
    N_OPERATING_CLUSTERS,
    TEST_SIZE,
    VALIDATION_SIZE,
    RANDOM_STATE,
    DATA_DIR,
    MODELS_DIR,
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
    train_df[feature_cols] = train_df[feature_cols].astype(np.float32)
    test_df[feature_cols] = test_df[feature_cols].astype(np.float32)

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
    train_df: pd.DataFrame,
    test_df: pd.DataFrame = None,
    n_clusters: int = N_OPERATING_CLUSTERS,
):
    """
    Cluster rows by (op_set1, op_set2, op_set3) using KMeans on normalized settings.
    Fits strictly on train_df and applies cluster labels to test_df if provided.
    """
    train_df = train_df.copy()
    op_cols = ["op_set1", "op_set2", "op_set3"]
    op_scaler = MinMaxScaler()
    train_op_scaled = op_scaler.fit_transform(train_df[op_cols])
    km = KMeans(n_clusters=n_clusters,random_state=RANDOM_STATE,n_init=10, init="random")
    train_df["op_cluster"] = km.fit_predict(train_op_scaled)

    if test_df is not None:
        test_df = test_df.copy()
        test_op_scaled = op_scaler.transform(test_df[op_cols])
        test_df["op_cluster"] = km.predict(test_op_scaled)
        return train_df, test_df, km, op_scaler

    return train_df, km, op_scaler


# ---------------------------------------------------------------------------
# Sequence generation
# ---------------------------------------------------------------------------

def make_sequences(
    df: pd.DataFrame,
    seq_len: int = SEQUENCE_LENGTH,
    feature_cols=None,
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

        for start in range(n_cycles - seq_len + 1):
            end = start + seq_len
            X_list.append(features[start:end])
            y_list.append(rul_vals[end - 1])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def make_test_sequences(
    df: pd.DataFrame,
    rul_series: pd.Series,
    seq_len: int = SEQUENCE_LENGTH,
    feature_cols=None,
    max_rul: int = MAX_RUL,
):
    """
    Build standard NASA C-MAPSS benchmark test sequences.

    For each test engine, takes the final `seq_len` cycles leading up to the
    cutoff cycle, with true RUL from `rul_series` (piecewise capped at `max_rul`).

    Returns
    -------
    X : np.ndarray, shape (n_engines, seq_len, n_features)
    y : np.ndarray, shape (n_engines,)
    engine_ids : list
    """
    if feature_cols is None:
        feature_cols = SELECTED_SENSORS

    X_list, y_list, engine_ids = [], [], []

    for engine_id, group in df.groupby("engine_id"):
        group = group.sort_values("cycle")
        features = group[feature_cols].values
        n_cycles = len(features)

        if n_cycles < seq_len:
            pad = np.repeat(features[:1], seq_len - n_cycles, axis=0)
            features = np.vstack([pad, features])

        window = features[-seq_len:]
        true_rul = float(rul_series.loc[engine_id])
        if max_rul is not None:
            true_rul = min(true_rul, float(max_rul))

        X_list.append(window)
        y_list.append(true_rul)
        engine_ids.append(engine_id)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, engine_ids


# ---------------------------------------------------------------------------
# Full end-to-end pipeline
# ---------------------------------------------------------------------------

def preprocess_pipeline(subset: str = "FD001", data_dir: str = None) -> dict:
    """
    Execute the complete preprocessing pipeline and return arrays ready for
    model training and standard benchmark evaluation without data leakage.

    Steps
    -----
    1. Load raw txt files.
    2. Remove constant sensors.
    3. Compute train RUL (max_cycle − current_cycle), clip at MAX_RUL.
    4. Split train engines into train and val engines (zero sequence leakage).
    5. Cluster operating conditions on train and project onto val and test.
    6. Normalize sensors (fit strictly on train split).
    7. Generate sliding sequences for train and val.
    8. Generate standard benchmark test sequences (last window of each engine).

    Parameters
    ----------
    subset : str   one of 'FD001'–'FD004'
    data_dir : str or None

    Returns
    -------
    dict with keys:
        X_train, y_train, X_val, y_val, X_test, y_test,
        X_test_all, y_test_all, test_engine_ids,
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

    # 4. Engine-level Train / Val split to prevent temporal leakage
    unique_engines = train_df["engine_id"].unique()
    train_engs, val_engs = train_test_split(
        unique_engines,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )
    train_split_df = train_df[train_df["engine_id"].isin(train_engs)].copy()
    val_split_df = train_df[train_df["engine_id"].isin(val_engs)].copy()

   
    # 5. Operating-condition clustering (multi-condition subsets)
    multi_condition = subset in ("FD002", "FD004")

    if multi_condition:
        train_split_df, test_df, km, op_scaler = cluster_operating_conditions(
            train_split_df, test_df, N_OPERATING_CLUSTERS
        )

        op_cols = ["op_set1", "op_set2", "op_set3"]

        # Use the same scaling used to fit K-Means.
        val_op_scaled = op_scaler.transform(val_split_df[op_cols])
        val_split_df["op_cluster"] = km.predict(val_op_scaled)

        train_op_scaled = op_scaler.transform(train_df[op_cols])
        train_df["op_cluster"] = km.predict(train_op_scaled)

        # Allow scaled sensor values to be stored as floats.
        test_df[feature_cols] = test_df[feature_cols].astype(np.float32)


    # 6. Normalize sensors (fit strictly on train split)
    if multi_condition:
        train_split_df, val_split_df, scaler = _normalize_by_cluster(
            train_split_df, val_split_df, feature_cols
        )
        # Apply cluster scalers to test_df
        for cluster_id, sc in scaler.items():
            mask = test_df["op_cluster"] == cluster_id
            if mask.any():
                test_df.loc[mask, feature_cols] = sc.transform(test_df.loc[mask, feature_cols])
    else:
        scaler = MinMaxScaler()
        train_split_df[feature_cols] = scaler.fit_transform(train_split_df[feature_cols])
        val_split_df[feature_cols] = scaler.transform(val_split_df[feature_cols])
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # 7. Build test ground-truth for all cycles (for trajectory simulation)
    last_cycles = test_df.groupby("engine_id")["cycle"].max().reset_index()
    last_cycles.columns = ["engine_id", "last_cycle"]
    test_df_all = test_df.merge(last_cycles, on="engine_id")

    def _test_rul(row):
        rul_at_last = rul_series.loc[row["engine_id"]]
        return rul_at_last + (row["last_cycle"] - row["cycle"])

    test_df_all["RUL"] = test_df_all.apply(_test_rul, axis=1).clip(upper=MAX_RUL)
    test_df_all = test_df_all.drop(columns=["last_cycle"])

    # 8. Sequences
    X_train, y_train = make_sequences(train_split_df, SEQUENCE_LENGTH, feature_cols)
    X_val, y_val = make_sequences(val_split_df, SEQUENCE_LENGTH, feature_cols)

    # Standard C-MAPSS benchmark test set (last sequence of each engine)
    X_test, y_test, test_engine_ids = make_test_sequences(
        test_df, rul_series, SEQUENCE_LENGTH, feature_cols, MAX_RUL
    )

    # All-cycles test sequences (for trajectory replay)
    X_test_all, y_test_all = make_sequences(test_df_all, SEQUENCE_LENGTH, feature_cols)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "X_test_all": X_test_all,
        "y_test_all": y_test_all,
        "test_engine_ids": test_engine_ids,
        "scaler": scaler,
        "km": km if multi_condition else None,
        "op_scaler": op_scaler if multi_condition else None,
        "feature_cols": feature_cols,
        "subset": subset,
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Preprocessing Artifact Persistence (Save / Load)
# ---------------------------------------------------------------------------

def save_preprocessor(
    preprocessor: dict,
    subset: str = "FD001",
    save_dir: str = None,
) -> str:
    """
    Save fitted preprocessing artifacts to disk using joblib.

    For FD001/FD003: preserves the single MinMaxScaler.
    For FD002/FD004: preserves the fitted KMeans model, operational settings
    MinMaxScaler, and cluster-specific sensor MinMaxScaler dictionary.

    Parameters
    ----------
    preprocessor : dict
        Dictionary containing 'scaler'. For multi-condition subsets, also
        contains 'km' and 'op_scaler'.
    subset : str
        C-MAPSS subset identifier ('FD001', 'FD002', 'FD003', 'FD004').
    save_dir : str or None
        Destination directory. Defaults to config.MODELS_DIR.

    Returns
    -------
    save_path : str
        Path to the saved joblib file.
    """
    if save_dir is None:
        save_dir = MODELS_DIR
    os.makedirs(save_dir, exist_ok=True)

    multi_condition = subset in ("FD002", "FD004")
    feature_cols = preprocessor.get("feature_cols", SELECTED_SENSORS)

    artifact = {
        "subset": subset,
        "feature_cols": list(feature_cols),
        "scaler": preprocessor["scaler"],
        "km": preprocessor.get("km"),
        "op_scaler": preprocessor.get("op_scaler"),
        "multi_condition": multi_condition,
    }

    save_path = os.path.join(save_dir, f"preprocessor_{subset}.joblib")
    joblib.dump(artifact, save_path)
    return save_path


def load_preprocessor(
    subset: str = "FD001",
    save_dir: str = None,
) -> dict:
    """
    Load fitted preprocessing artifacts from disk.

    Parameters
    ----------
    subset : str
        C-MAPSS subset identifier ('FD001', 'FD002', 'FD003', 'FD004').
    save_dir : str or None
        Directory where artifacts are stored. Defaults to config.MODELS_DIR.

    Returns
    -------
    preprocessor : dict
        Dictionary with keys:
            'subset', 'feature_cols', 'scaler', 'km', 'op_scaler', 'multi_condition'
    """
    if save_dir is None:
        save_dir = MODELS_DIR

    save_path = os.path.join(save_dir, f"preprocessor_{subset}.joblib")
    if not os.path.exists(save_path):
        fallback_path = os.path.join(save_dir, f"scaler_{subset}.joblib")
        if os.path.exists(fallback_path):
            save_path = fallback_path
        else:
            raise FileNotFoundError(
                f"Preprocessor artifact for subset '{subset}' not found at '{save_path}'."
            )

    artifact = joblib.load(save_path)
    return artifact


# ---------------------------------------------------------------------------
# Single-engine inference helper
# ---------------------------------------------------------------------------

def load_single_engine(
    engine_df: pd.DataFrame,
    scaler,
    seq_len: int = SEQUENCE_LENGTH,
    feature_cols=None,
    km=None,
    op_scaler=None,
) -> np.ndarray:
    """
    Prepare a single engine's sensor data for real-time inference without data leakage.

    Strictly preserves:
    - Exactly 14 sensor features in identical order (SELECTED_SENSORS)
    - Sliding sequence length of seq_len (defaults to 30)
    - Shape (1, seq_len, 14)
    - Does NOT pass operational settings into the model as extra sensor features
    - Reuses the fitted training scaler(s) and KMeans model (no refitting on live data)

    Parameters
    ----------
    engine_df : pd.DataFrame
        Rows for ONE engine, sorted ascending by cycle.
    scaler : fitted MinMaxScaler, dict of cluster scalers, or preprocessor dict
    seq_len : int
    feature_cols : list[str] or None
    km : fitted KMeans or None
    op_scaler : fitted MinMaxScaler for operational settings or None

    Returns
    -------
    X : np.ndarray, shape (1, seq_len, 14)
    """
    # Unpack preprocessor bundle if provided
    if isinstance(scaler, dict) and "scaler" in scaler:
        km = km or scaler.get("km")
        op_scaler = op_scaler or scaler.get("op_scaler")
        feature_cols = feature_cols or scaler.get("feature_cols")
        scaler = scaler["scaler"]

    if feature_cols is None:
        feature_cols = SELECTED_SENSORS

    engine_df = engine_df.copy()
    if "cycle" in engine_df.columns:
        engine_df = engine_df.sort_values("cycle")

    # Multi-condition subsets (cluster-specific normalization)
    if isinstance(scaler, dict):
        # Determine operating cluster if not already present
        if "op_cluster" not in engine_df.columns:
            op_cols = ["op_set1", "op_set2", "op_set3"]
            for i, col in enumerate(op_cols, start=1):
                if col not in engine_df.columns:
                    for alt in [f"setting_{i}", f"setting{i}", f"op_{i}"]:
                        if alt in engine_df.columns:
                            engine_df[col] = engine_df[alt]
                            break
                    if col not in engine_df.columns:
                        engine_df[col] = 0.0

            if op_scaler is not None and km is not None:
                op_scaled = op_scaler.transform(engine_df[op_cols])
                engine_df["op_cluster"] = km.predict(op_scaled)
            else:
                first_cluster = list(scaler.keys())[0]
                engine_df["op_cluster"] = first_cluster

        # Ensure sensor columns exist
        for col in feature_cols:
            if col not in engine_df.columns:
                engine_df[col] = 0.0

        engine_df[feature_cols] = engine_df[feature_cols].astype(np.float32)

        for cluster_id, sc in scaler.items():
            mask = engine_df["op_cluster"] == cluster_id
            if mask.any():
                engine_df.loc[mask, feature_cols] = sc.transform(
                    engine_df.loc[mask, feature_cols]
                )
    elif scaler is not None:
        # Single-condition subsets (FD001 / FD003)
        for col in feature_cols:
            if col not in engine_df.columns:
                engine_df[col] = 0.0
        engine_df[feature_cols] = engine_df[feature_cols].astype(np.float32)
        engine_df[feature_cols] = scaler.transform(engine_df[feature_cols])

    # Extract strictly the 14 sensor features
    features = engine_df[feature_cols].values
    T = len(features)

    if T >= seq_len:
        window = features[-seq_len:]
    else:
        pad_size = seq_len - T
        pad = np.repeat(features[:1], pad_size, axis=0)
        window = np.vstack([pad, features])

    X = window[np.newaxis, :, :].astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
    return X
