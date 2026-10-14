
"""
Multibranch LSTM with Dual Attention
Project: Autonomous Agentic Digital Twin for Turbofan Engines

Input: 30 cycles x 14 selected sensors
Output: Predicted Remaining Useful Life (RUL), in cycles

Attention 1: Temporal attention within each sensor branch.
Attention 2: Sensor-group attention across the branches.
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras import Model, Input, layers
from tensorflow.keras.initializers import Constant
from config import (
    SEQUENCE_LENGTH,
    N_FEATURES,
    SELECTED_SENSORS,
    DROPOUT_RATE,
)

MULTIBRANCH_LEARNING_RATE = 0.0003

# Sensor groupings based on the C-MAPSS sensor descriptions.
# These are engineering-inspired groups, not guaranteed
# independent physical subsystems.

SENSOR_GROUPS = {
    "temperature": ["s2", "s3", "s4"],
    "pressure": ["s7", "s11"],
    "shaft_speed": ["s8", "s9", "s13", "s14"],
    "airflow_ratio_bleed": ["s12", "s15", "s17", "s20", "s21"],
}


def select_sensors(inputs, feature_cols, sensor_names, name):
    """Select a branch's sensor columns without changing their order."""

    missing = [sensor for sensor in sensor_names
               if sensor not in feature_cols]

    if missing:
        raise ValueError(f"{name}: missing sensors {missing}")

    indices = [feature_cols.index(sensor) for sensor in sensor_names]

    # Fixed selection matrix: selects the requested sensor columns.
    selection_matrix = np.zeros(
        (len(feature_cols), len(indices)),
        dtype=np.float32,
    )

    for output_index, input_index in enumerate(indices):
        selection_matrix[input_index, output_index] = 1.0

    return layers.Dense(
        len(indices),
        use_bias=False,
        trainable=False,
        kernel_initializer=Constant(selection_matrix),
        name=f"{name}_sensor_selection",
    )(inputs)


def temporal_attention(sequence, name):
    """
    Learn which of the 30 time steps matter most
    for a particular sensor branch.
    """

    scores = layers.Dense(
        1,
        name=f"{name}_temporal_scores",
    )(sequence)

    weights = layers.Softmax(
        axis=1,
        name=f"{name}_temporal_weights",
    )(scores)

    # Shape: (batch, 1, time)
    weights = layers.Permute(
        (2, 1),
        name=f"{name}_temporal_transpose",
    )(weights)

    # Weighted combination of LSTM outputs over time.
    context = layers.Dot(
        axes=(2, 1),
        name=f"{name}_temporal_context",
    )([weights, sequence])

    return layers.Flatten(
        name=f"{name}_temporal_flatten",
    )(context)


def build_multibranch_lstm(
    input_shape=(SEQUENCE_LENGTH, N_FEATURES),
    feature_cols=None,
    dropout_rate=DROPOUT_RATE,
):
    """
    Build and compile the multibranch dual-attention RUL model.

    feature_cols must match the feature order used to construct
    the training sequences.
    """

    if feature_cols is None:
        feature_cols = list(SELECTED_SENSORS)
    else:
        feature_cols = list(feature_cols)

    if len(feature_cols) != input_shape[-1]:
        raise ValueError(
            f"Input has {input_shape[-1]} features, "
            f"but {len(feature_cols)} feature names were provided."
        )

    # Verify every input sensor belongs to exactly one branch.
    all_group_sensors = [
        sensor
        for group in SENSOR_GROUPS.values()
        for sensor in group
    ]

    if len(all_group_sensors) != len(set(all_group_sensors)):
        raise ValueError("A sensor appears in more than one branch.")

    if set(all_group_sensors) != set(feature_cols):
        raise ValueError(
            "Sensor groups do not match the model input features. "
            f"Missing: {sorted(set(feature_cols) - set(all_group_sensors))}; "
            f"unexpected: {sorted(set(all_group_sensors) - set(feature_cols))}"
        )

    inputs = Input(
        shape=input_shape,
        name="sensor_sequence",
    )

    branch_vectors = []

    # ---------------------------------------------------------
    # Four independent LSTM branches
    # ---------------------------------------------------------
    for group_name, sensor_names in SENSOR_GROUPS.items():

        group_input = select_sensors(
            inputs,
            feature_cols,
            sensor_names,
            group_name,
        )

        # Learn degradation patterns specific to this group.
        sequence = layers.LSTM(
            32,
            return_sequences=True,
            name=f"{group_name}_lstm",
        )(group_input)

        sequence = layers.LayerNormalization(
            name=f"{group_name}_normalization",
        )(sequence)

        sequence = layers.Dropout(
            dropout_rate,
            name=f"{group_name}_dropout",
        )(sequence)

        # Attention 1: select informative time steps.
        branch_vector = temporal_attention(
            sequence,
            group_name,
        )

        branch_vector = layers.Dense(
            32,
            activation="relu",
            name=f"{group_name}_projection",
        )(branch_vector)

        branch_vectors.append(
            layers.Reshape(
                (1, 32),
                name=f"{group_name}_branch_axis",
            )(branch_vector)
        )

    # Shape: (batch, 4, 32)
    combined_branches = layers.Concatenate(
        axis=1,
        name="combine_sensor_branches",
    )(branch_vectors)

    # ---------------------------------------------------------
    # Attention 2: learn the importance of each sensor group
    # ---------------------------------------------------------
    group_scores = layers.Dense(
        1,
        name="group_attention_scores",
    )(combined_branches)

    group_weights = layers.Softmax(
        axis=1,
        name="group_attention_weights",
    )(group_scores)

    # Shape: (batch, 1, 4)
    group_weights_transposed = layers.Permute(
        (2, 1),
        name="group_attention_transpose",
    )(group_weights)

    # Weighted combination of all four branches.
    group_context = layers.Dot(
        axes=(2, 1),
        name="group_attention_context",
    )([group_weights_transposed, combined_branches])

    combined = layers.Flatten(
        name="combined_context",
    )(group_context)

    # ---------------------------------------------------------
    # RUL regression head
    # ---------------------------------------------------------
    x = layers.Dense(
        32,
        activation="relu",
        name="rul_dense",
    )(combined)

    x = layers.Dropout(
        dropout_rate,
        name="rul_dropout",
    )(x)

    output = layers.Dense(
        1,
        activation="linear",
        name="rul_output",
    )(x)

    model = Model(
        inputs=inputs,
        outputs=output,
        name="Multibranch_LSTM_Dual_Attention",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=MULTIBRANCH_LEARNING_RATE,
        ),
        loss="mse",
        metrics=["mae"],
    )

    return model
