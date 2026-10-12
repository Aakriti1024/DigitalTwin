"""
ml/models.py
------------
Keras model architectures for C-MAPSS RUL prediction.

All models use MC-Dropout: Dropout layers are active at inference time by
passing `training=True`, enabling Monte-Carlo uncertainty quantification.

Architectures provided
----------------------
* LSTM          — stacked LSTM
* GRU           — stacked GRU
* CNN-LSTM      — convolutional feature extractor + LSTM
* TCN           — Temporal Convolutional Network with dilated causal convs
* Transformer   — positional encoding + multi-head attention
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam

from config import (
    SEQUENCE_LENGTH,
    N_FEATURES,
    DROPOUT_RATE,
    LEARNING_RATE,
    SELECTED_SENSORS,
    SENSOR_GROUPS,
)


# ---------------------------------------------------------------------------
# Utility: positional encoding (sinusoidal, as in "Attention Is All You Need")
# ---------------------------------------------------------------------------

def positional_encoding(seq_len: int, d_model: int) -> tf.Tensor:
    """
    Compute sinusoidal positional encodings.

    Parameters
    ----------
    seq_len : int   — number of time steps
    d_model : int   — embedding / feature dimension

    Returns
    -------
    tf.Tensor of shape (1, seq_len, d_model)
    """
    positions = np.arange(seq_len)[:, np.newaxis]          # (T, 1)
    dims = np.arange(d_model)[np.newaxis, :]               # (1, D)

    # Divide terms: 10000^(2i/d_model)
    angles = positions / np.power(10_000, (2 * (dims // 2)) / np.float32(d_model))

    # Even indices → sin, odd → cos
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])

    pos_enc = tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)  # (1, T, D)
    return pos_enc


# ---------------------------------------------------------------------------
# 1. LSTM
# ---------------------------------------------------------------------------

def build_lstm(
    input_shape: tuple = (SEQUENCE_LENGTH, N_FEATURES),
    dropout_rate: float = DROPOUT_RATE,
) -> Model:
    """
    Two-layer stacked LSTM with MC-Dropout.

    Architecture
    ------------
    Input → LSTM(128, return_sequences=True) → Dropout → LSTM(64) →
    Dropout → Dense(32, relu) → Dense(1)
    """
    inp = Input(shape=input_shape, name="input")

    x = layers.LSTM(128, return_sequences=True, name="lstm_1")(inp)
    x = layers.Dropout(dropout_rate, name="drop_1")(x)

    x = layers.LSTM(64, return_sequences=False, name="lstm_2")(x)
    x = layers.Dropout(dropout_rate, name="drop_2")(x)

    x = layers.Dense(32, activation="relu", name="dense_1")(x)
    out = layers.Dense(1, name="rul_output")(x)

    model = Model(inputs=inp, outputs=out, name="LSTM")
    return model


# ---------------------------------------------------------------------------
# 2. Multi-branch LSTM
# ---------------------------------------------------------------------------

def _sensor_group_indices() -> dict:
    """Return indices into SELECTED_SENSORS for every configured group."""
    feature_index = {sensor: index for index, sensor in enumerate(SELECTED_SENSORS)}
    missing = [
        sensor
        for sensors in SENSOR_GROUPS.values()
        for sensor in sensors
        if sensor not in feature_index
    ]
    if missing:
        raise ValueError(f"SENSOR_GROUPS contains unknown selected sensors: {missing}")
    return {
        group_name: [feature_index[sensor] for sensor in sensors]
        for group_name, sensors in SENSOR_GROUPS.items()
    }


def get_group_attention_model(model: Model) -> Model:
    """Return a model that maps inputs to the four group attention weights."""
    return Model(
        inputs=model.inputs,
        outputs=model.get_layer("group_attention_weights").output,
        name=f"{model.name}_group_attention",
    )


def build_multibranch_lstm(
    input_shape: tuple = (SEQUENCE_LENGTH, N_FEATURES),
    dropout_rate: float = DROPOUT_RATE,
    use_attention: bool = True,
) -> Model:
    """Build the four-branch sensor-group LSTM or its no-attention ablation.

    Each named sensor group is encoded by an LSTM(32). The standard variant
    applies temporal attention within a group and group attention across the
    four resulting vectors. With ``use_attention=False``, both reductions are
    plain averages; the exposed group weights are uniformly 0.25.
    """
    if input_shape[-1] != len(SELECTED_SENSORS):
        raise ValueError(
            "Multi-branch LSTM requires one input feature per SELECTED_SENSORS "
            f"entry ({len(SELECTED_SENSORS)}), got {input_shape[-1]}."
        )

    inp = Input(shape=input_shape, name="input")
    group_vectors = []
    group_indices = _sensor_group_indices()

    for group_name, indices in group_indices.items():
        branch = layers.Lambda(
            lambda tensor, selected=indices: tf.gather(tensor, selected, axis=-1),
            name=f"{group_name}_slice",
        )(inp)
        branch = layers.LSTM(
            32, return_sequences=True, name=f"{group_name}_lstm"
        )(branch)

        if use_attention:
            temporal_scores = layers.Dense(1, name=f"{group_name}_temporal_score")(branch)
            temporal_weights = layers.Softmax(
                axis=1, name=f"{group_name}_temporal_attention"
            )(temporal_scores)
            branch_vector = layers.Lambda(
                lambda tensors: tf.reduce_sum(tensors[0] * tensors[1], axis=1),
                name=f"{group_name}_temporal_context",
            )([branch, temporal_weights])
        else:
            branch_vector = layers.GlobalAveragePooling1D(
                name=f"{group_name}_temporal_average"
            )(branch)
        group_vectors.append(branch_vector)

    group_stack = layers.Lambda(
        lambda vectors: tf.stack(vectors, axis=1), name="group_vectors"
    )(group_vectors)
    if use_attention:
        group_scores = layers.Dense(1, name="group_attention_score")(group_stack)
        group_weights_3d = layers.Softmax(axis=1, name="group_attention_softmax")(group_scores)
        group_weights = layers.Lambda(
            lambda weights: tf.squeeze(weights, axis=-1), name="group_attention_weights"
        )(group_weights_3d)
        group_context = layers.Lambda(
            lambda tensors: tf.reduce_sum(
                tensors[0] * tf.expand_dims(tensors[1], axis=-1), axis=1
            ),
            name="group_attention_context",
        )([group_stack, group_weights])
    else:
        group_weights = layers.Lambda(
            lambda vectors: tf.ones_like(vectors[:, :, 0]) / tf.cast(
                tf.shape(vectors)[1], vectors.dtype
            ),
            name="group_attention_weights",
        )(group_stack)
        group_context = layers.Lambda(
            lambda tensors: tf.reduce_sum(
                tensors[0] * tf.expand_dims(tensors[1], axis=-1), axis=1
            ),
            name="group_average",
        )([group_stack, group_weights])

    x = layers.Dense(32, activation="relu", name="dense_1")(group_context)
    x = layers.Dropout(dropout_rate, name="final_drop")(x)
    out = layers.Dense(1, name="rul_output")(x)

    model_name = "MultiBranchLSTM" if use_attention else "MultiBranchLSTM_NoAttn"
    return Model(inputs=inp, outputs=out, name=model_name)


def build_multibranch_noattn(
    input_shape: tuple = (SEQUENCE_LENGTH, N_FEATURES),
    dropout_rate: float = DROPOUT_RATE,
) -> Model:
    """Build the multi-branch LSTM ablation with temporal/group averaging."""
    return build_multibranch_lstm(
        input_shape=input_shape, dropout_rate=dropout_rate, use_attention=False
    )


# ---------------------------------------------------------------------------
# 3. GRU
# ---------------------------------------------------------------------------

def build_gru(
    input_shape: tuple = (SEQUENCE_LENGTH, N_FEATURES),
    dropout_rate: float = DROPOUT_RATE,
) -> Model:
    """
    Two-layer stacked GRU with MC-Dropout.

    Architecture
    ------------
    Input → GRU(128, return_sequences=True) → Dropout → GRU(64) →
    Dropout → Dense(32, relu) → Dense(1)
    """
    inp = Input(shape=input_shape, name="input")

    x = layers.GRU(128, return_sequences=True, name="gru_1")(inp)
    x = layers.Dropout(dropout_rate, name="drop_1")(x)

    x = layers.GRU(64, return_sequences=False, name="gru_2")(x)
    x = layers.Dropout(dropout_rate, name="drop_2")(x)

    x = layers.Dense(32, activation="relu", name="dense_1")(x)
    out = layers.Dense(1, name="rul_output")(x)

    model = Model(inputs=inp, outputs=out, name="GRU")
    return model


# ---------------------------------------------------------------------------
# 3. CNN-LSTM
# ---------------------------------------------------------------------------

def build_cnn_lstm(
    input_shape: tuple = (SEQUENCE_LENGTH, N_FEATURES),
    dropout_rate: float = DROPOUT_RATE,
) -> Model:
    """
    Two Conv1D feature-extraction blocks followed by an LSTM encoder.

    Architecture
    ------------
    Input
    → Conv1D(64, k=3, relu) → BatchNorm → MaxPool1D(2)
    → Conv1D(64, k=3, relu) → BatchNorm → MaxPool1D(2)
    → LSTM(64)
    → Dropout
    → Dense(32, relu) → Dense(1)
    """
    inp = Input(shape=input_shape, name="input")

    # Conv block 1
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu",
                      name="conv_1")(inp)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool_1")(x)

    # Conv block 2
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu",
                      name="conv_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool_2")(x)

    # Temporal encoder
    x = layers.LSTM(64, return_sequences=False, name="lstm")(x)
    x = layers.Dropout(dropout_rate, name="drop")(x)

    x = layers.Dense(32, activation="relu", name="dense_1")(x)
    out = layers.Dense(1, name="rul_output")(x)

    model = Model(inputs=inp, outputs=out, name="CNN_LSTM")
    return model


# ---------------------------------------------------------------------------
# 4. TCN (Temporal Convolutional Network)
# ---------------------------------------------------------------------------

def _tcn_residual_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    dilation_rate: int,
    dropout_rate: float,
    block_idx: int,
) -> tf.Tensor:
    """
    A single TCN residual block:

    x → Conv1D(causal, dilation=d, relu)
      → Dropout
      → Conv1D(causal, dilation=d)
      → [+ residual projection if channels differ]
      → Activation(relu)
    """
    residual = x

    # Causal dilated conv ×2
    x = layers.Conv1D(
        filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        activation="relu",
        name=f"tcn_conv1_{block_idx}",
    )(x)
    x = layers.Dropout(dropout_rate, name=f"tcn_drop1_{block_idx}")(x)

    x = layers.Conv1D(
        filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        name=f"tcn_conv2_{block_idx}",
    )(x)
    x = layers.Dropout(dropout_rate, name=f"tcn_drop2_{block_idx}")(x)

    # Projection residual connection if channel dims differ
    if residual.shape[-1] != filters:
        residual = layers.Conv1D(
            filters, kernel_size=1, padding="same",
            name=f"tcn_proj_{block_idx}"
        )(residual)

    x = layers.Add(name=f"tcn_add_{block_idx}")([x, residual])
    x = layers.Activation("relu", name=f"tcn_relu_{block_idx}")(x)
    return x


def build_tcn(
    input_shape: tuple = (SEQUENCE_LENGTH, N_FEATURES),
    dropout_rate: float = DROPOUT_RATE,
) -> Model:
    """
    Temporal Convolutional Network with dilated causal convolutions.

    Architecture
    ------------
    Input
    → TCN residual block (dilation=1)
    → TCN residual block (dilation=2)
    → TCN residual block (dilation=4)
    → TCN residual block (dilation=8)
    → GlobalAveragePooling1D
    → Dense(32, relu) → Dense(1)
    """
    dilation_rates = [1, 2, 4, 8]
    filters = 64
    kernel_size = 3

    inp = Input(shape=input_shape, name="input")
    x = inp

    for idx, d in enumerate(dilation_rates):
        x = _tcn_residual_block(
            x,
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=d,
            dropout_rate=dropout_rate,
            block_idx=idx,
        )

    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(32, activation="relu", name="dense_1")(x)
    out = layers.Dense(1, name="rul_output")(x)

    model = Model(inputs=inp, outputs=out, name="TCN")
    return model


# ---------------------------------------------------------------------------
# 5. Transformer
# ---------------------------------------------------------------------------

def _transformer_block(
    x: tf.Tensor,
    num_heads: int,
    ff_dim: int,
    dropout_rate: float,
    block_idx: int,
) -> tf.Tensor:
    """
    Single Transformer encoder block:
    LayerNorm → MultiHeadAttention + residual
    LayerNorm → FFN (Dense-relu-Dense) + residual
    """
    d_model = x.shape[-1]

    # Self-attention sub-layer
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout_rate,
        name=f"mha_{block_idx}",
    )(x, x)
    attn_out = layers.Dropout(dropout_rate, name=f"attn_drop_{block_idx}")(attn_out)
    x = layers.LayerNormalization(epsilon=1e-6, name=f"ln1_{block_idx}")(
        x + attn_out
    )

    # Feed-forward sub-layer
    ffn = layers.Dense(ff_dim, activation="relu",
                       name=f"ffn_1_{block_idx}")(x)
    ffn = layers.Dense(d_model, name=f"ffn_2_{block_idx}")(ffn)
    ffn = layers.Dropout(dropout_rate, name=f"ffn_drop_{block_idx}")(ffn)
    x = layers.LayerNormalization(epsilon=1e-6, name=f"ln2_{block_idx}")(
        x + ffn
    )
    return x


def build_transformer(
    input_shape: tuple = (SEQUENCE_LENGTH, N_FEATURES),
    dropout_rate: float = DROPOUT_RATE,
    num_heads: int = 4,
    ff_dim: int = 128,
    n_blocks: int = 2,
) -> Model:
    """
    Transformer encoder for RUL regression.

    Architecture
    ------------
    Input
    → Dense(d_model) projection (to align feature dim with positional enc)
    → Sinusoidal Positional Encoding
    → 2× Transformer Block (MultiHeadAttention + FFN + LayerNorm)
    → GlobalAveragePooling1D
    → Dense(64, relu) → Dropout → Dense(1)
    """
    inp = Input(shape=input_shape, name="input")

    # Project to a model dimension divisible by num_heads
    d_model = max(N_FEATURES, num_heads * 8)  # ensure head_dim >= 8
    # Align to num_heads multiple
    d_model = num_heads * ((d_model + num_heads - 1) // num_heads)

    x = layers.Dense(d_model, name="input_projection")(inp)

    # Add positional encoding (non-trainable, sinusoidal)
    seq_len = input_shape[0]
    pos_enc = positional_encoding(seq_len, d_model)
    x = x + pos_enc  # broadcast over batch dimension

    for i in range(n_blocks):
        x = _transformer_block(
            x,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            block_idx=i,
        )

    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(64, activation="relu", name="dense_1")(x)
    x = layers.Dropout(dropout_rate, name="final_drop")(x)
    out = layers.Dense(1, name="rul_output")(x)

    model = Model(inputs=inp, outputs=out, name="Transformer")
    return model


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def get_all_models(
    input_shape: tuple = (SEQUENCE_LENGTH, N_FEATURES),
) -> dict:
    """
    Build and compile all architectures.

    Parameters
    ----------
    input_shape : tuple  (seq_len, n_features)

    Returns
    -------
    dict mapping model name → compiled Keras Model
        Includes 'multibranch_lstm' and the 'multibranch_noattn' ablation.
    """
    builders = {
        "lstm": build_lstm,
        "multibranch_lstm": build_multibranch_lstm,
        "multibranch_noattn": build_multibranch_noattn,
        "gru": build_gru,
        "cnn_lstm": build_cnn_lstm,
        "tcn": build_tcn,
        "transformer": build_transformer,
    }

    compiled_models = {}
    for name, builder in builders.items():
        model = builder(input_shape=input_shape, dropout_rate=DROPOUT_RATE)
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss="mse",
            metrics=["mae"],
        )
        compiled_models[name] = model

    return compiled_models
