"""Model definitions following Tables 5 and 6 (CNN encoder, BiLSTM + attention), plus E1/E2 competitors and the MoE of D4."""
from __future__ import annotations
import numpy as np


def _tf():
    import tensorflow as tf
    return tf


def build_cnn(n_classes: int, n_feat: int = 28):
    """Table 5: 4x7x1 -> Conv(32,3x3) BN MaxPool -> Conv(64,1x1) BN GAP -> Dense32 Dropout.25 -> Dense5 (SF-1..SF-5) -> class head.
    The class head (softmax over n_classes) is an addition needed to obtain posteriors; SF-1..SF-5 feed ONLY this head."""
    tf = _tf(); L = tf.keras.layers
    inp = L.Input((n_feat,), name="x")
    x = L.Reshape((4, 7, 1))(inp)
    x = L.Conv2D(32, 3, padding="same", activation="relu")(x); x = L.BatchNormalization()(x)
    x = L.MaxPool2D(2)(x)
    x = L.Conv2D(64, 1, activation="relu")(x); x = L.BatchNormalization()(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dense(32, activation="relu")(x); x = L.Dropout(0.25)(x)
    sf = L.Dense(5, name="sf")(x)
    out = L.Dense(n_classes, activation="softmax", name="p")(sf)
    return tf.keras.Model(inp, out, name="cnn")


import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="hai")
class TemporalAttention(tf.keras.layers.Layer):
    """Scaled dot-product temporal self-attention (Table 6), mean-pooled over time."""
    def __init__(self, units=96, **kw):
        super().__init__(**kw); self.units = units
        self.q = tf.keras.layers.Dense(units); self.k = tf.keras.layers.Dense(units); self.v = tf.keras.layers.Dense(units)

    def call(self, seq):
        q, k, v = self.q(seq), self.k(seq), self.v(seq)
        a = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / np.sqrt(self.units), axis=-1)
        return tf.reduce_mean(tf.matmul(a, v), axis=1)

    def get_config(self):
        c = super().get_config(); c["units"] = self.units; return c


def build_bilstm(n_classes: int, T: int = 20, n_feat: int = 28):
    """Table 6: 3 stacked BiLSTM (128/64/48 per direction, dropout .30/.25/.20) + temporal self-attention + Dense64 + softmax."""
    tf = _tf(); L = tf.keras.layers
    inp = L.Input((T, n_feat), name="w")
    x = L.Bidirectional(L.LSTM(128, return_sequences=True, dropout=0.30))(inp)
    x = L.Bidirectional(L.LSTM(64, return_sequences=True, dropout=0.25))(x)
    x = L.Bidirectional(L.LSTM(48, return_sequences=True, dropout=0.20))(x)
    x = TemporalAttention(96)(x)
    x = L.Dense(64, activation="relu")(x); x = L.Dropout(0.20)(x)
    out = L.Dense(n_classes, activation="softmax", name="p")(x)
    return tf.keras.Model(inp, out, name="bilstm")


def build_cnn_attention(n_classes: int, T: int = 20, n_feat: int = 28):
    """E1: CNN with self-attention in the spirit of Hasanpuri et al. [27] (re-implemented on the HFR windows)."""
    tf = _tf(); L = tf.keras.layers
    inp = L.Input((T, n_feat))
    x = L.Conv1D(64, 3, padding="same", activation="relu")(inp); x = L.BatchNormalization()(x)
    x = L.Conv1D(64, 3, padding="same", activation="relu")(x)
    a = L.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = L.LayerNormalization()(x + a)
    x = L.GlobalAveragePooling1D()(x)
    x = L.Dense(64, activation="relu")(x); x = L.Dropout(0.25)(x)
    return tf.keras.Model(inp, L.Dense(n_classes, activation="softmax")(x), name="cnn_attn")


def build_bilstm_embed(n_classes: int, T: int = 20, n_feat: int = 28):
    """E2 stage 1: BiLSTM encoder (embedding is later fed to a random forest, Singh et al. [26])."""
    tf = _tf(); L = tf.keras.layers
    inp = L.Input((T, n_feat))
    x = L.Bidirectional(L.LSTM(64, return_sequences=False))(inp)
    emb = L.Dense(32, activation="relu", name="emb")(x)
    out = L.Dense(n_classes, activation="softmax")(L.Dropout(0.2)(emb))
    return tf.keras.Model(inp, out, name="bilstm_embed")


def build_moe(n_classes: int, T: int = 20, n_feat: int = 28):
    """D4: jointly trained mixture of experts. Trees are not differentiable, so the third expert is an MLP on the last
    time step (documented substitution). The gate sees the last-step HFR vector. Output = sum_k g_k(x) p_k(x)."""
    tf = _tf(); L = tf.keras.layers
    inp = L.Input((T, n_feat))
    last = L.Lambda(lambda z: z[:, -1, :])(inp)
    e1 = build_cnn(n_classes, n_feat)(last)
    e2 = build_bilstm(n_classes, T, n_feat)(inp)
    h = L.Dense(128, activation="relu")(last); h = L.Dense(64, activation="relu")(h)
    e3 = L.Dense(n_classes, activation="softmax")(h)
    g = L.Dense(32, activation="relu")(last); g = L.Dense(3, activation="softmax", name="gate")(g)
    st = L.Lambda(lambda z: tf.stack(z, axis=1))([e1, e2, e3])           # (B,3,C)
    out = L.Lambda(lambda z: tf.reduce_sum(z[0][:, :, None] * z[1], axis=1))([g, st])
    return tf.keras.Model(inp, out, name="moe")


def build_gate_mlp(n_in: int, hidden: int = 16, wd: float = 0.0):
    """D2/D3: learned gating network: 2-layer MLP, softmax over the three component weights."""
    tf = _tf(); L = tf.keras.layers
    inp = L.Input((n_in,))
    x = L.Dense(hidden, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(wd))(inp)
    x = L.Dense(hidden, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(wd))(x)
    return tf.keras.Model(inp, L.Dense(3, activation="softmax")(x), name="gate")


def compile_fit(model, X, y, Xv, yv, nn_cfg: dict, loss="sparse_categorical_crossentropy", verbose=0, sw=None):
    tf = _tf()
    model.compile(tf.keras.optimizers.Adam(nn_cfg["lr"]), loss=loss, metrics=["accuracy"])
    cb = [tf.keras.callbacks.EarlyStopping(patience=nn_cfg["patience"], restore_best_weights=True, monitor="val_loss")]
    return model.fit(X, y, validation_data=(Xv, yv), epochs=nn_cfg["epochs"], batch_size=nn_cfg["batch"],
                     callbacks=cb, verbose=verbose, shuffle=True, sample_weight=sw)


def predict_batched(model, make_batch, n: int, bs: int = 8192) -> np.ndarray:
    out = []
    for a in range(0, n, bs):
        out.append(model.predict(make_batch(slice(a, min(a + bs, n))), batch_size=bs, verbose=0))
    return np.concatenate(out, 0).astype(np.float32)
