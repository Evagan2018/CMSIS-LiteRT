#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Train the "Hello World" sine model and export it for TensorFlow Lite Micro.

    Training/.venv/bin/python Training/train_model.py [--epochs N]

Writes Model/model_float.tflite and Model/model_int8.tflite (the int8 model is
fully integer-quantized with a representative dataset). This is the training
part of the pack's train_HelloWorld_model.ipynb as a script; converting the
models into the AI layer is create_ai_layer.py's job.

Needs TensorFlow (Training/requirements.txt): ./setup_venv.sh --training
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent.parent / "Model"
SAMPLES = 1000
SEED = 1


def dataset() -> tuple[np.ndarray, np.ndarray]:
    """x in [0, 2*pi) and y = sin(x) with a little noise, like the notebook."""
    np.random.seed(SEED)
    x = np.random.uniform(low=0, high=2 * math.pi, size=SAMPLES).astype(np.float32)
    np.random.shuffle(x)
    y = np.sin(x).astype(np.float32) + 0.1 * np.random.randn(*x.shape).astype(np.float32)
    return x, y


def train(x: np.ndarray, y: np.ndarray, epochs: int) -> tf.keras.Model:
    tf.random.set_seed(SEED)
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(1,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    split = int(0.8 * SAMPLES)
    model.fit(x[:split], y[:split], epochs=epochs, batch_size=64, validation_data=(x[split:], y[split:]), verbose=0)
    loss, mae = model.evaluate(x[split:], y[split:], verbose=0)
    print(f"[train] validation loss {loss:.4f}, mae {mae:.4f}")
    return model


def export(model: tf.keras.Model, x: np.ndarray) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    (MODEL_DIR / "model_float.tflite").write_bytes(converter.convert())

    def representative_dataset():
        for i in range(500):
            yield [x[i].reshape(1, 1)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.representative_dataset = representative_dataset
    (MODEL_DIR / "model_int8.tflite").write_bytes(converter.convert())

    for name in ("model_float.tflite", "model_int8.tflite"):
        print(f"[train] wrote {MODEL_DIR / name} ({(MODEL_DIR / name).stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()
    x, y = dataset()
    export(train(x, y, args.epochs), x)


if __name__ == "__main__":
    main()
