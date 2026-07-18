from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from ai_training.src.config import load_config, resolve_project_path, validate_label_sources
from ai_training.src.dataset import load_image_for_inference, read_manifest
from ai_training.src.metrics import classification_metrics
from ai_training.src.utils import ensure_dirs, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TensorFlow Lite models against the test split.")
    parser.add_argument("--config", default="ai_training/configs/mobilenetv2_config.yaml")
    parser.add_argument("--keras-model", default="ai_training/outputs/checkpoints/best_finetuned.keras")
    return parser.parse_args()


def apply_input_quantization(image: np.ndarray, input_detail: dict) -> np.ndarray:
    dtype = input_detail["dtype"]
    if dtype == np.float32:
        return image.astype(np.float32)
    scale, zero_point = input_detail["quantization"]
    if scale == 0:
        raise ValueError("Quantized input has zero scale.")
    quantized = image / scale + zero_point
    info = np.iinfo(dtype)
    return np.clip(np.round(quantized), info.min, info.max).astype(dtype)


def dequantize_output(output: np.ndarray, output_detail: dict) -> np.ndarray:
    if output_detail["dtype"] == np.float32:
        return output.astype(np.float32)
    scale, zero_point = output_detail["quantization"]
    return (output.astype(np.float32) - zero_point) * scale


def main() -> int:
    args = parse_args()
    setup_logging()
    config, _, repo_root = load_config(args.config)
    labels = validate_label_sources(config, repo_root)
    test_csv = resolve_project_path(config["dataset"]["processed_dir"], repo_root) / "test.csv"
    keras_path = resolve_project_path(args.keras_model, repo_root)
    tflite_dir = resolve_project_path(config["output"]["tflite_dir"], repo_root)
    report_dir = resolve_project_path(config["output"]["report_dir"], repo_root)
    ensure_dirs(report_dir)
    if not test_csv.exists():
        raise FileNotFoundError(f"Test manifest not found: {test_csv}. Run prepare_dataset.py first.")
    if not keras_path.exists():
        raise FileNotFoundError(f"Keras model not found: {keras_path}. Run train.py first.")

    import tensorflow as tf

    manifest = read_manifest(test_csv)
    y_true = manifest["class_index"].to_numpy()
    keras_model = tf.keras.models.load_model(keras_path)
    keras_probs = []
    for filepath in manifest["filepath"]:
        image = load_image_for_inference(filepath, config)
        keras_probs.append(keras_model.predict(tf.expand_dims(image, 0), verbose=0)[0])
    keras_probs = np.asarray(keras_probs)
    keras_acc = classification_metrics(y_true, keras_probs, labels)["top1_accuracy"]

    rows = []
    for model_path in sorted(tflite_dir.glob("*.tflite")):
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        if int(output_detail["shape"][-1]) != len(labels):
            raise ValueError(f"{model_path} output class count does not match labels.")
        probs = []
        timings = []
        for filepath in manifest["filepath"]:
            image = load_image_for_inference(filepath, config).numpy()
            image = np.expand_dims(image, 0)
            image = apply_input_quantization(image, input_detail)
            interpreter.set_tensor(input_detail["index"], image)
            start = time.perf_counter()
            interpreter.invoke()
            timings.append((time.perf_counter() - start) * 1000.0)
            output = interpreter.get_tensor(output_detail["index"])
            probs.append(dequantize_output(output, output_detail)[0])
        probs = np.asarray(probs)
        metrics = classification_metrics(y_true, probs, labels)
        rows.append(
            {
                "model": model_path.name,
                "size_mb": model_path.stat().st_size / (1024 * 1024),
                "input_dtype": str(input_detail["dtype"]),
                "output_dtype": str(output_detail["dtype"]),
                "top1_accuracy": metrics["top1_accuracy"],
                "top3_accuracy": metrics["top3_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "average_inference_ms": float(np.mean(timings)),
                "accuracy_drop_from_keras": float(keras_acc - metrics["top1_accuracy"]),
            }
        )
    pd.DataFrame(rows).to_csv(report_dir / "tflite_comparison.csv", index=False)
    logging.info("TFLite comparison written to %s", report_dir / "tflite_comparison.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
