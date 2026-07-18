from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from ai_training.src.config import load_config, resolve_project_path, validate_label_sources
from ai_training.src.dataset import build_dataset
from ai_training.src.utils import ensure_dirs, setup_logging, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert the trained Keras model to TensorFlow Lite.")
    parser.add_argument("--config", default="ai_training/configs/mobilenetv2_config.yaml")
    parser.add_argument("--model", default="ai_training/outputs/checkpoints/best_finetuned.keras")
    parser.add_argument("--representative-samples", type=int, default=100)
    return parser.parse_args()


def tensor_details(path: Path) -> dict:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    return {
        "input_dtype": str(input_details["dtype"]),
        "output_dtype": str(output_details["dtype"]),
        "input_quantization": list(input_details.get("quantization", (0.0, 0))),
        "output_quantization": list(output_details.get("quantization", (0.0, 0))),
        "input_shape": input_details["shape"].tolist(),
        "output_shape": output_details["shape"].tolist(),
    }


def main() -> int:
    args = parse_args()
    setup_logging()
    config, _, repo_root = load_config(args.config)
    labels = validate_label_sources(config, repo_root)
    model_path = resolve_project_path(args.model, repo_root)
    train_csv = resolve_project_path(config["dataset"]["processed_dir"], repo_root) / "train.csv"
    tflite_dir = resolve_project_path(config["output"]["tflite_dir"], repo_root)
    ensure_dirs(tflite_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")

    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)

    outputs = {}
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    float32_path = tflite_dir / "nutrisnap_food_classifier_float32.tflite"
    float32_path.write_bytes(converter.convert())
    outputs["float32"] = tensor_details(float32_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    float16_path = tflite_dir / "nutrisnap_food_classifier_float16.tflite"
    float16_path.write_bytes(converter.convert())
    outputs["float16"] = tensor_details(float16_path)

    if not train_csv.exists():
        logging.warning("Train manifest missing; skipped INT8 conversion because representative data is required: %s", train_csv)
    else:
        representative_ds = build_dataset(train_csv, config, training=False, batch_size=1).take(args.representative_samples)

        def representative_data_gen():
            for image, _label in representative_ds:
                yield [np.asarray(image, dtype=np.float32)]

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        int8_path = tflite_dir / "nutrisnap_food_classifier_int8.tflite"
        int8_path.write_bytes(converter.convert())
        outputs["int8"] = tensor_details(int8_path)

    model_info = {
        "model_name": "NutriSnap Food Classifier",
        "architecture": "MobileNetV2",
        "input_shape": [1, int(config["model"]["input_height"]), int(config["model"]["input_width"]), 3],
        "class_count": len(labels),
        "labels_file": "food_labels.txt",
        "preprocessing": "MobileNetV2 preprocess_input",
        "output_activation": "softmax",
        "converted_models": outputs,
    }
    write_json(tflite_dir / "model_info.json", model_info)
    logging.info("TFLite models written to %s", tflite_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
