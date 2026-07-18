from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from ai_training.src.config import load_class_names, load_config, resolve_project_path, validate_label_sources
from ai_training.src.dataset import load_image_for_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one food image.")
    parser.add_argument("--config", default="ai_training/configs/mobilenetv2_config.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, _, repo_root = load_config(args.config)
    labels = validate_label_sources(config, repo_root)
    class_names = load_class_names(repo_root)
    model_path = resolve_project_path(args.model, repo_root)
    image_path = resolve_project_path(args.image, repo_root)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)
    image = load_image_for_inference(image_path, config)
    probs = model.predict(tf.expand_dims(image, axis=0), verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][: args.top_k]
    for rank, index in enumerate(top_indices, 1):
        print(f"{rank}. {labels[index]} - {class_names[index]['display_name']}: {probs[index] * 100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
