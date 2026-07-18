from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ai_training.src.config import load_config, resolve_project_path, validate_label_sources
from ai_training.src.utils import ensure_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Android asset package without modifying the Android app.")
    parser.add_argument("--config", default="ai_training/configs/mobilenetv2_config.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="android_export")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, _, repo_root = load_config(args.config)
    validate_label_sources(config, repo_root)
    model = resolve_project_path(args.model, repo_root)
    output = resolve_project_path(args.output, repo_root)
    if not model.exists():
        raise FileNotFoundError(f"TFLite model not found: {model}")
    ensure_dirs(output)
    shutil.copy2(model, output / "nutrisnap_food_classifier.tflite")
    shutil.copy2(repo_root / "ai_training" / "metadata" / "food_labels.txt", output / "food_labels.txt")
    shutil.copy2(repo_root / "ai_training" / "metadata" / "class_names.json", output / "class_names.json")
    model_info = resolve_project_path(config["output"]["tflite_dir"], repo_root) / "model_info.json"
    if model_info.exists():
        shutil.copy2(model_info, output / "model_info.json")
    print(f"Android export package written to: {output}")
    print("Android app assets were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
