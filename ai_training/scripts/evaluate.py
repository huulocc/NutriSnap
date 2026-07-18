from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ai_training.src.config import load_config, resolve_project_path, validate_label_sources
from ai_training.src.dataset import build_dataset, read_manifest
from ai_training.src.metrics import classification_metrics, classification_report_frame
from ai_training.src.utils import ensure_dirs, setup_logging, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Keras food classifier.")
    parser.add_argument("--config", default="ai_training/configs/mobilenetv2_config.yaml")
    parser.add_argument("--model", default="ai_training/outputs/checkpoints/best_finetuned.keras")
    return parser.parse_args()


def save_confusion_matrix(matrix: list[list[int]], labels: list[str], path: Path) -> None:
    plt.figure(figsize=(12, 10))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_image_grid(rows: pd.DataFrame, output_path: Path, title: str, max_images: int = 16) -> None:
    from PIL import Image

    sample = rows.head(max_images)
    if sample.empty:
        return
    cols = 4
    grid_rows = int(np.ceil(len(sample) / cols))
    plt.figure(figsize=(cols * 4, grid_rows * 4))
    for idx, row in enumerate(sample.itertuples(), 1):
        plt.subplot(grid_rows, cols, idx)
        try:
            plt.imshow(Image.open(row.filepath).convert("RGB"))
        except Exception:
            continue
        plt.title(f"{row.true_label}->{row.predicted_label}\n{row.confidence:.2f}")
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> int:
    args = parse_args()
    setup_logging()
    config, _, repo_root = load_config(args.config)
    labels = validate_label_sources(config, repo_root)
    test_csv = resolve_project_path(config["dataset"]["processed_dir"], repo_root) / "test.csv"
    model_path = resolve_project_path(args.model, repo_root)
    report_dir = resolve_project_path(config["output"]["report_dir"], repo_root)
    plot_dir = resolve_project_path(config["output"]["plot_dir"], repo_root)
    ensure_dirs(report_dir, plot_dir)
    if not test_csv.exists():
        raise FileNotFoundError(f"Test manifest not found: {test_csv}. Run prepare_dataset.py first.")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")

    import tensorflow as tf

    test_manifest = read_manifest(test_csv)
    test_ds = build_dataset(test_csv, config, training=False)
    model = tf.keras.models.load_model(model_path)
    loss, accuracy, top3 = model.evaluate(test_ds, verbose=0)
    probabilities = model.predict(test_ds)
    y_true = test_manifest["class_index"].to_numpy()
    y_pred = np.argmax(probabilities, axis=1)
    confidence = probabilities.max(axis=1)
    metrics = classification_metrics(y_true, probabilities, labels)
    metrics.update({"test_loss": float(loss), "keras_accuracy_metric": float(accuracy), "keras_top3_metric": float(top3)})
    write_json(report_dir / "test_metrics.json", metrics)

    report = classification_report_frame(y_true, y_pred, labels)
    report.to_csv(report_dir / "classification_report.csv", index=False)
    predictions = pd.DataFrame(
        {
            "filepath": test_manifest["filepath"],
            "true_index": y_true,
            "true_label": [labels[index] for index in y_true],
            "predicted_index": y_pred,
            "predicted_label": [labels[index] for index in y_pred],
            "confidence": confidence,
            "correct": y_true == y_pred,
        }
    )
    predictions.to_csv(report_dir / "test_predictions.csv", index=False)
    errors = predictions[~predictions["correct"]].sort_values("confidence", ascending=False)
    errors[["filepath", "true_label", "predicted_label", "confidence"]].to_csv(report_dir / "misclassified_samples.csv", index=False)
    save_confusion_matrix(metrics["confusion_matrix"], labels, plot_dir / "confusion_matrix.png")
    save_image_grid(errors, plot_dir / "high_confidence_errors.png", "High confidence errors")
    save_image_grid(predictions.sort_values("confidence", ascending=True), plot_dir / "low_confidence_predictions.png", "Low confidence predictions")
    logging.info("Evaluation reports written to %s", report_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
