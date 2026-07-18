from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ai_training.src.config import load_class_names, load_config, resolve_project_path, validate_label_sources
from ai_training.src.dataset import build_dataset, read_manifest
from ai_training.src.model import get_sparse_smoothing_loss
from ai_training.src.utils import ensure_dirs, get_git_commit, set_global_seed, setup_logging, write_json


EXPECTED_MODEL = "ai_training/outputs/runs/20260717_162321_mobilenetv2_continuation/checkpoints/best_continued.keras"
EXPECTED_SOURCE_RUN = "ai_training/outputs/runs/20260717_162321_mobilenetv2_continuation"
EXPECTED_MANIFEST_SHA = "c0788ce4d943b61f11f34d55f6f9bd5541d8ad8707bac0ab194d09837d53c7a9"
EXPECTED_SPLIT_PROTOCOL = "deduplicated_stratified_80_10_10_v1"
EXPECTED_COUNTS = {"train": 11623, "validation": 1451, "test": 1454}
EXPECTED_LABELS = [
    "pho",
    "banh_mi",
    "com_tam",
    "bun_bo_hue",
    "goi_cuon",
    "banh_xeo",
    "mi_quang",
    "xoi_xeo",
    "chao_long",
    "bun_thit_nuong",
    "bun_rieu",
    "hu_tieu",
    "banh_cuon",
    "banh_trang_nuong",
    "cao_lau",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the locked one-time AI-7 test evaluation.")
    parser.add_argument("--config", default="ai_training/configs/mobilenetv2_config.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--confirm-open-test", action="store_true")
    return parser.parse_args()


def require_confirmation(args: argparse.Namespace) -> None:
    if not args.confirm_open_test:
        raise SystemExit(
            "Refusing to open the locked test split without --confirm-open-test. "
            "This command is the one-time final model test evaluation."
        )


def configure_tensorflow(require_gpu: bool):
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            logging.debug("Could not set memory growth for %s", gpu)
    if require_gpu and not gpus:
        raise RuntimeError("--require-gpu was set, but TensorFlow does not see a GPU.")
    if gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
    return tf, gpus


def shape_list(shape) -> list[int | None]:
    return [int(dim) if dim is not None else None for dim in tuple(shape)]


def make_evaluation_dir(repo_root: Path) -> tuple[Path, dict[str, Path]]:
    root = repo_root / "ai_training" / "outputs" / "evaluations"
    evaluation_dir = root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_mobilenetv2_v1_test"
    if evaluation_dir.exists():
        raise FileExistsError(f"Evaluation directory already exists: {evaluation_dir}")
    paths = {"reports": evaluation_dir / "reports", "plots": evaluation_dir / "plots"}
    ensure_dirs(evaluation_dir, *paths.values())
    return evaluation_dir, paths


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_split_overlap(left: pd.DataFrame, right: pd.DataFrame, column: str) -> bool:
    return set(left[column]).isdisjoint(set(right[column]))


def integrity_checks(
    repo_root: Path,
    config: dict[str, Any],
    labels: list[str],
    train_manifest: pd.DataFrame,
    validation_manifest: pd.DataFrame,
    test_manifest: pd.DataFrame,
    dataset_manifest: dict[str, Any],
) -> dict[str, Any]:
    split_counts = dataset_manifest.get("split_counts", {})
    rejected_path = repo_root / "ai_training" / "data" / "processed" / "rejected.csv"
    rejected_cross_test = False
    if rejected_path.exists():
        rejected = pd.read_csv(rejected_path)
        if "sha256" in rejected.columns and "reason" in rejected.columns:
            test_hashes = set(test_manifest["sha256"])
            rejected_cross_test = bool(
                rejected[
                    rejected["sha256"].isin(test_hashes)
                    & rejected["reason"].astype(str).str.contains("cross", case=False, na=False)
                ].shape[0]
            )
    checks = {
        "manifest_sha256_matches_expected": dataset_manifest.get("manifest_sha256") == EXPECTED_MANIFEST_SHA,
        "split_protocol_matches_expected": dataset_manifest.get("split_protocol") == EXPECTED_SPLIT_PROTOCOL,
        "train_samples": len(train_manifest),
        "validation_samples": len(validation_manifest),
        "test_samples": len(test_manifest),
        "train_count_matches_expected": len(train_manifest) == EXPECTED_COUNTS["train"] == split_counts.get("train"),
        "validation_count_matches_expected": len(validation_manifest) == EXPECTED_COUNTS["validation"] == split_counts.get("validation"),
        "test_count_matches_expected": len(test_manifest) == EXPECTED_COUNTS["test"] == split_counts.get("test"),
        "class_count_matches_expected": len(labels) == int(config["model"]["num_classes"]) == 15,
        "class_order_matches_expected": labels == EXPECTED_LABELS,
        "test_filepath_no_overlap_train": check_split_overlap(test_manifest, train_manifest, "filepath"),
        "test_filepath_no_overlap_validation": check_split_overlap(test_manifest, validation_manifest, "filepath"),
        "test_sha256_no_overlap_train": check_split_overlap(test_manifest, train_manifest, "sha256"),
        "test_sha256_no_overlap_validation": check_split_overlap(test_manifest, validation_manifest, "sha256"),
        "no_rejected_cross_class_duplicate_in_test_manifest": not rejected_cross_test,
        "each_class_has_test_sample": set(test_manifest["class_index"]) == set(range(15)),
    }
    failed = [key for key, value in checks.items() if isinstance(value, bool) and not value]
    if failed:
        raise RuntimeError(f"Preflight integrity check failed before inference: {failed}")
    return checks


def validate_model_before_inference(tf, model, labels: list[str], config: dict[str, Any]) -> dict[str, Any]:
    input_shape = [None, int(config["model"]["input_height"]), int(config["model"]["input_width"]), int(config["model"]["input_channels"])]
    output_shape = [None, len(labels)]
    dummy = tf.zeros([1, *input_shape[1:]], dtype=tf.float32)
    probs = model(dummy, training=False).numpy()
    checks = {
        "model_input_shape": shape_list(model.input_shape),
        "model_output_shape": shape_list(model.output_shape),
        "model_input_shape_matches_expected": shape_list(model.input_shape) == input_shape,
        "model_output_shape_matches_expected": shape_list(model.output_shape) == output_shape,
        "model_output_dtype": str(model.output.dtype),
        "model_output_dtype_valid": str(model.output.dtype) in {"float16", "float32"},
        "dummy_inference_shape": list(probs.shape),
        "dummy_inference_no_nan": bool(not np.isnan(probs).any()),
        "dummy_inference_no_inf": bool(not np.isinf(probs).any()),
        "dummy_probability_sum_approximately_one": bool(np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-3)),
    }
    failed = [key for key, value in checks.items() if isinstance(value, bool) and not value]
    if failed:
        raise RuntimeError(f"Model preflight check failed before test inference: {failed}")
    return checks


def compute_test_loss(tf, y_true: np.ndarray, probabilities: np.ndarray, config: dict[str, Any]) -> float:
    loss = get_sparse_smoothing_loss(
        int(config["model"]["num_classes"]),
        float(config["training"].get("label_smoothing", 0.0)),
    )
    return float(tf.reduce_mean(loss(tf.convert_to_tensor(y_true), tf.convert_to_tensor(probabilities))).numpy())


def compute_metrics(tf, y_true: np.ndarray, probabilities: np.ndarray, config: dict[str, Any], labels: list[str]) -> dict[str, Any]:
    y_pred = np.argmax(probabilities, axis=1)
    top3 = np.argsort(probabilities, axis=1)[:, -3:]
    metrics = {
        "test_loss": compute_test_loss(tf, y_true, probabilities, config),
        "top1_accuracy": float(np.mean(y_pred == y_true)),
        "top3_accuracy": float(np.mean([truth in row for truth, row in zip(y_true, top3)])),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", labels=list(range(len(labels))), zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", labels=list(range(len(labels))), zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(range(len(labels))), zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", labels=list(range(len(labels))), zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", labels=list(range(len(labels))), zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=list(range(len(labels))), zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sample_count": int(len(y_true)),
    }
    if any(np.isnan(value) or np.isinf(value) for value in metrics.values() if isinstance(value, float)):
        raise RuntimeError("NaN or Inf test metric detected.")
    return metrics


def bootstrap_ci(y_true: np.ndarray, probabilities: np.ndarray, labels: list[str], *, seed: int = 42, resamples: int = 1000) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    y_pred = np.argmax(probabilities, axis=1)
    n = len(y_true)
    top1_values = []
    macro_f1_values = []
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        sample_true = y_true[idx]
        sample_pred = y_pred[idx]
        top1_values.append(np.mean(sample_pred == sample_true))
        macro_f1_values.append(f1_score(sample_true, sample_pred, average="macro", labels=list(range(len(labels))), zero_division=0))
    return {
        "method": "percentile_bootstrap_95_ci",
        "seed": seed,
        "resamples": resamples,
        "top1_accuracy": {
            "low": float(np.percentile(top1_values, 2.5)),
            "high": float(np.percentile(top1_values, 97.5)),
        },
        "macro_f1": {
            "low": float(np.percentile(macro_f1_values, 2.5)),
            "high": float(np.percentile(macro_f1_values, 97.5)),
        },
    }


def make_predictions_frame(test_manifest: pd.DataFrame, probabilities: np.ndarray, labels: list[str], class_names: dict[int, dict[str, str]]) -> pd.DataFrame:
    y_true = test_manifest["class_index"].to_numpy()
    y_pred = np.argmax(probabilities, axis=1)
    top3 = np.argsort(probabilities, axis=1)[:, -3:][:, ::-1]
    rows = []
    for idx, row in test_manifest.reset_index(drop=True).iterrows():
        pred = int(y_pred[idx])
        second = int(top3[idx][1])
        third = int(top3[idx][2])
        rows.append(
            {
                "manifest_row_index": int(idx),
                "filepath": row["filepath"],
                "source_split": row.get("source_split", ""),
                "true_index": int(y_true[idx]),
                "true_label": labels[int(y_true[idx])],
                "true_display_name": row.get("display_name", class_names[int(y_true[idx])]["display_name"]),
                "predicted_index": pred,
                "predicted_label": labels[pred],
                "predicted_display_name": class_names[pred]["display_name"],
                "confidence": float(probabilities[idx, pred]),
                "second_label": labels[second],
                "second_confidence": float(probabilities[idx, second]),
                "third_label": labels[third],
                "third_confidence": float(probabilities[idx, third]),
                "correct": bool(pred == int(y_true[idx])),
            }
        )
    return pd.DataFrame(rows)


def make_classification_outputs(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
    class_names: dict[int, dict[str, str]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    y_pred = np.argmax(probabilities, axis=1)
    raw = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for index, label in enumerate(labels):
        item = raw[label]
        true_mask = y_true == index
        pred_mask = y_pred == index
        rows.append(
            {
                "class_index": index,
                "model_label": label,
                "display_name": class_names[index]["display_name"],
                "support": int(item["support"]),
                "precision": float(item["precision"]),
                "recall": float(item["recall"]),
                "f1_score": float(item["f1-score"]),
                "top1_correct": int(np.sum(true_mask & pred_mask)),
                "top1_total": int(np.sum(true_mask)),
            }
        )
    return pd.DataFrame(rows), rows


def save_confusion_matrix(matrix: np.ndarray, labels: list[str], path: Path, title: str, *, normalized: bool) -> None:
    fig, ax = plt.subplots(figsize=(15, 13))
    image = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text = f"{value:.2f}" if normalized else str(int(value))
            ax.text(col, row, text, ha="center", va="center", fontsize=6, color="black")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_barplot(frame: pd.DataFrame, value: str, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(frame["model_label"], frame[value])
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel(value)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_confidence_distribution(predictions: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(predictions["confidence"], bins=np.linspace(0, 1, 11), color="#4169e1", edgecolor="white")
    ax.set_title("Test confidence distribution")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Sample count")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_correct_vs_incorrect_confidence(predictions: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        [predictions[predictions["correct"]]["confidence"], predictions[~predictions["correct"]]["confidence"]],
        bins=np.linspace(0, 1, 11),
        label=["correct", "incorrect"],
        color=["#2e8b57", "#c44536"],
        edgecolor="white",
    )
    ax.set_title("Correct vs incorrect confidence")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Sample count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_image_grid(rows: pd.DataFrame, output_path: Path, title: str, max_images: int = 16) -> None:
    from PIL import Image

    sample = rows.head(max_images)
    cols = 4
    grid_rows = max(1, int(np.ceil(max(len(sample), 1) / cols)))
    fig = plt.figure(figsize=(cols * 4, grid_rows * 4))
    if sample.empty:
        plt.text(0.5, 0.5, "No samples", ha="center", va="center")
        plt.axis("off")
    for idx, row in enumerate(sample.itertuples(), 1):
        ax = fig.add_subplot(grid_rows, cols, idx)
        try:
            ax.imshow(Image.open(row.filepath).convert("RGB"))
        except Exception as exc:
            ax.text(0.5, 0.5, f"Image load failed\n{exc}", ha="center", va="center", fontsize=8)
        ax.set_title(
            f"True: {row.true_label}\nPred: {row.predicted_label}\nConf: {row.confidence:.2f}",
            fontsize=8,
        )
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def confidence_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    display_bins = np.linspace(0, 1, 11)
    labels = [f"{display_bins[i]:.1f}-{display_bins[i + 1]:.1f}" for i in range(len(display_bins) - 1)]
    bins = display_bins.copy()
    bins[-1] = np.nextafter(1.0, 2.0)
    bucket = pd.cut(predictions["confidence"], bins=bins, labels=labels, include_lowest=True, right=False)
    rows = []
    for label in labels:
        subset = predictions[bucket == label]
        correct = int(subset["correct"].sum())
        total = int(len(subset))
        rows.append(
            {
                "confidence_bin": label,
                "sample_count": total,
                "correct_count": correct,
                "incorrect_count": total - correct,
                "empirical_accuracy": float(correct / total) if total else None,
            }
        )
    return pd.DataFrame(rows)


def threshold_analysis(predictions: pd.DataFrame, class_count: int) -> pd.DataFrame:
    rows = []
    y_true = predictions["true_index"].to_numpy()
    y_pred = predictions["predicted_index"].to_numpy()
    for threshold in [0.40, 0.50, 0.60, 0.70, 0.80]:
        accepted = predictions["confidence"].to_numpy() >= threshold
        accepted_count = int(np.sum(accepted))
        rejected_count = int(len(predictions) - accepted_count)
        if accepted_count:
            accepted_accuracy = float(np.mean(y_true[accepted] == y_pred[accepted]))
            accepted_macro_f1 = float(
                f1_score(y_true[accepted], y_pred[accepted], average="macro", labels=list(range(class_count)), zero_division=0)
            )
            incorrect_accepted = int(np.sum(y_true[accepted] != y_pred[accepted]))
        else:
            accepted_accuracy = None
            accepted_macro_f1 = None
            incorrect_accepted = 0
        rows.append(
            {
                "threshold": threshold,
                "accepted_samples": accepted_count,
                "rejected_samples": rejected_count,
                "coverage": float(accepted_count / len(predictions)),
                "accuracy_on_accepted": accepted_accuracy,
                "macro_f1_on_accepted": accepted_macro_f1,
                "incorrect_accepted": incorrect_accepted,
                "correct_rejected": int(np.sum((y_true[~accepted] == y_pred[~accepted]))),
            }
        )
    return pd.DataFrame(rows)


def suggest_threshold(thresholds: pd.DataFrame) -> dict[str, Any]:
    candidates = thresholds[(thresholds["coverage"] >= 0.70) & (thresholds["accuracy_on_accepted"].fillna(0) >= 0.85)]
    if candidates.empty:
        row = thresholds.sort_values(["accuracy_on_accepted", "coverage"], ascending=False).iloc[0]
        reason = "No threshold reached both 70% coverage and 85% accepted accuracy; selecting the strongest observed tradeoff."
    else:
        row = candidates.sort_values(["incorrect_accepted", "coverage"], ascending=[True, False]).iloc[0]
        reason = "Selected from thresholds with at least 70% coverage and 85% accepted accuracy."
    return {
        "threshold": float(row["threshold"]),
        "coverage": float(row["coverage"]),
        "accuracy_on_accepted": None if pd.isna(row["accuracy_on_accepted"]) else float(row["accuracy_on_accepted"]),
        "incorrect_accepted": int(row["incorrect_accepted"]),
        "reason": reason,
    }


def confusion_pairs(predictions: pd.DataFrame, class_counts: pd.Series) -> pd.DataFrame:
    errors = predictions[~predictions["correct"]]
    if errors.empty:
        return pd.DataFrame(columns=["true_label", "predicted_label", "count", "percentage_of_true_class", "average_confidence"])
    grouped = (
        errors.groupby(["true_label", "predicted_label"], as_index=False)
        .agg(count=("filepath", "count"), average_confidence=("confidence", "mean"))
    )
    grouped["percentage_of_true_class"] = grouped.apply(
        lambda row: float(row["count"] / class_counts[row["true_label"]]), axis=1
    )
    return grouped.sort_values(["count", "average_confidence"], ascending=[False, False])


def environment_report(tf, gpus: list[Any]) -> dict[str, Any]:
    gpu_details = []
    for gpu in gpus:
        item = {"name": gpu.name}
        try:
            item.update(tf.config.experimental.get_device_details(gpu))
        except Exception:
            pass
        gpu_details.append(item)
    return {
        "python_version": sys.version,
        "tensorflow_version": tf.__version__,
        "keras_version": getattr(tf.keras, "__version__", "unknown"),
        "operating_system": platform.platform(),
        "gpu_devices": [gpu.name for gpu in gpus],
        "gpu_details": gpu_details,
        "mixed_precision_policy": tf.keras.mixed_precision.global_policy().name,
    }


def safe_git_diff_app(repo_root: Path) -> bool:
    result = subprocess.run(["git", "diff", "--name-only", "--", "app"], cwd=repo_root, capture_output=True, text=True, check=False)
    return result.returncode == 0 and not result.stdout.strip()


def write_model_card(
    path: Path,
    metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
    per_class: pd.DataFrame,
    pairs: pd.DataFrame,
    threshold: dict[str, Any],
) -> None:
    best = per_class.sort_values("f1_score", ascending=False).head(3)
    weakest = per_class.sort_values("f1_score", ascending=True).head(3)
    pair_text = "\n".join(
        f"- {row.true_label} -> {row.predicted_label}: {int(row.count)} ({row.percentage_of_true_class:.2%})"
        for row in pairs.head(5).itertuples()
    ) or "- No confusion pairs observed."
    text = f"""# Final Model Card

## Model identity

Name: NutriSnap Food Classifier
Version: mobilenetv2-v1
Architecture: MobileNetV2
Classes: 15
Input: RGB image 224 x 224
Output: Softmax over 15 classes

## Dataset

30VNFoods subset
14,528 accepted samples
Custom deduplicated stratified split
Train: 11,623
Validation: 1,451
Test: 1,454

## Important disclaimer

This evaluation uses a custom deduplicated 80/10/10 split and is not directly
comparable to results reported using the original official 30VNFoods split.

## Validation metrics

- Validation loss: {validation_metrics['val_loss']:.6f}
- Top-1 accuracy: {validation_metrics['top1_accuracy']:.6f}
- Top-3 accuracy: {validation_metrics['top3_accuracy']:.6f}
- Macro F1: {validation_metrics['macro_f1']:.6f}
- Weighted F1: {validation_metrics['weighted_f1']:.6f}

## Test metrics

- Test loss: {metrics['test_loss']:.6f}
- Top-1 accuracy: {metrics['top1_accuracy']:.6f}
- Top-3 accuracy: {metrics['top3_accuracy']:.6f}
- Macro precision: {metrics['macro_precision']:.6f}
- Macro recall: {metrics['macro_recall']:.6f}
- Macro F1: {metrics['macro_f1']:.6f}
- Weighted F1: {metrics['weighted_f1']:.6f}
- Balanced accuracy: {metrics['balanced_accuracy']:.6f}

## Per-class performance

Best classes by F1: {", ".join(best['model_label'].tolist())}

Weakest classes by F1: {", ".join(weakest['model_label'].tolist())}

Most confused class pairs:

{pair_text}

## Intended use

- Recognize a primary Vietnamese food item in an image.
- Support meal logging workflows.
- Return a suggestion for user confirmation.

## Limitations

- Supports only 15 classes.
- Images containing multiple dishes may be unreliable.
- Foods outside the taxonomy are still forced into one of the 15 classes.
- Low-confidence predictions need user confirmation.
- Nutrition is not inferred directly from the model.
- This is not a medical tool.
- The model does not estimate exact portion size from an image.
- The dataset may contain label noise.
- Real-world performance can be lower than test-set performance.

## Deployment recommendation

- Server API should use the `.keras` checkpoint.
- Client should map predictions through canonical `model_label`.
- Do not auto-save meals before user confirmation.
- Use a confidence threshold to warn or request confirmation.
- Suggested review threshold: {threshold['threshold']:.2f} based on locked test confidence analysis.
- Coverage at suggested threshold: {threshold['coverage']:.6f}
- Accuracy on accepted samples: {threshold['accuracy_on_accepted']:.6f}
- Continue monitoring real-world data after deployment.

This model is locked for the first version, but it is not claimed to be perfect or production-safe for all real-world conditions.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    require_confirmation(args)
    setup_logging()
    config, _config_path, repo_root = load_config(args.config)
    labels = validate_label_sources(config, repo_root)
    class_names = load_class_names(repo_root)
    set_global_seed(int(config["project"]["seed"]))

    model_path = resolve_project_path(args.model, repo_root)
    expected_model_path = resolve_project_path(EXPECTED_MODEL, repo_root)
    if model_path != expected_model_path:
        raise RuntimeError(f"AI-7 may only evaluate the locked checkpoint: {EXPECTED_MODEL}")
    if args.model_version != "mobilenetv2-v1":
        raise RuntimeError("AI-7 model version must be mobilenetv2-v1.")
    if not model_path.exists():
        raise FileNotFoundError(f"Locked model checkpoint not found: {model_path}")

    processed_dir = resolve_project_path(config["dataset"]["processed_dir"], repo_root)
    train_csv = processed_dir / "train.csv"
    validation_csv = processed_dir / "validation.csv"
    test_csv = processed_dir / "test.csv"
    dataset_manifest_path = processed_dir / "dataset_manifest.json"
    for path in (train_csv, validation_csv, test_csv, dataset_manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"Required locked evaluation input missing: {path}")

    dataset_manifest = load_json(dataset_manifest_path)
    train_manifest = read_manifest(train_csv)
    validation_manifest = read_manifest(validation_csv)
    test_manifest = read_manifest(test_csv)
    preflight_checks = integrity_checks(repo_root, config, labels, train_manifest, validation_manifest, test_manifest, dataset_manifest)

    evaluation_dir, paths = make_evaluation_dir(repo_root)
    selected_candidate_path = repo_root / EXPECTED_SOURCE_RUN / "reports" / "selected_candidate.json"
    validation_metrics_path = repo_root / EXPECTED_SOURCE_RUN / "reports" / "continued_candidate_validation_metrics.json"
    selected_candidate = load_json(selected_candidate_path)
    validation_metrics = load_json(validation_metrics_path)
    (evaluation_dir / "dataset_manifest_snapshot.json").write_text(
        json.dumps(dataset_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (evaluation_dir / "selected_candidate_snapshot.json").write_text(
        json.dumps(selected_candidate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    tf, gpus = configure_tensorflow(args.require_gpu)
    env = environment_report(tf, gpus)
    write_json(evaluation_dir / "environment.json", env)

    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=True)
    model_checks = validate_model_before_inference(tf, model, labels, config)

    test_ds = build_dataset(test_csv, config, training=False)
    probabilities = model.predict(test_ds, verbose=1)
    y_true = test_manifest["class_index"].to_numpy()
    if len(probabilities) != EXPECTED_COUNTS["test"]:
        raise RuntimeError(f"Prediction count mismatch: expected 1454, got {len(probabilities)}")
    if np.isnan(probabilities).any() or np.isinf(probabilities).any():
        raise RuntimeError("NaN or Inf probabilities detected.")
    if not np.allclose(np.sum(probabilities, axis=1), 1.0, atol=1e-3):
        raise RuntimeError("Probability rows do not sum approximately to one.")

    metrics = compute_metrics(tf, y_true, probabilities, config, labels)
    metrics["confidence_intervals"] = bootstrap_ci(y_true, probabilities, labels)
    write_json(paths["reports"] / "test_metrics.json", metrics)

    predictions = make_predictions_frame(test_manifest, probabilities, labels, class_names)
    predictions.to_csv(paths["reports"] / "test_predictions.csv", index=False)
    per_class_frame, per_class_rows = make_classification_outputs(y_true, probabilities, labels, class_names)
    per_class_frame.to_csv(paths["reports"] / "classification_report.csv", index=False)
    write_json(paths["reports"] / "per_class_metrics.json", {"classes": per_class_rows})

    misclassified = predictions[~predictions["correct"]].sort_values("confidence", ascending=False)
    high_confidence_errors = misclassified[misclassified["confidence"] >= 0.70]
    low_confidence = predictions[predictions["confidence"] < 0.50].sort_values("confidence", ascending=True)
    misclassified.to_csv(paths["reports"] / "misclassified_samples.csv", index=False)
    high_confidence_errors.to_csv(paths["reports"] / "high_confidence_errors.csv", index=False)
    low_confidence.to_csv(paths["reports"] / "low_confidence_predictions.csv", index=False)

    class_counts = predictions.groupby("true_label")["filepath"].count()
    pairs = confusion_pairs(predictions, class_counts)
    pairs.to_csv(paths["reports"] / "class_confusion_pairs.csv", index=False)

    confidence_frame = confidence_analysis(predictions)
    confidence_frame.to_csv(paths["reports"] / "confidence_analysis.csv", index=False)
    thresholds = threshold_analysis(predictions, len(labels))
    thresholds.to_csv(paths["reports"] / "confidence_threshold_analysis.csv", index=False)
    threshold = suggest_threshold(thresholds)

    cm = confusion_matrix(y_true, predictions["predicted_index"].to_numpy(), labels=list(range(len(labels))))
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.divide(cm, cm.sum(axis=1, keepdims=True), where=cm.sum(axis=1, keepdims=True) != 0)
    cm_norm = np.nan_to_num(cm_norm)
    save_confusion_matrix(cm, labels, paths["plots"] / "confusion_matrix_counts.png", "Confusion matrix counts", normalized=False)
    save_confusion_matrix(cm_norm, labels, paths["plots"] / "confusion_matrix_normalized.png", "Confusion matrix row-normalized", normalized=True)
    save_barplot(per_class_frame, "f1_score", paths["plots"] / "per_class_f1.png", "Per-class F1")
    save_barplot(per_class_frame, "recall", paths["plots"] / "per_class_recall.png", "Per-class recall")
    save_confidence_distribution(predictions, paths["plots"] / "confidence_distribution.png")
    save_correct_vs_incorrect_confidence(predictions, paths["plots"] / "correct_vs_incorrect_confidence.png")
    save_image_grid(high_confidence_errors, paths["plots"] / "high_confidence_errors.png", "High-confidence errors")
    save_image_grid(low_confidence, paths["plots"] / "low_confidence_predictions.png", "Low-confidence predictions")

    evaluation_checks = {
        **preflight_checks,
        **model_checks,
        "prediction_count": int(len(probabilities)),
        "prediction_count_matches_expected": len(probabilities) == EXPECTED_COUNTS["test"],
        "metrics_finite": all(np.isfinite(value) for value in metrics.values() if isinstance(value, float)),
        "probabilities_no_nan": bool(not np.isnan(probabilities).any()),
        "probabilities_no_inf": bool(not np.isinf(probabilities).any()),
        "probability_sums_approximately_one": bool(np.allclose(np.sum(probabilities, axis=1), 1.0, atol=1e-3)),
        "classification_report_contains_all_15_classes": len(per_class_frame) == 15,
        "test_used_for_model_selection": False,
        "fit_called": False,
        "tflite_export_executed": False,
        "android_export_executed": False,
        "android_unchanged": safe_git_diff_app(repo_root),
    }
    write_json(paths["reports"] / "evaluation_validation_checks.json", evaluation_checks)

    metadata = {
        "phase": "AI-7",
        "evaluation_type": "locked_one_time_test_evaluation",
        "model_version": args.model_version,
        "checkpoint": EXPECTED_MODEL,
        "source_run": EXPECTED_SOURCE_RUN,
        "split_protocol": EXPECTED_SPLIT_PROTOCOL,
        "dataset_manifest_sha256": EXPECTED_MANIFEST_SHA,
        "test_samples": EXPECTED_COUNTS["test"],
        "model_selected_using": "validation_macro_f1",
        "test_used_for_model_selection": False,
        "retraining_after_test_allowed": False,
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(repo_root),
        **env,
    }
    write_json(evaluation_dir / "evaluation_metadata.json", metadata)

    write_model_card(paths["reports"] / "final_model_card.md", metrics, validation_metrics, per_class_frame, pairs, threshold)
    final_model = {
        "model_name": "NutriSnap Food Classifier",
        "model_version": args.model_version,
        "architecture": "MobileNetV2",
        "checkpoint": EXPECTED_MODEL,
        "class_count": 15,
        "labels_path": "ai_training/metadata/food_labels.txt",
        "class_names_path": "ai_training/metadata/class_names.json",
        "input_shape": [None, 224, 224, 3],
        "output_activation": "softmax",
        "dataset_manifest_sha256": EXPECTED_MANIFEST_SHA,
        "split_protocol": EXPECTED_SPLIT_PROTOCOL,
        "selected_using": "validation_macro_f1",
        "test_evaluation_completed": True,
        "retraining_after_test": False,
        "test_metrics": metrics,
        "evaluation_directory": str(evaluation_dir.relative_to(repo_root)),
        "suggested_review_threshold": threshold,
    }
    write_json(repo_root / "ai_training" / "outputs" / "final_model.json", final_model)
    (repo_root / "ai_training" / "outputs" / "latest_evaluation.txt").write_text(str(evaluation_dir), encoding="utf-8")

    logging.info("Locked test evaluation complete: %s", evaluation_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
