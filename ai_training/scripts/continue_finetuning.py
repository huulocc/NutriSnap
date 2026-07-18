from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from ai_training.scripts.train import (
    LearningRateLogger,
    audit_batch,
    compute_class_weights,
    configure_tensorflow,
    input_shape,
    model_summary_text,
    read_json,
    run_prediction_checks,
)
from ai_training.src.config import load_config, load_labels, resolve_project_path, validate_label_sources
from ai_training.src.dataset import build_dataset, read_manifest
from ai_training.src.model import compile_model, get_sparse_smoothing_loss, parameter_report, set_fine_tuning
from ai_training.src.utils import ensure_dirs, get_git_commit, set_global_seed, setup_logging, write_json


EXPECTED_MANIFEST_SHA = "c0788ce4d943b61f11f34d55f6f9bd5541d8ad8707bac0ab194d09837d53c7a9"
EXPECTED_TRAIN_COUNT = 11623
EXPECTED_VALIDATION_COUNT = 1451
EXPECTED_TEST_COUNT = 1454


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled continuation fine-tuning for NutriSnap AI-6B.")
    parser.add_argument("--config", default="ai_training/configs/mobilenetv2_config.yaml")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--source-checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--minimum-learning-rate", type=float, default=5e-7)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--reduce-lr-patience", type=int, default=1)
    parser.add_argument("--require-gpu", action="store_true")
    return parser.parse_args()


def prepare_run_structure(run_dir: Path) -> dict[str, Path]:
    paths = {
        "logs": run_dir / "logs",
        "checkpoints": run_dir / "checkpoints",
        "histories": run_dir / "histories",
        "reports": run_dir / "reports",
        "plots": run_dir / "plots",
    }
    ensure_dirs(run_dir, *paths.values())
    return paths


def load_custom_model(tf, model_path: Path, config: dict[str, Any]):
    loss_object = get_sparse_smoothing_loss(
        int(config["model"]["num_classes"]),
        float(config["training"].get("label_smoothing", 0.0)),
    )
    return tf.keras.models.load_model(
        model_path,
        custom_objects={
            "NutriSnap>SparseCategoricalCrossentropyWithLabelSmoothing": loss_object.__class__,
            "SparseCategoricalCrossentropyWithLabelSmoothing": loss_object.__class__,
        },
    )


def make_run_dir(config: dict[str, Any], repo_root: Path) -> Path:
    run_root = resolve_project_path(config["output"]["run_dir"], repo_root)
    run_dir = run_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_mobilenetv2_continuation"
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    return run_dir


def make_callbacks(
    tf,
    run_paths: dict[str, Path],
    best_path: Path,
    last_path: Path,
    *,
    minimum_learning_rate: float,
    early_stopping_patience: int,
    reduce_lr_patience: int,
):
    monitor = "val_loss"
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=int(early_stopping_patience),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=int(reduce_lr_patience),
            min_lr=float(minimum_learning_rate),
        ),
        tf.keras.callbacks.ModelCheckpoint(best_path, monitor=monitor, save_best_only=True),
        tf.keras.callbacks.ModelCheckpoint(last_path, save_best_only=False),
        tf.keras.callbacks.CSVLogger(run_paths["histories"] / "continuation_history.csv"),
        tf.keras.callbacks.TensorBoard(log_dir=run_paths["logs"] / "continuation"),
        LearningRateLogger().callback(),
    ]


def validate_manifest(
    dataset_manifest: dict[str, Any],
    train_manifest: pd.DataFrame,
    validation_manifest: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    split_counts = dataset_manifest.get("split_counts", {})
    checks = {
        "manifest_sha256_matches_expected": dataset_manifest.get("manifest_sha256") == EXPECTED_MANIFEST_SHA,
        "split_protocol_matches_config": dataset_manifest.get("split_protocol") == config["dataset"].get("split_protocol"),
        "train_count_matches_expected": len(train_manifest) == EXPECTED_TRAIN_COUNT,
        "validation_count_matches_expected": len(validation_manifest) == EXPECTED_VALIDATION_COUNT,
        "test_count_from_manifest_metadata_matches_expected": split_counts.get("test") == EXPECTED_TEST_COUNT,
        "no_train_validation_filepath_overlap": not bool(set(train_manifest["filepath"]).intersection(set(validation_manifest["filepath"]))),
        "no_train_validation_sha256_overlap": not bool(set(train_manifest["sha256"]).intersection(set(validation_manifest["sha256"]))),
        "test_manifest_loaded": False,
        "test_dataset_created": False,
        "test_evaluation_executed": False,
    }
    false_expected = {"test_manifest_loaded", "test_dataset_created", "test_evaluation_executed"}
    failed = [
        key
        for key, value in checks.items()
        if (key in false_expected and value is not False) or (key not in false_expected and value is not True)
    ]
    if failed:
        raise RuntimeError(f"Continuation manifest validation failed: {failed}")
    return checks


def recursive_layers(model):
    for layer in model.layers:
        yield layer
        if hasattr(layer, "layers"):
            yield from recursive_layers(layer)


def batch_norm_frozen(tf, model) -> bool:
    bn_layers = [layer for layer in recursive_layers(model) if isinstance(layer, tf.keras.layers.BatchNormalization)]
    return bool(bn_layers) and all(not layer.trainable for layer in bn_layers)


def shape_list(shape) -> list[int | None]:
    return [int(dim) if dim is not None else None for dim in tuple(shape)]


def validate_model_contract(tf, model, config: dict[str, Any], expected_report: dict[str, Any]) -> dict[str, Any]:
    report = parameter_report(model)
    output_layer = model.layers[-1]
    checks = {
        "input_shape_matches": shape_list(model.input_shape) == [None, *input_shape(config)],
        "output_shape_matches": shape_list(model.output_shape) == [None, int(config["model"]["num_classes"])],
        "output_dtype_float32": getattr(output_layer, "dtype", None) == "float32",
        "batch_norm_layers_frozen": batch_norm_frozen(tf, model),
        "trainable_layers_match_stage2": report.get("trainable_layers") == expected_report.get("trainable_layers"),
        "frozen_layers_match_stage2": report.get("frozen_layers") == expected_report.get("frozen_layers"),
        "trainable_parameters_match_stage2": report.get("trainable_parameters") == expected_report.get("trainable_parameters"),
        "dropout_rate_matches_config": bool(
            np.isclose(
                next(layer.rate for layer in recursive_layers(model) if layer.name == "dropout"),
                float(config["model"]["dropout"]),
            )
        ),
    }
    failed = [key for key, value in checks.items() if value is not True]
    if failed:
        raise RuntimeError(f"Continuation model validation failed: {failed}")
    return {**checks, **report}


def validation_metrics_and_predictions(
    model,
    validation_csv: Path,
    validation_ds,
    labels: list[str],
    output_predictions_path: Path,
    output_report_path: Path,
) -> dict[str, Any]:
    eval_values = model.evaluate(validation_ds, verbose=0)
    probs = model.predict(validation_ds, verbose=0)
    manifest = read_manifest(validation_csv)
    y_true = manifest["class_index"].to_numpy()
    y_pred = np.argmax(probs, axis=1)
    top3 = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
    per_class = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    per_class_rows = []
    for index, label in enumerate(labels):
        row = per_class[label]
        per_class_rows.append(
            {
                "class_index": index,
                "class_name": label,
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1_score": float(row["f1-score"]),
                "support": int(row["support"]),
            }
        )
    pd.DataFrame(per_class_rows).to_csv(output_report_path, index=False)

    prediction_rows = []
    for row_index, row in manifest.reset_index(drop=True).iterrows():
        prediction = {
            "filepath": row["filepath"],
            "true_index": int(y_true[row_index]),
            "true_label": labels[int(y_true[row_index])],
            "predicted_index": int(y_pred[row_index]),
            "predicted_label": labels[int(y_pred[row_index])],
            "correct": bool(y_pred[row_index] == y_true[row_index]),
        }
        for rank, class_index in enumerate(top3[row_index], start=1):
            prediction[f"top{rank}_index"] = int(class_index)
            prediction[f"top{rank}_label"] = labels[int(class_index)]
            prediction[f"top{rank}_probability"] = float(probs[row_index, class_index])
        for class_index, label in enumerate(labels):
            prediction[f"prob_{label}"] = float(probs[row_index, class_index])
        prediction_rows.append(prediction)
    pd.DataFrame(prediction_rows).to_csv(output_predictions_path, index=False)

    metrics = {
        "val_loss": float(eval_values[0]),
        "top1_accuracy": float(np.mean(y_pred == y_true)),
        "top3_accuracy": float(np.mean([truth in row for truth, row in zip(y_true, top3)])),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": per_class_rows,
        "prediction_checks": {
            "prediction_shape": list(probs.shape),
            "no_nan": bool(not np.isnan(probs).any()),
            "no_inf": bool(not np.isinf(probs).any()),
            "probabilities_sum_approximately_one": bool(np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-3)),
        },
    }
    numeric_values = [value for value in metrics.values() if isinstance(value, float)]
    if any(np.isnan(value) or np.isinf(value) for value in numeric_values):
        raise RuntimeError("NaN or Inf validation metrics detected.")
    return metrics


def select_candidate(original: dict[str, Any], continued: dict[str, Any]) -> dict[str, Any]:
    macro_diff = float(continued["macro_f1"]) - float(original["macro_f1"])
    val_loss_diff = float(continued["val_loss"]) - float(original["val_loss"])
    if macro_diff <= -0.005:
        selected = "original"
        reason = "continued macro_f1 dropped by at least 0.005; keeping original validation candidate"
    elif macro_diff >= 0.002:
        selected = "continued"
        reason = "continued macro_f1 improved by at least 0.002"
    elif abs(macro_diff) < 0.002 and val_loss_diff < 0:
        selected = "continued"
        reason = "macro_f1 difference is below 0.002 and continued val_loss is lower"
    else:
        selected = "original"
        reason = "macro_f1 and val_loss are near equivalent; preferring original candidate"
    checkpoint = "checkpoints/best_continued.keras" if selected == "continued" else None
    return {
        "selected_candidate": selected,
        "selected_checkpoint": checkpoint,
        "selection_dataset": "validation",
        "test_set_used": False,
        "macro_f1_delta": macro_diff,
        "val_loss_delta": val_loss_diff,
        "reason": reason,
    }


def save_continuation_plots(history: pd.DataFrame, plot_dir: Path) -> None:
    for metric, filename in (
        ("accuracy", "continuation_accuracy.png"),
        ("loss", "continuation_loss.png"),
    ):
        plt.figure(figsize=(9, 5))
        for column in history.columns:
            if metric in column:
                plt.plot(range(1, len(history) + 1), history[column].values, label=column)
        plt.title(f"Continuation {metric}")
        plt.xlabel("Continuation epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / filename)
        plt.close()
    if "learning_rate" in history.columns:
        plt.figure(figsize=(9, 5))
        plt.plot(range(1, len(history) + 1), history["learning_rate"].values, label="learning_rate")
        plt.title("Continuation learning rate")
        plt.xlabel("Continuation epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "continuation_learning_rate.png")
        plt.close()


def full_history(source_run: Path, continuation_history: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    frames = []
    for stage, filename in (("stage1", "stage1_history.csv"), ("stage2", "stage2_history.csv")):
        frame = pd.read_csv(source_run / "histories" / filename)
        frame.insert(0, "stage_epoch", range(1, len(frame) + 1))
        frame.insert(0, "stage", stage)
        frames.append(frame)
    continuation = continuation_history.copy()
    continuation.insert(0, "stage_epoch", range(1, len(continuation) + 1))
    continuation.insert(0, "stage", "continuation")
    frames.append(continuation)
    combined = pd.concat(frames, ignore_index=True)
    combined.insert(0, "global_epoch", range(1, len(combined) + 1))
    combined.to_csv(output_path, index=False)
    return combined


def save_full_plots(history: pd.DataFrame, plot_dir: Path) -> None:
    stage1_len = int((history["stage"] == "stage1").sum())
    stage2_len = int((history["stage"] == "stage2").sum())
    boundaries = [
        (stage1_len + 0.5, "Stage 2 starts"),
        (stage1_len + stage2_len + 0.5, "Continuation starts"),
    ]
    for metric, filename in (("accuracy", "full_accuracy.png"), ("loss", "full_loss.png")):
        plt.figure(figsize=(11, 5))
        for column in history.columns:
            if metric in column:
                plt.plot(history["global_epoch"], history[column].values, label=column)
        for x_value, label in boundaries:
            plt.axvline(x_value, linestyle="--", color="black", alpha=0.6, label=label)
        plt.title(f"Full training {metric}")
        plt.xlabel("Global epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / filename)
        plt.close()
    if "learning_rate" in history.columns:
        plt.figure(figsize=(11, 5))
        plt.plot(history["global_epoch"], history["learning_rate"].values, label="learning_rate")
        for x_value, label in boundaries:
            plt.axvline(x_value, linestyle="--", color="black", alpha=0.6, label=label)
        plt.title("Full training learning rate")
        plt.xlabel("Global epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "full_learning_rate.png")
        plt.close()


def diagnose(original: dict[str, Any], continued: dict[str, Any], history: pd.DataFrame) -> tuple[str, str]:
    macro_diff = float(continued["macro_f1"]) - float(original["macro_f1"])
    loss_diff = float(continued["val_loss"]) - float(original["val_loss"])
    val_losses = history["val_loss"].dropna().astype(float).tolist() if "val_loss" in history else []
    train_losses = history["loss"].dropna().astype(float).tolist() if "loss" in history else []
    train_improves = len(train_losses) >= 2 and train_losses[-1] < train_losses[0]
    val_best_final = bool(val_losses) and np.isclose(val_losses[-1], min(val_losses), rtol=0.0, atol=1e-5)
    val_degraded = len(val_losses) >= 2 and val_losses[-1] > min(val_losses) + 0.02

    if any(np.isnan(value) or np.isinf(value) for value in val_losses + train_losses):
        label = "Unstable"
    elif macro_diff >= 0.002 and loss_diff < 0 and val_best_final:
        label = "Still improving"
    elif train_improves and (macro_diff < -0.005 or loss_diff > 0.02) and val_degraded:
        label = "Clear overfitting"
    elif train_improves and (macro_diff < 0 or loss_diff > 0):
        label = "Mild overfitting"
    else:
        label = "Converged"
    reason = (
        f"Continuation macro_f1 delta={macro_diff:.6f}, val_loss delta={loss_diff:.6f}, "
        f"epochs_run={len(history)}, best_continuation_val_loss={min(val_losses) if val_losses else None}."
    )
    return label, reason


def android_unchanged(repo_root: Path) -> bool:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", "app"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and not result.stdout.strip()


def collect_environment(tf, seed: int, config: dict[str, Any], dataset_manifest: dict[str, Any]) -> dict[str, Any]:
    build_info = {}
    try:
        build_info = tf.sysconfig.get_build_info()
    except Exception:
        pass
    return {
        "python_version": sys.version,
        "tensorflow_version": tf.__version__,
        "keras_version": getattr(tf.keras, "__version__", "unknown"),
        "operating_system": platform.platform(),
        "gpu_devices": [gpu.name for gpu in tf.config.list_physical_devices("GPU")],
        "cuda_build_status": build_info.get("is_cuda_build"),
        "cudnn_version": build_info.get("cudnn_version"),
        "mixed_precision_policy": tf.keras.mixed_precision.global_policy().name,
        "random_seed": seed,
        "training_protocol": config["dataset"].get("split_protocol"),
        "dataset_manifest_hash": dataset_manifest.get("manifest_sha256"),
        "batch_size": int(config["training"]["batch_size"]),
        "input_shape": input_shape(config),
        "number_of_classes": len(config["dataset"]["classes"]),
        "class_order": config["dataset"]["classes"],
    }


def main() -> int:
    args = parse_args()
    if args.epochs > 10:
        raise ValueError("AI-6B allows at most 10 additional epochs.")

    setup_logging()
    config, config_path, repo_root = load_config(args.config)
    labels = validate_label_sources(config, repo_root)
    config["training"]["fine_tune_learning_rate"] = float(args.learning_rate)
    config["training"]["minimum_learning_rate"] = float(args.minimum_learning_rate)
    config["training"]["early_stopping_patience"] = int(args.early_stopping_patience)
    config["training"]["reduce_lr_patience"] = int(args.reduce_lr_patience)

    tf, gpus = configure_tensorflow(config, require_gpu=args.require_gpu, no_mixed_precision=False)
    seed = int(config["project"]["seed"])
    set_global_seed(seed)

    source_run = resolve_project_path(args.source_run, repo_root)
    source_checkpoint = resolve_project_path(args.source_checkpoint, repo_root) if args.source_checkpoint else source_run / "checkpoints" / "best_finetuned.keras"
    if not source_run.exists():
        raise FileNotFoundError(f"Source run not found: {source_run}")
    if not source_checkpoint.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {source_checkpoint}")

    processed_dir = resolve_project_path(config["dataset"]["processed_dir"], repo_root)
    train_csv = processed_dir / "train.csv"
    validation_csv = processed_dir / "validation.csv"
    dataset_manifest_path = processed_dir / "dataset_manifest.json"
    for path in (train_csv, validation_csv, dataset_manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"Required AI-6B input not found: {path}")

    dataset_manifest = read_json(dataset_manifest_path)
    train_manifest = read_manifest(train_csv)
    validation_manifest = read_manifest(validation_csv)
    manifest_checks = validate_manifest(dataset_manifest, train_manifest, validation_manifest, config)

    source_metadata = read_json(source_run / "run_metadata.json")
    source_stage2_metrics = read_json(source_run / "reports" / "stage2_validation_metrics.json")
    if source_metadata.get("class_order") != labels:
        raise RuntimeError("Source run class order differs from current config.")
    if source_metadata.get("dataset_manifest_hash") != dataset_manifest.get("manifest_sha256"):
        raise RuntimeError("Source run manifest hash differs from current manifest.")

    run_dir = make_run_dir(config, repo_root)
    run_paths = prepare_run_structure(run_dir)
    shutil.copy2(config_path, run_dir / "config_resolved.yaml")
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    shutil.copy2(dataset_manifest_path, run_dir / "dataset_manifest_snapshot.json")

    train_ds = build_dataset(train_csv, config, training=True)
    validation_ds = build_dataset(validation_csv, config, training=False)
    batch_audit = {
        "train": audit_batch(train_ds, labels, "train"),
        "validation": audit_batch(validation_ds, labels, "validation"),
    }
    class_weights = compute_class_weights(train_manifest, labels) if config["dataset"].get("use_class_weights", True) else None

    source_candidate = {
        "source_run": str(source_run),
        "source_checkpoint": str(source_checkpoint),
        "selected_stage": "stage2",
        "checkpoint_role": "best_finetuned",
        "metrics": source_stage2_metrics,
        "test_set_used": False,
    }
    write_json(run_dir / "source_candidate.json", source_candidate)
    write_json(run_paths["reports"] / "original_candidate_validation_metrics.json", source_stage2_metrics)

    model = load_custom_model(tf, source_checkpoint, config)
    loaded_report = parameter_report(model)
    if (
        loaded_report.get("trainable_layers") != source_stage2_metrics.get("trainable_layers")
        or loaded_report.get("frozen_layers") != source_stage2_metrics.get("frozen_layers")
    ):
        set_fine_tuning(
            model,
            int(config["training"].get("fine_tune_last_layers", 50)),
            bool(config["training"].get("freeze_batch_normalization", True)),
        )
    model_contract = validate_model_contract(tf, model, config, source_stage2_metrics)
    compile_model(config, model, float(args.learning_rate))
    (run_dir / "model_summary.txt").write_text(model_summary_text(model), encoding="utf-8")

    best_continued = run_paths["checkpoints"] / "best_continued.keras"
    last_continued = run_paths["checkpoints"] / "last_continued.keras"
    history_obj = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=int(args.epochs),
        callbacks=make_callbacks(
            tf,
            run_paths,
            best_continued,
            last_continued,
            minimum_learning_rate=float(args.minimum_learning_rate),
            early_stopping_patience=int(args.early_stopping_patience),
            reduce_lr_patience=int(args.reduce_lr_patience),
        ),
        class_weight=class_weights,
    )
    continuation_history = pd.DataFrame(history_obj.history)
    continuation_history.to_csv(run_paths["histories"] / "continuation_history.csv", index=False)
    model.save(last_continued)
    save_continuation_plots(continuation_history, run_paths["plots"])

    if not best_continued.exists():
        raise FileNotFoundError(f"Best continuation checkpoint was not written: {best_continued}")
    continued_model = load_custom_model(tf, best_continued, config)
    continued_metrics = validation_metrics_and_predictions(
        continued_model,
        validation_csv,
        validation_ds,
        labels,
        run_paths["reports"] / "continued_validation_predictions.csv",
        run_paths["reports"] / "continued_classification_report.csv",
    )
    continued_metrics.update(parameter_report(continued_model))
    write_json(run_paths["reports"] / "continued_candidate_validation_metrics.json", continued_metrics)

    comparison = pd.DataFrame(
        [
            {"candidate": "original", "checkpoint": str(source_checkpoint), **source_stage2_metrics},
            {"candidate": "continued", "checkpoint": "checkpoints/best_continued.keras", **{k: v for k, v in continued_metrics.items() if k != "per_class"}},
        ]
    )
    comparison.to_csv(run_paths["reports"] / "candidate_comparison.csv", index=False)
    selected = select_candidate(source_stage2_metrics, continued_metrics)
    if selected["selected_candidate"] == "original":
        selected["selected_checkpoint"] = str(source_checkpoint)
    write_json(run_paths["reports"] / "selected_candidate.json", selected)

    combined_history = full_history(
        source_run,
        continuation_history,
        run_paths["histories"] / "full_training_history.csv",
    )
    save_full_plots(combined_history, run_paths["plots"])
    diagnosis, diagnosis_reason = diagnose(source_stage2_metrics, continued_metrics, continuation_history)

    prediction_checks = run_prediction_checks(tf, best_continued, validation_ds, labels, config)
    validation_checks = {
        **manifest_checks,
        "class_order_unchanged": labels == source_metadata.get("class_order") == load_labels(repo_root),
        "source_checkpoint_load_predict": True,
        "continued_checkpoint_load_predict": prediction_checks["checkpoint_load"],
        "continued_checkpoint_can_predict_one_batch": True,
        "exactly_15_output_classes": prediction_checks["exactly_15_output_classes"],
        "model_input_shape": shape_list(continued_model.input_shape),
        "model_output_shape": shape_list(continued_model.output_shape),
        "no_nan_predictions": continued_metrics["prediction_checks"]["no_nan"],
        "no_inf_predictions": continued_metrics["prediction_checks"]["no_inf"],
        "probability_outputs_sum_approximately_to_one": continued_metrics["prediction_checks"]["probabilities_sum_approximately_one"],
        "batch_norm_layers_frozen": batch_norm_frozen(tf, continued_model),
        "trainable_layers": parameter_report(continued_model)["trainable_layers"],
        "frozen_layers": parameter_report(continued_model)["frozen_layers"],
        "android_unchanged": android_unchanged(repo_root),
        "tflite_export_executed": False,
        "android_export_executed": False,
    }
    false_expected = {
        "test_manifest_loaded",
        "test_dataset_created",
        "test_evaluation_executed",
        "tflite_export_executed",
        "android_export_executed",
    }
    informational = {"model_input_shape", "model_output_shape", "trainable_layers", "frozen_layers"}
    failed_checks = [
        key
        for key, value in validation_checks.items()
        if key not in informational
        and ((key in false_expected and value is not False) or (key not in false_expected and value is not True))
    ]
    if failed_checks:
        raise RuntimeError(f"Continuation validation checks failed: {failed_checks}")
    write_json(run_paths["reports"] / "continuation_validation_checks.json", validation_checks)

    metadata = {
        **collect_environment(tf, seed, config, dataset_manifest),
        "phase": "AI-6B",
        "run_type": "controlled_continuation_fine_tuning",
        "git_commit": get_git_commit(repo_root),
        "source_run": str(source_run),
        "source_checkpoint": str(source_checkpoint),
        "epochs_requested": int(args.epochs),
        "epochs_executed": int(len(continuation_history)),
        "learning_rate": float(args.learning_rate),
        "minimum_learning_rate": float(args.minimum_learning_rate),
        "early_stopping_patience": int(args.early_stopping_patience),
        "reduce_lr_patience": int(args.reduce_lr_patience),
        "test_manifest_loaded": False,
        "test_dataset_created": False,
        "test_evaluation_executed": False,
        "tflite_export_executed": False,
        "android_export_executed": False,
        "batch_audit": batch_audit,
        "model_contract": model_contract,
    }
    write_json(run_dir / "continuation_metadata.json", metadata)

    analysis = f"""# Continuation Analysis

Phase: AI-6B

Source checkpoint: `{source_checkpoint}`

Continuation run: `{run_dir}`

Test set used: false.

## Original Validation Candidate

- Validation loss: {source_stage2_metrics['val_loss']:.6f}
- Top-1 accuracy: {source_stage2_metrics['top1_accuracy']:.6f}
- Top-3 accuracy: {source_stage2_metrics['top3_accuracy']:.6f}
- Macro F1: {source_stage2_metrics['macro_f1']:.6f}

## Continued Validation Candidate

- Validation loss: {continued_metrics['val_loss']:.6f}
- Top-1 accuracy: {continued_metrics['top1_accuracy']:.6f}
- Top-3 accuracy: {continued_metrics['top3_accuracy']:.6f}
- Macro F1: {continued_metrics['macro_f1']:.6f}

## Selection

Selected candidate: `{selected['selected_candidate']}`

Reason: {selected['reason']}

## Diagnosis

Diagnosis: {diagnosis}

{diagnosis_reason}
"""
    (run_paths["reports"] / "continuation_analysis.md").write_text(analysis, encoding="utf-8")
    logging.info("Continuation complete. Run directory: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
