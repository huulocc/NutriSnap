from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

from ai_training.src.config import load_config, load_labels, resolve_project_path, validate_label_sources
from ai_training.src.dataset import build_dataset, read_manifest
from ai_training.src.model import (
    build_mobilenetv2,
    compile_model,
    freeze_backbone,
    get_sparse_smoothing_loss,
    parameter_report,
    set_fine_tuning,
)
from ai_training.src.utils import ensure_dirs, get_git_commit, set_global_seed, setup_logging, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the NutriSnap MobileNetV2 food classifier.")
    parser.add_argument("--config", default="ai_training/configs/mobilenetv2_config.yaml")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume-run", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--initial-epochs", type=int, default=None)
    parser.add_argument("--fine-tune-epochs", type=int, default=None)
    parser.add_argument("--no-mixed-precision", action="store_true")
    return parser.parse_args()


class LearningRateLogger:
    def callback(self):
        import tensorflow as tf

        class _LearningRateLogger(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                lr = self.model.optimizer.learning_rate
                try:
                    lr_value = float(tf.keras.backend.get_value(lr))
                except TypeError:
                    lr_value = float(lr(self.model.optimizer.iterations))
                logs["learning_rate"] = lr_value
                logging.info("Epoch %s learning_rate=%s", epoch + 1, lr_value)

        return _LearningRateLogger()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def input_shape(config: dict[str, Any]) -> list[int]:
    return [
        int(config["model"]["input_height"]),
        int(config["model"]["input_width"]),
        int(config["model"]["input_channels"]),
    ]


def configure_tensorflow(config: dict[str, Any], *, require_gpu: bool, no_mixed_precision: bool):
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            logging.debug("Could not set memory growth for %s", gpu)
    if require_gpu and not gpus:
        raise RuntimeError("--require-gpu was set, but TensorFlow does not see a GPU.")

    if config["training"].get("mixed_precision", False) and not no_mixed_precision and gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
        if not gpus:
            logging.warning("TensorFlow does not see a GPU; mixed precision disabled and CPU smoke/training is allowed.")
    return tf, gpus


def collect_environment(tf, seed: int, config: dict[str, Any], dataset_manifest: dict[str, Any], train_count: int, val_count: int) -> dict[str, Any]:
    physical_gpus = tf.config.list_physical_devices("GPU")
    gpu_details = []
    for gpu in physical_gpus:
        detail = {"name": gpu.name}
        try:
            detail.update(tf.config.experimental.get_device_details(gpu))
        except Exception:
            pass
        gpu_details.append(detail)
    try:
        import psutil

        ram_gb = round(psutil.virtual_memory().available / (1024**3), 2)
        cpu_info = platform.processor() or platform.machine()
    except Exception:
        ram_gb = None
        cpu_info = platform.processor() or platform.machine()

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
        "cpu_information": cpu_info,
        "available_ram_gb": ram_gb,
        "gpu_devices": [gpu.name for gpu in physical_gpus],
        "gpu_details": gpu_details,
        "cuda_build_status": build_info.get("is_cuda_build"),
        "cudnn_version": build_info.get("cudnn_version"),
        "mixed_precision_policy": tf.keras.mixed_precision.global_policy().name,
        "xla_status": bool(tf.config.optimizer.get_jit()),
        "random_seed": seed,
        "training_protocol": config["dataset"].get("split_protocol"),
        "dataset_manifest_hash": dataset_manifest.get("manifest_sha256"),
        "train_sample_count": train_count,
        "validation_sample_count": val_count,
        "test_sample_count_from_manifest_metadata": dataset_manifest.get("split_counts", {}).get("test"),
        "number_of_classes": len(config["dataset"]["classes"]),
        "class_order": config["dataset"]["classes"],
        "batch_size": int(config["training"]["batch_size"]),
        "input_shape": input_shape(config),
    }


def make_run_dir(config: dict[str, Any], repo_root: Path, *, smoke_test: bool, resume_run: str | None) -> Path:
    if resume_run:
        run_dir = resolve_project_path(resume_run, repo_root)
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume run directory not found: {run_dir}")
        return run_dir
    suffix = "smoke_test" if smoke_test else "mobilenetv2"
    run_root = resolve_project_path(config["output"]["run_dir"], repo_root)
    run_dir = run_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}"
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    return run_dir


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


def validate_resume(run_dir: Path, config: dict[str, Any], dataset_manifest: dict[str, Any]) -> None:
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Cannot resume without run metadata: {metadata_path}")
    metadata = read_json(metadata_path)
    if metadata.get("class_order") != config["dataset"]["classes"]:
        raise RuntimeError("Cannot resume: class order differs.")
    if metadata.get("dataset_manifest_hash") != dataset_manifest.get("manifest_sha256"):
        raise RuntimeError("Cannot resume: dataset manifest hash differs.")
    if metadata.get("input_shape") != input_shape(config):
        raise RuntimeError("Cannot resume: input shape differs.")
    if not (run_dir / "checkpoints" / "best_head.keras").exists():
        raise FileNotFoundError("Cannot resume Stage 2: checkpoints/best_head.keras is missing.")
    logging.info("Resume mode: continuing Stage 2 from best_head.keras")


def compute_class_weights(train_manifest: pd.DataFrame, labels: list[str]) -> dict[int, float]:
    classes = np.arange(len(labels))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_manifest["class_index"].to_numpy(),
    )
    return {int(class_index): float(weight) for class_index, weight in zip(classes, weights)}


def audit_batch(dataset, labels: list[str], prefix: str) -> dict[str, Any]:
    images, y = next(iter(dataset))
    image_min = float(np.min(images.numpy()))
    image_max = float(np.max(images.numpy()))
    label_min = int(np.min(y.numpy()))
    label_max = int(np.max(y.numpy()))
    result = {
        "image_batch_shape": list(images.shape),
        "label_batch_shape": list(y.shape),
        "image_dtype": str(images.dtype),
        "image_min": image_min,
        "image_max": image_max,
        "label_min": label_min,
        "label_max": label_max,
    }
    logging.info("%s batch audit: %s", prefix, result)
    if result["image_batch_shape"][1:] != [224, 224, 3]:
        raise RuntimeError(f"{prefix} image shape is invalid: {result['image_batch_shape']}")
    if label_min < 0 or label_max >= len(labels):
        raise RuntimeError(f"{prefix} label range is invalid: {label_min}..{label_max}")
    if image_min < -1.1 or image_max > 1.1:
        raise RuntimeError(f"{prefix} preprocessing range is invalid: {image_min}..{image_max}")
    return result


def model_summary_text(model) -> str:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines)


def make_callbacks(tf, run_paths: dict[str, Path], stage: str, best_path: Path, last_path: Path, config: dict[str, Any]):
    monitor = config["training"].get("monitor", "val_loss")
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=int(config["training"]["early_stopping_patience"]),
            restore_best_weights=bool(config["training"].get("restore_best_weights", True)),
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            patience=int(config["training"]["reduce_lr_patience"]),
            min_lr=float(config["training"]["minimum_learning_rate"]),
        ),
        tf.keras.callbacks.ModelCheckpoint(best_path, monitor=monitor, save_best_only=True),
        tf.keras.callbacks.ModelCheckpoint(last_path, save_best_only=False),
        tf.keras.callbacks.CSVLogger(run_paths["histories"] / f"{stage}_history.csv"),
        tf.keras.callbacks.TensorBoard(log_dir=run_paths["logs"] / stage),
        LearningRateLogger().callback(),
    ]


def save_stage_plots(history: pd.DataFrame, plot_dir: Path, stage: str) -> None:
    for metric, filename in (("accuracy", f"{stage}_accuracy.png"), ("loss", f"{stage}_loss.png")):
        plt.figure(figsize=(9, 5))
        for column in history.columns:
            if metric in column:
                plt.plot(history[column].values, label=column)
        plt.title(f"{stage} {metric}")
        plt.xlabel("Epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / filename)
        plt.close()
    if "learning_rate" in history.columns:
        plt.figure(figsize=(9, 5))
        plt.plot(history["learning_rate"].values, label="learning_rate")
        plt.title(f"{stage} learning rate")
        plt.xlabel("Epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{stage}_learning_rate.png")
        plt.close()


def save_combined_plots(stage1: pd.DataFrame, stage2: pd.DataFrame, plot_dir: Path) -> None:
    combined = pd.concat([stage1.assign(stage="stage1"), stage2.assign(stage="stage2")], ignore_index=True)
    boundary = len(stage1)
    for metric, filename in (("accuracy", "combined_accuracy.png"), ("loss", "combined_loss.png")):
        plt.figure(figsize=(10, 5))
        for column in combined.columns:
            if metric in column:
                plt.plot(range(1, len(combined) + 1), combined[column].values, label=column)
        plt.axvline(boundary + 0.5, linestyle="--", color="black", label="fine-tuning starts")
        plt.title(f"Combined {metric}")
        plt.xlabel("Epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / filename)
        plt.close()
    if "learning_rate" in combined.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(combined) + 1), combined["learning_rate"].values, label="learning_rate")
        plt.axvline(boundary + 0.5, linestyle="--", color="black", label="fine-tuning starts")
        plt.title("Combined learning rate")
        plt.xlabel("Epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "combined_learning_rate.png")
        plt.close()


def validation_metrics(tf, model_path: Path, validation_csv: Path, validation_ds, config: dict[str, Any], labels: list[str]) -> tuple[dict[str, Any], Any]:
    loss_object = get_sparse_smoothing_loss(
        int(config["model"]["num_classes"]),
        float(config["training"].get("label_smoothing", 0.0)),
    )
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "NutriSnap>SparseCategoricalCrossentropyWithLabelSmoothing": loss_object.__class__,
            "SparseCategoricalCrossentropyWithLabelSmoothing": loss_object.__class__,
        },
    )
    eval_values = model.evaluate(validation_ds, verbose=0)
    probs = model.predict(validation_ds, verbose=0)
    y_true = read_manifest(validation_csv)["class_index"].to_numpy()
    y_pred = np.argmax(probs, axis=1)
    top3 = np.argsort(probs, axis=1)[:, -3:]
    metrics = {
        "val_loss": float(eval_values[0]),
        "top1_accuracy": float(np.mean(y_pred == y_true)),
        "top3_accuracy": float(np.mean([truth in row for truth, row in zip(y_true, top3)])),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if any(np.isnan(value) for value in metrics.values()):
        raise RuntimeError(f"NaN validation metrics detected for {model_path}")
    return metrics, model


def candidate_selection(stage1: dict[str, Any], stage2: dict[str, Any], paths: dict[str, str]) -> dict[str, Any]:
    delta = float(stage2["macro_f1"]) - float(stage1["macro_f1"])
    if delta >= -0.01:
        selected = "stage2"
        reason = "stage2 macro_f1 is not materially worse than stage1; selecting fine-tuned candidate"
    else:
        selected = "stage1"
        reason = "stage2 macro_f1 decreased materially; selecting stage1 head checkpoint"
    if abs(delta) < 1e-6 and float(stage2["val_loss"]) > float(stage1["val_loss"]):
        selected = "stage1"
        reason = "macro_f1 tie; stage1 has lower validation loss"
    return {
        "selected_stage": selected,
        "checkpoint": paths[selected],
        "selection_dataset": "validation",
        "selection_metric": "macro_f1",
        "reason": reason,
        "test_set_used": False,
    }


def run_prediction_checks(tf, model_path: Path, validation_ds, labels: list[str], config: dict[str, Any]) -> dict[str, Any]:
    loss_object = get_sparse_smoothing_loss(
        int(config["model"]["num_classes"]),
        float(config["training"].get("label_smoothing", 0.0)),
    )
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "NutriSnap>SparseCategoricalCrossentropyWithLabelSmoothing": loss_object.__class__,
            "SparseCategoricalCrossentropyWithLabelSmoothing": loss_object.__class__,
        },
    )
    images, _labels = next(iter(validation_ds))
    probs = model.predict(images, verbose=0)
    return {
        "checkpoint_load": True,
        "prediction_shape": list(probs.shape),
        "exactly_15_output_classes": int(probs.shape[-1]) == len(labels),
        "no_nan": bool(not np.isnan(probs).any()),
        "no_inf": bool(not np.isinf(probs).any()),
        "probabilities_sum_approximately_one": bool(np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-3)),
    }


def smoke_test(tf, config: dict[str, Any], train_ds, validation_ds, run_paths: dict[str, Path], labels: list[str]) -> dict[str, Any]:
    smoke_cfg = config.get("smoke_test", {})
    model = build_mobilenetv2(config, train_backbone=False)
    freeze_backbone(model)
    compile_model(config, model, float(config["training"]["initial_learning_rate"]))
    train_batches = int(smoke_cfg.get("train_batches", 3))
    validation_batches = int(smoke_cfg.get("validation_batches", 2))
    history = model.fit(
        train_ds.take(train_batches),
        validation_data=validation_ds.take(validation_batches),
        epochs=int(smoke_cfg.get("epochs", 1)),
        verbose=1,
    )
    checkpoint = run_paths["checkpoints"] / "smoke_checkpoint.keras"
    model.save(checkpoint)
    loss_object = get_sparse_smoothing_loss(
        int(config["model"]["num_classes"]),
        float(config["training"].get("label_smoothing", 0.0)),
    )
    loaded = tf.keras.models.load_model(
        checkpoint,
        custom_objects={
            "NutriSnap>SparseCategoricalCrossentropyWithLabelSmoothing": loss_object.__class__,
            "SparseCategoricalCrossentropyWithLabelSmoothing": loss_object.__class__,
        },
    )
    images, _labels = next(iter(validation_ds.take(1)))
    probs = loaded.predict(images, verbose=0)
    result = {
        "status": "passed",
        "train_batches": train_batches,
        "validation_batches": validation_batches,
        "output_shape": list(probs.shape),
        "checkpoint_save_load": True,
        "no_nan": bool(not np.isnan(probs).any()),
        "no_inf": bool(not np.isinf(probs).any()),
        "probabilities_sum_approximately_one": bool(np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-3)),
        "history": {key: [float(v) for v in value] for key, value in history.history.items()},
    }
    if result["output_shape"][-1] != len(labels) or not result["no_nan"] or not result["no_inf"] or not result["probabilities_sum_approximately_one"]:
        raise RuntimeError(f"Smoke prediction checks failed: {result}")
    return result


def update_model_selection_template(run_dir: Path, stage1: dict[str, Any], stage2: dict[str, Any], selected: dict[str, Any], repo_root: Path) -> None:
    text = f"""# Model Selection

Status: Pending Phase AI-7 test evaluation and Pending Phase AI-8 TFLite validation

## Training Setup

Run directory: `{run_dir}`

Selection dataset: validation only. Test set used: false.

## Stage 1 Validation Metrics

- Validation loss: {stage1['val_loss']:.6f}
- Top-1 accuracy: {stage1['top1_accuracy']:.6f}
- Top-3 accuracy: {stage1['top3_accuracy']:.6f}
- Macro F1: {stage1['macro_f1']:.6f}

## Stage 2 Validation Metrics

- Validation loss: {stage2['val_loss']:.6f}
- Top-1 accuracy: {stage2['top1_accuracy']:.6f}
- Top-3 accuracy: {stage2['top3_accuracy']:.6f}
- Macro F1: {stage2['macro_f1']:.6f}

## Candidate

Selected stage: `{selected['selected_stage']}`

Checkpoint: `{selected['checkpoint']}`

Reason: {selected['reason']}

## Test Metrics

Pending Phase AI-7.

## TFLite Metrics

Pending Phase AI-8.
"""
    (repo_root / "ai_training" / "outputs" / "reports" / "model_selection.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    setup_logging()
    config, config_path, repo_root = load_config(args.config)
    labels = validate_label_sources(config, repo_root)
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.initial_epochs is not None:
        config["training"]["initial_epochs"] = args.initial_epochs
    if args.fine_tune_epochs is not None:
        config["training"]["fine_tune_epochs"] = args.fine_tune_epochs

    tf, gpus = configure_tensorflow(config, require_gpu=args.require_gpu, no_mixed_precision=args.no_mixed_precision)
    seed = int(config["project"]["seed"])
    set_global_seed(seed)

    processed_dir = resolve_project_path(config["dataset"]["processed_dir"], repo_root)
    train_csv = processed_dir / "train.csv"
    validation_csv = processed_dir / "validation.csv"
    dataset_manifest_path = processed_dir / "dataset_manifest.json"
    for path in (train_csv, validation_csv, dataset_manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"Required training input not found: {path}")
    dataset_manifest = read_json(dataset_manifest_path)
    if dataset_manifest.get("split_protocol") != config["dataset"].get("split_protocol"):
        raise RuntimeError("Dataset split protocol is not locked. Run lock_training_protocol.py first.")

    run_dir = make_run_dir(config, repo_root, smoke_test=args.smoke_test, resume_run=args.resume_run)
    run_paths = prepare_run_structure(run_dir)
    if args.resume_run:
        validate_resume(run_dir, config, dataset_manifest)

    train_manifest = read_manifest(train_csv)
    validation_manifest = read_manifest(validation_csv)
    if set(train_manifest["filepath"]).intersection(set(validation_manifest["filepath"])):
        raise RuntimeError("Train/validation filepath overlap detected.")
    if set(train_manifest["sha256"]).intersection(set(validation_manifest["sha256"])):
        raise RuntimeError("Train/validation SHA-256 overlap detected.")

    train_ds = build_dataset(train_csv, config, training=True)
    validation_ds = build_dataset(validation_csv, config, training=False)
    batch_audit = {
        "train": audit_batch(train_ds, labels, "train"),
        "validation": audit_batch(validation_ds, labels, "validation"),
    }

    env = collect_environment(tf, seed, config, dataset_manifest, len(train_manifest), len(validation_manifest))
    for key, value in env.items():
        logging.info("%s: %s", key, value)

    shutil.copy2(config_path, run_dir / "config_resolved.yaml")
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    shutil.copy2(dataset_manifest_path, run_dir / "dataset_manifest_snapshot.json")
    class_weights = compute_class_weights(train_manifest, labels) if config["dataset"].get("use_class_weights", True) else None
    write_json(run_dir / "class_weights.json", {labels[index]: weight for index, weight in (class_weights or {}).items()})

    metadata = {
        **env,
        "run_type": "smoke_test" if args.smoke_test else "training",
        "git_commit": get_git_commit(repo_root),
        "class_order": labels,
        "dataset_manifest_hash": dataset_manifest.get("manifest_sha256"),
        "test_manifest_loaded": False,
        "test_evaluation_executed": False,
        "input_shape": input_shape(config),
        "batch_audit": batch_audit,
    }
    write_json(run_dir / "run_metadata.json", metadata)
    latest_run = resolve_project_path("ai_training/outputs/latest_run.txt", repo_root)
    latest_run.write_text(str(run_dir), encoding="utf-8")

    try:
        if args.smoke_test:
            result = smoke_test(tf, config, train_ds, validation_ds, run_paths, labels)
            write_json(run_paths["reports"] / "smoke_test_result.json", result)
            logging.info("Smoke test passed. Run directory: %s", run_dir)
            return 0

        if args.resume_run:
            stage1_history = pd.read_csv(run_paths["histories"] / "stage1_history.csv")
            stage1_metrics = read_json(run_paths["reports"] / "stage1_validation_metrics.json")
        else:
            model = build_mobilenetv2(config, train_backbone=False)
            freeze_backbone(model)
            compile_model(config, model, float(config["training"]["initial_learning_rate"]))
            (run_dir / "model_summary.txt").write_text(model_summary_text(model), encoding="utf-8")
            logging.info("Stage 1 parameter report: %s", parameter_report(model))
            best_head = run_paths["checkpoints"] / "best_head.keras"
            last_head = run_paths["checkpoints"] / "last_head.keras"
            head_history_obj = model.fit(
                train_ds,
                validation_data=validation_ds,
                epochs=int(config["training"]["initial_epochs"]),
                callbacks=make_callbacks(tf, run_paths, "stage1", best_head, last_head, config),
                class_weight=class_weights,
            )
            stage1_history = pd.DataFrame(head_history_obj.history)
            stage1_history.to_csv(run_paths["histories"] / "stage1_history.csv", index=False)
            save_stage_plots(stage1_history, run_paths["plots"], "stage1")
            stage1_metrics, stage1_model = validation_metrics(tf, best_head, validation_csv, validation_ds, config, labels)
            stage1_metrics.update(parameter_report(stage1_model))
            write_json(run_paths["reports"] / "stage1_validation_metrics.json", stage1_metrics)

        best_head = run_paths["checkpoints"] / "best_head.keras"
        if not best_head.exists():
            raise FileNotFoundError(f"Stage 2 requires Stage 1 checkpoint: {best_head}")
        loss_object = get_sparse_smoothing_loss(int(config["model"]["num_classes"]), float(config["training"].get("label_smoothing", 0.0)))
        model = tf.keras.models.load_model(
            best_head,
            custom_objects={
                "NutriSnap>SparseCategoricalCrossentropyWithLabelSmoothing": loss_object.__class__,
                "SparseCategoricalCrossentropyWithLabelSmoothing": loss_object.__class__,
            },
        )
        set_fine_tuning(
            model,
            int(config["training"].get("fine_tune_last_layers", 50)),
            bool(config["training"].get("freeze_batch_normalization", True)),
        )
        compile_model(config, model, float(config["training"]["fine_tune_learning_rate"]))
        logging.info("Stage 2 parameter report: %s", parameter_report(model))
        best_finetuned = run_paths["checkpoints"] / "best_finetuned.keras"
        last_finetuned = run_paths["checkpoints"] / "last_finetuned.keras"
        fine_history_obj = model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=int(config["training"]["fine_tune_epochs"]),
            callbacks=make_callbacks(tf, run_paths, "stage2", best_finetuned, last_finetuned, config),
            class_weight=class_weights,
        )
        stage2_history = pd.DataFrame(fine_history_obj.history)
        stage2_history.to_csv(run_paths["histories"] / "stage2_history.csv", index=False)
        combined_history = pd.concat([stage1_history.assign(stage="stage1"), stage2_history.assign(stage="stage2")], ignore_index=True)
        combined_history.to_csv(run_paths["histories"] / "combined_history.csv", index=False)
        save_stage_plots(stage2_history, run_paths["plots"], "stage2")
        save_combined_plots(stage1_history, stage2_history, run_paths["plots"])
        stage2_metrics, stage2_model = validation_metrics(tf, best_finetuned, validation_csv, validation_ds, config, labels)
        stage2_metrics.update(parameter_report(stage2_model))
        write_json(run_paths["reports"] / "stage2_validation_metrics.json", stage2_metrics)

        comparison = pd.DataFrame(
            [
                {
                    "model_stage": "stage1",
                    "checkpoint": "checkpoints/best_head.keras",
                    **stage1_metrics,
                    "parameter_count": stage1_metrics["total_parameters"],
                    "trainable_parameter_count": stage1_metrics["trainable_parameters"],
                },
                {
                    "model_stage": "stage2",
                    "checkpoint": "checkpoints/best_finetuned.keras",
                    **stage2_metrics,
                    "parameter_count": stage2_metrics["total_parameters"],
                    "trainable_parameter_count": stage2_metrics["trainable_parameters"],
                },
            ]
        )
        comparison.to_csv(run_paths["reports"] / "validation_model_comparison.csv", index=False)
        selected = candidate_selection(
            stage1_metrics,
            stage2_metrics,
            {"stage1": "checkpoints/best_head.keras", "stage2": "checkpoints/best_finetuned.keras"},
        )
        write_json(run_paths["reports"] / "selected_candidate.json", selected)

        selected_checkpoint = run_dir / selected["checkpoint"]
        prediction_checks = run_prediction_checks(tf, selected_checkpoint, validation_ds, labels, config)
        validation_checks = {
            "no_train_validation_filepath_overlap": True,
            "no_train_validation_sha256_overlap": True,
            "exactly_15_output_classes": prediction_checks["exactly_15_output_classes"],
            "class_order_matches_food_labels": labels == load_labels(repo_root),
            "no_nan_metrics": not any(np.isnan(v) for metrics in (stage1_metrics, stage2_metrics) for v in metrics.values() if isinstance(v, float)),
            "checkpoint_can_be_loaded": prediction_checks["checkpoint_load"],
            "checkpoint_can_predict_one_batch": True,
            "probability_outputs_sum_approximately_to_one": prediction_checks["probabilities_sum_approximately_one"],
            "test_manifest_loaded": False,
            "test_evaluation_executed": False,
        }
        write_json(run_paths["reports"] / "training_validation_checks.json", validation_checks)
        summary = f"""# Training Summary

Run directory: `{run_dir}`

Test set used: false.

## Stage 1

- Epochs executed: {len(stage1_history)}
- Validation loss: {stage1_metrics['val_loss']:.6f}
- Validation top-1 accuracy: {stage1_metrics['top1_accuracy']:.6f}
- Validation top-3 accuracy: {stage1_metrics['top3_accuracy']:.6f}
- Validation macro F1: {stage1_metrics['macro_f1']:.6f}

## Stage 2

- Epochs executed: {len(stage2_history)}
- Validation loss: {stage2_metrics['val_loss']:.6f}
- Validation top-1 accuracy: {stage2_metrics['top1_accuracy']:.6f}
- Validation top-3 accuracy: {stage2_metrics['top3_accuracy']:.6f}
- Validation macro F1: {stage2_metrics['macro_f1']:.6f}

## Selected Candidate

- Stage: {selected['selected_stage']}
- Checkpoint: {selected['checkpoint']}
- Reason: {selected['reason']}
"""
        (run_paths["reports"] / "training_summary.md").write_text(summary, encoding="utf-8")
        update_model_selection_template(run_dir, stage1_metrics, stage2_metrics, selected, repo_root)
        logging.info("Training complete. Run directory: %s", run_dir)
        return 0
    except tf.errors.ResourceExhaustedError as exc:
        error_report = {
            "error": "ResourceExhaustedError",
            "batch_size": int(config["training"]["batch_size"]),
            "suggested_command": f"python3 ai_training/scripts/train.py --config {args.config} --batch-size 16",
            "traceback": traceback.format_exc(),
        }
        write_json(run_paths["reports"] / "oom_error.json", error_report)
        logging.error("OOM at batch size %s. Try --batch-size 16.", config["training"]["batch_size"])
        raise SystemExit(1) from exc


if __name__ == "__main__":
    raise SystemExit(main())
