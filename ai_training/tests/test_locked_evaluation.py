from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_training.scripts.evaluate_locked_test import (
    EXPECTED_COUNTS,
    EXPECTED_LABELS,
    EXPECTED_MANIFEST_SHA,
    EXPECTED_MODEL,
    bootstrap_ci,
    compute_metrics,
    threshold_analysis,
)
from ai_training.src.config import load_config, load_labels


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_locked_checkpoint_path() -> None:
    assert EXPECTED_MODEL == "ai_training/outputs/runs/20260717_162321_mobilenetv2_continuation/checkpoints/best_continued.keras"


def test_locked_manifest_sha_when_present() -> None:
    path = repo_root() / "ai_training" / "data" / "processed" / "dataset_manifest.json"
    if not path.exists():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["manifest_sha256"] == EXPECTED_MANIFEST_SHA


def test_locked_class_order() -> None:
    assert load_labels(repo_root()) == EXPECTED_LABELS


def test_locked_config_output_class_count() -> None:
    config, _, _repo = load_config("ai_training/configs/mobilenetv2_config.yaml")
    assert int(config["model"]["num_classes"]) == 15


def test_locked_no_split_overlap_when_present() -> None:
    base = repo_root() / "ai_training" / "data" / "processed"
    paths = {split: base / f"{split}.csv" for split in ("train", "validation", "test")}
    if not all(path.exists() for path in paths.values()):
        return
    manifests = {split: pd.read_csv(path) for split, path in paths.items()}
    assert len(manifests["test"]) == EXPECTED_COUNTS["test"]
    for left, right in (("train", "test"), ("validation", "test")):
        assert set(manifests[left]["filepath"]).isdisjoint(set(manifests[right]["filepath"]))
        assert set(manifests[left]["sha256"]).isdisjoint(set(manifests[right]["sha256"]))


def test_locked_metrics_are_finite_on_synthetic_probabilities() -> None:
    tf = pytest.importorskip("tensorflow", exc_type=ImportError)

    config, _, _repo = load_config("ai_training/configs/mobilenetv2_config.yaml")
    y_true = np.array([0, 1, 2, 3])
    probs = np.full((4, 15), 0.01, dtype=np.float32)
    probs[np.arange(4), y_true] = 0.86
    probs = probs / probs.sum(axis=1, keepdims=True)
    metrics = compute_metrics(tf, y_true, probs, config, EXPECTED_LABELS)
    for key, value in metrics.items():
        if isinstance(value, float):
            assert np.isfinite(value)
            if key != "test_loss":
                assert 0.0 <= value <= 1.0


def test_locked_probability_sums_on_synthetic_probabilities() -> None:
    probs = np.eye(15, dtype=np.float32)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_locked_bootstrap_ci_shape() -> None:
    y_true = np.array([0, 1, 2, 3, 4])
    probs = np.full((5, 15), 0.01, dtype=np.float32)
    probs[np.arange(5), y_true] = 0.86
    probs = probs / probs.sum(axis=1, keepdims=True)
    ci = bootstrap_ci(y_true, probs, EXPECTED_LABELS, seed=42, resamples=10)
    assert ci["method"] == "percentile_bootstrap_95_ci"
    assert set(ci) >= {"top1_accuracy", "macro_f1"}


def test_locked_threshold_analysis_sanity() -> None:
    predictions = pd.DataFrame(
        {
            "confidence": [0.95, 0.75, 0.45],
            "true_index": [0, 1, 2],
            "predicted_index": [0, 3, 2],
        }
    )
    result = threshold_analysis(predictions, 15)
    assert set(result["threshold"]) == {0.40, 0.50, 0.60, 0.70, 0.80}
    assert result["coverage"].between(0, 1).all()


def test_locked_report_contains_all_classes_when_evaluation_exists() -> None:
    latest = repo_root() / "ai_training" / "outputs" / "latest_evaluation.txt"
    if not latest.exists():
        return
    evaluation_dir = Path(latest.read_text(encoding="utf-8").strip())
    report = evaluation_dir / "reports" / "classification_report.csv"
    predictions = evaluation_dir / "reports" / "test_predictions.csv"
    if not report.exists() or not predictions.exists():
        return
    report_frame = pd.read_csv(report)
    prediction_frame = pd.read_csv(predictions)
    assert len(report_frame) == 15
    assert len(prediction_frame) == EXPECTED_COUNTS["test"]
