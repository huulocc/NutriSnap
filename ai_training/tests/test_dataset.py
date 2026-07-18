from __future__ import annotations

from pathlib import Path

import pandas as pd

from ai_training.src.dataset import REQUIRED_MANIFEST_COLUMNS


def load_manifest_if_present(repo_root: Path, split: str) -> pd.DataFrame | None:
    path = repo_root / "ai_training" / "data" / "processed" / f"{split}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def test_manifest_columns_when_present() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    for split in ("train", "validation", "test"):
        manifest = load_manifest_if_present(repo_root, split)
        if manifest is not None:
            assert REQUIRED_MANIFEST_COLUMNS.issubset(manifest.columns)


def test_no_filepath_or_hash_overlap_between_splits_when_present() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifests = {split: load_manifest_if_present(repo_root, split) for split in ("train", "validation", "test")}
    if any(value is None for value in manifests.values()):
        return
    split_items = {
        split: {
            "filepath": set(df["filepath"]),
            "sha256": set(df["sha256"]),
        }
        for split, df in manifests.items()
        if df is not None
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert split_items[left]["filepath"].isdisjoint(split_items[right]["filepath"])
        assert split_items[left]["sha256"].isdisjoint(split_items[right]["sha256"])


def test_train_manifest_contains_all_classes_when_present() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    train = load_manifest_if_present(repo_root, "train")
    if train is None:
        return
    assert set(train["class_index"]) == set(range(15))
