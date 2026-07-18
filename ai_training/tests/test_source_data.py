from __future__ import annotations

from pathlib import Path

from PIL import Image

from ai_training.scripts.prepare_dataset import assign_groups
from ai_training.src.config import load_config, validate_label_sources
from ai_training.src.source_data import (
    cross_class_duplicate_hashes,
    discover_source_class_folders,
    exact_duplicate_groups,
    load_source_mapping,
    normalize_source_name,
)


def write_image(path: Path, color: tuple[int, int, int] = (255, 0, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)


def test_source_folder_normalization() -> None:
    assert normalize_source_name("  Bún_bò-Huế  ") == "bun bo hue"
    assert normalize_source_name("Xôi   xéo") == "xoi xeo"


def test_source_folder_mapping() -> None:
    config, _, repo_root = load_config("ai_training/configs/mobilenetv2_config.yaml")
    labels = validate_label_sources(config, repo_root)
    mapping = load_source_mapping(repo_root, labels)
    assert mapping[normalize_source_name("Phở")] == "pho"
    assert mapping[normalize_source_name("Banh trang nuong")] == "banh_trang_nuong"


def test_nested_dataset_discovery_and_unknown_folder(tmp_path: Path) -> None:
    config, _, repo_root = load_config("ai_training/configs/mobilenetv2_config.yaml")
    source = tmp_path / "Vietnamese Foods" / "Images"
    write_image(source / "Phở" / "a.jpg")
    write_image(source / "Unknown Dish" / "b.jpg")
    folders = discover_source_class_folders(tmp_path, config, repo_root)
    mapped = [folder for folder in folders if folder.canonical_label == "pho"]
    unknown = [folder for folder in folders if folder.status == "unknown_source_class"]
    assert len(mapped) == 1
    assert mapped[0].image_count == 1
    assert len(unknown) == 1


def test_duplicate_grouping_and_cross_class_detection() -> None:
    rows = [
        {"filepath": "a.jpg", "model_label": "pho", "sha256": "same"},
        {"filepath": "b.jpg", "model_label": "pho", "sha256": "same"},
        {"filepath": "c.jpg", "model_label": "banh_mi", "sha256": "cross"},
        {"filepath": "d.jpg", "model_label": "com_tam", "sha256": "cross"},
    ]
    assert set(exact_duplicate_groups(rows)) == {"same", "cross"}
    assert cross_class_duplicate_hashes(rows) == {"cross"}


def test_split_leakage_prevention_for_duplicate_group() -> None:
    split_map = assign_groups(["duplicate_hash"], (0.8, 0.1, 0.1), seed=42)
    assert set(split_map) == {"duplicate_hash"}
    assert split_map["duplicate_hash"] in {"train", "validation", "test"}
