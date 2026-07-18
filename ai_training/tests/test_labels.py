from __future__ import annotations

from pathlib import Path

from ai_training.src.config import load_class_names, load_config, load_labels, validate_label_sources


def test_label_sources_are_consistent() -> None:
    config, _, repo_root = load_config("ai_training/configs/mobilenetv2_config.yaml")
    labels = validate_label_sources(config, repo_root)
    assert labels == load_labels(repo_root)
    assert len(labels) == 15
    assert len(set(labels)) == 15


def test_class_indexes_are_contiguous() -> None:
    _config, _, repo_root = load_config("ai_training/configs/mobilenetv2_config.yaml")
    class_names = load_class_names(repo_root)
    assert sorted(class_names) == list(range(15))
    assert all(class_names[index]["display_name"] for index in range(15))


def test_labels_file_order() -> None:
    expected = [
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
    repo_root = Path(__file__).resolve().parents[2]
    assert load_labels(repo_root) == expected
