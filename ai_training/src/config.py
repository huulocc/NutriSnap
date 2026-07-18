from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path("ai_training/configs/mobilenetv2_config.yaml")


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for path in (current, *current.parents):
        if (path / ".git").exists() or (path / "settings.gradle.kts").exists():
            return path
    return current


def resolve_project_path(path_value: str | Path, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], Path, Path]:
    repo_root = find_repo_root()
    config = Path(config_path)
    if not config.is_absolute():
        config = (repo_root / config).resolve()
    if not config.exists():
        raise FileNotFoundError(f"Config file not found: {config}")

    with config.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config file is empty or invalid: {config}")
    return data, config, repo_root


def load_labels(repo_root: Path) -> list[str]:
    labels_path = repo_root / "ai_training" / "metadata" / "food_labels.txt"
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != len(set(labels)):
        raise ValueError("Duplicate labels found in food_labels.txt")
    return labels


def load_class_names(repo_root: Path) -> dict[int, dict[str, str]]:
    path = repo_root / "ai_training" / "metadata" / "class_names.json"
    if not path.exists():
        raise FileNotFoundError(f"Class metadata file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(index): value for index, value in raw.items()}


def validate_label_sources(config: dict[str, Any], repo_root: Path) -> list[str]:
    yaml_labels = list(config["dataset"]["classes"])
    file_labels = load_labels(repo_root)
    class_names = load_class_names(repo_root)
    json_labels = [class_names[index]["model_label"] for index in sorted(class_names)]

    if yaml_labels != file_labels or yaml_labels != json_labels:
        raise ValueError("Class order mismatch between YAML, food_labels.txt, and class_names.json")
    if len(yaml_labels) != 15:
        raise ValueError(f"Expected 15 classes, found {len(yaml_labels)}")
    for index, item in class_names.items():
        if not item.get("display_name"):
            raise ValueError(f"Missing display_name for class index {index}")
    return yaml_labels
