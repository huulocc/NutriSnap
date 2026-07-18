from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXCLUDED_DIR_NAMES = {
    ".git",
    ".idea",
    ".venv",
    "venv",
    "__pycache__",
    "processed",
    "outputs",
    "checkpoints",
    "logs",
    "runs",
    "tflite",
    "downloads",
}


@dataclass(frozen=True)
class SourceClassFolder:
    source_folder: Path
    normalized_source_name: str
    canonical_label: str | None
    image_count: int
    status: str


def strip_accents(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def normalize_source_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = strip_accents(normalized)
    normalized = normalized.lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def load_source_mapping(repo_root: Path, selected_labels: list[str]) -> dict[str, str]:
    mapping_path = repo_root / "ai_training" / "metadata" / "source_class_mapping.json"
    raw = json.loads(mapping_path.read_text(encoding="utf-8"))
    normalized: dict[str, str] = {}
    for source_name, canonical in raw.items():
        if canonical not in selected_labels:
            raise ValueError(f"Mapping points to unknown selected class: {source_name} -> {canonical}")
        normalized[normalize_source_name(source_name)] = canonical
    for canonical in selected_labels:
        normalized[normalize_source_name(canonical)] = canonical
    return normalized


def has_image_files(path: Path, extensions: set[str]) -> bool:
    return any(child.is_file() and child.suffix.lower() in extensions for child in path.iterdir())


def count_images(path: Path, extensions: set[str]) -> int:
    return sum(1 for child in path.iterdir() if child.is_file() and child.suffix.lower() in extensions)


def should_skip_dir(path: Path, source_root: Path) -> bool:
    if path == source_root:
        return False
    return any(part in EXCLUDED_DIR_NAMES for part in path.relative_to(source_root).parts)


def discover_source_class_folders(
    source_root: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> list[SourceClassFolder]:
    selected_labels = list(config["dataset"]["classes"])
    extensions = {item.lower() for item in config["dataset"]["image_extensions"]}
    source_mapping = load_source_mapping(repo_root, selected_labels)
    folders: list[SourceClassFolder] = []

    if not source_root.exists():
        return [
            SourceClassFolder(
                source_folder=source_root,
                normalized_source_name=normalize_source_name(source_root.name),
                canonical_label=None,
                image_count=0,
                status="source_root_missing",
            )
        ]

    for folder in sorted([source_root, *[path for path in source_root.rglob("*") if path.is_dir()]]):
        if should_skip_dir(folder, source_root):
            continue
        if not has_image_files(folder, extensions):
            continue
        normalized_name = normalize_source_name(folder.name)
        canonical = source_mapping.get(normalized_name)
        folders.append(
            SourceClassFolder(
                source_folder=folder,
                normalized_source_name=normalized_name,
                canonical_label=canonical,
                image_count=count_images(folder, extensions),
                status="mapped" if canonical else "unknown_source_class",
            )
        )

    counts: dict[str, int] = {}
    for folder in folders:
        if folder.canonical_label:
            counts[folder.canonical_label] = counts.get(folder.canonical_label, 0) + 1

    updated: list[SourceClassFolder] = []
    for folder in folders:
        status = folder.status
        if folder.canonical_label and counts.get(folder.canonical_label, 0) > 1:
            status = "duplicate_source_folder"
        updated.append(
            SourceClassFolder(
                source_folder=folder.source_folder,
                normalized_source_name=folder.normalized_source_name,
                canonical_label=folder.canonical_label,
                image_count=folder.image_count,
                status=status,
            )
        )
    return updated


def selected_folder_map(folders: list[SourceClassFolder]) -> dict[str, list[SourceClassFolder]]:
    result: dict[str, list[SourceClassFolder]] = {}
    for folder in folders:
        if folder.canonical_label:
            result.setdefault(folder.canonical_label, []).append(folder)
    return result


def exact_duplicate_groups(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        digest = row.get("sha256")
        if digest:
            groups.setdefault(digest, []).append(row)
    return {digest: items for digest, items in groups.items() if len(items) > 1}


def cross_class_duplicate_hashes(rows: list[dict]) -> set[str]:
    result: set[str] = set()
    for digest, items in exact_duplicate_groups(rows).items():
        labels = {item.get("model_label") or item.get("canonical_label") for item in items}
        labels.discard(None)
        if len(labels) > 1:
            result.add(digest)
    return result
