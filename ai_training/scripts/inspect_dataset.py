from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from PIL import Image, UnidentifiedImageError

from ai_training.src.config import load_class_names, load_config, resolve_project_path, validate_label_sources
from ai_training.src.source_data import discover_source_class_folders
from ai_training.src.utils import ensure_dirs, setup_logging, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the source food image dataset.")
    parser.add_argument("--config", default="ai_training/configs/mobilenetv2_config.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    config, _, repo_root = load_config(args.config)
    labels = validate_label_sources(config, repo_root)
    class_names = load_class_names(repo_root)
    source_dir = resolve_project_path(config["dataset"]["source_dir"], repo_root)
    report_dir = resolve_project_path(config["output"]["report_dir"], repo_root)
    ensure_dirs(report_dir)

    folders = discover_source_class_folders(source_dir, config, repo_root)
    selected_found = {folder.canonical_label for folder in folders if folder.canonical_label}
    missing = [label for label in labels if label not in selected_found]
    logging.info("Found selected classes: %s", sorted(selected_found))
    logging.info("Missing selected classes: %s", missing)

    mapping_rows = [
        {
            "source_folder": str(folder.source_folder),
            "normalized_source_name": folder.normalized_source_name,
            "canonical_label": folder.canonical_label or "",
            "image_count": folder.image_count,
            "status": folder.status,
        }
        for folder in folders
    ]
    pd.DataFrame(mapping_rows, columns=["source_folder", "normalized_source_name", "canonical_label", "image_count", "status"]).to_csv(
        report_dir / "source_class_mapping.csv", index=False
    )
    pd.DataFrame({"canonical_label": missing}).to_csv(report_dir / "missing_selected_classes.csv", index=False)
    ignored = [row for row in mapping_rows if row["status"] == "unknown_source_class"]
    pd.DataFrame(ignored, columns=["source_folder", "normalized_source_name", "canonical_label", "image_count", "status"]).to_csv(
        report_dir / "ignored_source_classes.csv", index=False
    )

    extensions = {item.lower() for item in config["dataset"]["image_extensions"]}
    min_size = int(config["dataset"].get("min_image_size", 64))
    summary_rows: list[dict] = []
    corrupted_rows: list[dict] = []
    by_hash: dict[str, list[dict]] = defaultdict(list)

    for folder in folders:
        if not folder.canonical_label:
            continue
        class_index = labels.index(folder.canonical_label)
        for path in sorted(folder.source_folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            try:
                digest = sha256_file(path)
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    width, height = image.size
                if width <= 0 or height <= 0:
                    issue = "zero_dimension"
                elif width < min_size or height < min_size:
                    issue = "too_small"
                else:
                    issue = ""
                row = {
                    "filepath": str(path.resolve()),
                    "source_folder": str(folder.source_folder),
                    "source_class": folder.source_folder.name,
                    "class_index": class_index,
                    "model_label": folder.canonical_label,
                    "display_name": class_names[class_index]["display_name"],
                    "sha256": digest,
                    "width": width,
                    "height": height,
                    "aspect_ratio": width / height if height else 0,
                    "file_size_bytes": path.stat().st_size,
                    "issue": issue,
                }
                summary_rows.append(row)
                by_hash[digest].append(row)
                if issue:
                    corrupted_rows.append(
                        {
                            "filepath": str(path.resolve()),
                            "source_class": folder.source_folder.name,
                            "canonical_label": folder.canonical_label,
                            "reason": issue,
                            "sha256": digest,
                        }
                    )
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                corrupted_rows.append(
                    {
                        "filepath": str(path.resolve()),
                        "source_class": folder.source_folder.name,
                        "canonical_label": folder.canonical_label,
                        "reason": f"unreadable:{exc}",
                        "sha256": "",
                    }
                )

    exact_duplicates = []
    cross_class_duplicates = []
    for digest, rows in by_hash.items():
        if len(rows) < 2:
            continue
        labels_for_hash = sorted({row["model_label"] for row in rows})
        duplicate_row = {
            "sha256": digest,
            "count": len(rows),
            "canonical_labels": "|".join(labels_for_hash),
            "filepaths": "|".join(row["filepath"] for row in rows),
        }
        exact_duplicates.append(duplicate_row)
        if len(labels_for_hash) > 1:
            cross_class_duplicates.append(duplicate_row)

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        class_distribution = pd.DataFrame({"model_label": labels, "image_count": [0] * len(labels)})
    else:
        class_distribution = summary.groupby("model_label").size().reindex(labels, fill_value=0).reset_index(name="image_count")

    pd.DataFrame(corrupted_rows, columns=["filepath", "source_class", "canonical_label", "reason", "sha256"]).to_csv(
        report_dir / "corrupted_images.csv", index=False
    )
    pd.DataFrame(exact_duplicates, columns=["sha256", "count", "canonical_labels", "filepaths"]).to_csv(
        report_dir / "exact_duplicates.csv", index=False
    )
    pd.DataFrame(cross_class_duplicates, columns=["sha256", "count", "canonical_labels", "filepaths"]).to_csv(
        report_dir / "cross_class_duplicates.csv", index=False
    )
    class_distribution.to_csv(report_dir / "class_distribution.csv", index=False)

    dataset_summary = {
        "source_dir": str(source_dir),
        "total_images": int(len(summary_rows)),
        "selected_classes_found": sorted(selected_found),
        "missing_selected_classes": missing,
        "unknown_folder_count": len(ignored),
        "corrupted_images": len(corrupted_rows),
        "exact_duplicate_groups": len(exact_duplicates),
        "cross_class_duplicate_groups": len(cross_class_duplicates),
        "class_distribution": dict(zip(class_distribution["model_label"], class_distribution["image_count"])),
    }
    (report_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Inspection reports written to %s", report_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
