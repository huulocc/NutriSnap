from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from PIL import Image, UnidentifiedImageError

from ai_training.scripts.download_dataset import DATASET_SLUG
from ai_training.src.config import load_class_names, load_config, resolve_project_path, validate_label_sources
from ai_training.src.source_data import cross_class_duplicate_hashes, discover_source_class_folders
from ai_training.src.utils import ensure_dirs, manifest_hash, setup_logging, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare train/validation/test manifests from source images.")
    parser.add_argument("--config", default="ai_training/configs/mobilenetv2_config.yaml")
    return parser.parse_args()


def assign_groups(group_ids: list[str], ratios: tuple[float, float, float], seed: int) -> dict[str, str]:
    import random

    shuffled = list(group_ids)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_end = max(1, round(total * ratios[0])) if total else 0
    val_count = round(total * ratios[1])
    val_end = min(total, train_end + val_count)
    split_map = {}
    for index, group_id in enumerate(shuffled):
        if index < train_end:
            split_map[group_id] = "train"
        elif index < val_end:
            split_map[group_id] = "validation"
        else:
            split_map[group_id] = "test"
    return split_map


def load_source_archive_sha256(repo_root: Path) -> str | None:
    metadata_path = repo_root / "ai_training" / "data" / "downloads" / "download_metadata.json"
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8")).get("archive_sha256")
    except json.JSONDecodeError:
        return None


def main() -> int:
    args = parse_args()
    setup_logging()
    config, _, repo_root = load_config(args.config)
    labels = validate_label_sources(config, repo_root)
    class_names = load_class_names(repo_root)
    source_dir = resolve_project_path(config["dataset"]["source_dir"], repo_root)
    processed_dir = resolve_project_path(config["dataset"]["processed_dir"], repo_root)
    report_dir = resolve_project_path(config["output"]["report_dir"], repo_root)
    ensure_dirs(processed_dir, report_dir)

    ratios = (
        float(config["dataset"]["train_ratio"]),
        float(config["dataset"]["validation_ratio"]),
        float(config["dataset"]["test_ratio"]),
    )
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")

    folders = discover_source_class_folders(source_dir, config, repo_root)
    selected_folders = [folder for folder in folders if folder.canonical_label]
    if not selected_folders:
        raise RuntimeError(f"No selected class folders found under {source_dir}. Run download_dataset.py or place data in data/source.")

    extensions = {item.lower() for item in config["dataset"]["image_extensions"]}
    min_size = int(config["dataset"].get("min_image_size", 64))
    minimum_images = int(config["dataset"].get("minimum_images_per_class", 30))
    warning_images = int(config["dataset"].get("warning_images_per_class", 200))
    accepted_candidates: list[dict] = []
    rejected_rows: list[dict] = []
    by_hash: dict[str, list[dict]] = defaultdict(list)

    for folder in selected_folders:
        class_index = labels.index(folder.canonical_label)
        for path in sorted(folder.source_folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    width, height = image.size
                digest = sha256_file(path)
                if width <= 0 or height <= 0:
                    reason = "zero_dimension"
                elif width < min_size or height < min_size:
                    reason = "too_small"
                else:
                    reason = ""
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                rejected_rows.append(
                    {
                        "filepath": str(path.resolve()),
                        "source_class": folder.source_folder.name,
                        "canonical_label": folder.canonical_label,
                        "reason": f"unreadable:{exc}",
                        "sha256": "",
                    }
                )
                continue
            row = {
                "filepath": str(path.resolve()),
                "source_class": folder.source_folder.name,
                "class_index": class_index,
                "model_label": folder.canonical_label,
                "display_name": class_names[class_index]["display_name"],
                "sha256": digest,
            }
            if reason:
                rejected_rows.append({**row, "canonical_label": row["model_label"], "reason": reason})
                continue
            accepted_candidates.append(row)
            by_hash[digest].append(row)

    cross_class_hashes = cross_class_duplicate_hashes(accepted_candidates)
    accepted_rows = []
    for row in accepted_candidates:
        if row["sha256"] in cross_class_hashes:
            rejected_rows.append({**row, "canonical_label": row["model_label"], "reason": "cross_class_duplicate"})
        else:
            accepted_rows.append(row)

    if not accepted_rows:
        raise RuntimeError("No valid accepted images found. Dataset manifests were not created.")

    accepted = pd.DataFrame(accepted_rows)
    rejected = pd.DataFrame(
        rejected_rows,
        columns=["filepath", "source_class", "canonical_label", "reason", "sha256"],
    )

    counts = accepted.groupby("model_label").size().reindex(labels, fill_value=0)
    missing = [label for label, count in counts.items() if count < minimum_images]
    for label, count in counts.items():
        if 0 < count < warning_images:
            logging.warning("Class %s has %s accepted images, below warning threshold %s", label, count, warning_images)
    if missing:
        raise RuntimeError(
            f"Classes below minimum_images_per_class={minimum_images}: "
            + ", ".join(f"{label}={int(counts[label])}" for label in missing)
        )

    seed = int(config["project"]["seed"])
    split_rows: list[dict] = []
    for label in labels:
        class_rows = accepted[accepted["model_label"] == label]
        group_ids = sorted(class_rows["sha256"].unique())
        group_split = assign_groups(group_ids, ratios, seed)
        for row in class_rows.to_dict("records"):
            split_rows.append({**row, "split": group_split[row["sha256"]]})

    manifest = pd.DataFrame(split_rows)
    for split in ("train", "validation", "test"):
        manifest[manifest["split"] == split].to_csv(processed_dir / f"{split}.csv", index=False)
    rejected.to_csv(processed_dir / "rejected.csv", index=False)

    split_summary = (
        manifest.pivot_table(index="model_label", columns="split", values="filepath", aggfunc="count", fill_value=0)
        .reindex(labels, fill_value=0)
        .reset_index()
    )
    for split in ("train", "validation", "test"):
        if split not in split_summary.columns:
            split_summary[split] = 0
    split_summary["total"] = split_summary[["train", "validation", "test"]].sum(axis=1)
    split_summary.to_csv(report_dir / "split_summary.csv", index=False)

    output_paths = [
        processed_dir / "train.csv",
        processed_dir / "validation.csv",
        processed_dir / "test.csv",
        processed_dir / "rejected.csv",
    ]
    manifest_sha = manifest_hash(output_paths)
    dataset_manifest = {
        "dataset_name": "30VNFoods",
        "dataset_source": DATASET_SLUG,
        "selected_classes": labels,
        "class_order": labels,
        "seed": seed,
        "train_ratio": ratios[0],
        "validation_ratio": ratios[1],
        "test_ratio": ratios[2],
        "total_accepted": int(len(manifest)),
        "total_rejected": int(len(rejected)),
        "source_archive_sha256": load_source_archive_sha256(repo_root),
        "manifest_sha256": manifest_sha,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(processed_dir / "dataset_manifest.json", dataset_manifest)

    print(split_summary.rename(columns={"model_label": "Class", "train": "Train", "validation": "Validation", "test": "Test", "total": "Total"}).to_string(index=False))
    logging.info("Prepared manifests in %s", processed_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logging.error("%s", exc)
        raise SystemExit(1)
