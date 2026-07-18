from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from ai_training.src.config import load_config, resolve_project_path, validate_label_sources
from ai_training.src.utils import ensure_dirs, manifest_hash, write_json


PROTOCOL = "deduplicated_stratified_80_10_10_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lock dataset split protocol metadata without changing split assignment.")
    parser.add_argument("--config", default="ai_training/configs/mobilenetv2_config.yaml")
    return parser.parse_args()


def infer_source_split(filepath: str) -> str:
    parts = {part.lower() for part in Path(filepath).parts}
    if "train" in parts:
        return "train"
    if "validate" in parts or "validation" in parts:
        return "validation"
    if "test" in parts:
        return "test"
    return "unknown"


def load_manifest(path: Path, custom_split: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"filepath", "class_index", "model_label", "display_name", "split", "sha256"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if set(df["split"]) != {custom_split}:
        raise ValueError(f"{path} contains unexpected split values: {sorted(df['split'].unique())}")
    df["source_split"] = df["filepath"].map(infer_source_split)
    return df


def main() -> int:
    args = parse_args()
    config, _, repo_root = load_config(args.config)
    labels = validate_label_sources(config, repo_root)
    processed_dir = resolve_project_path(config["dataset"]["processed_dir"], repo_root)
    report_dir = resolve_project_path(config["output"]["report_dir"], repo_root)
    ensure_dirs(report_dir)

    paths = {
        "train": processed_dir / "train.csv",
        "validation": processed_dir / "validation.csv",
        "test": processed_dir / "test.csv",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")

    previous_sha = manifest_hash(list(paths.values()) + [processed_dir / "rejected.csv"])
    frames = {split: load_manifest(path, split) for split, path in paths.items()}
    for split, frame in frames.items():
        frame.to_csv(paths[split], index=False)

    matrix = (
        pd.concat(frames.values(), ignore_index=True)
        .groupby(["source_split", "split"])
        .size()
        .reset_index(name="sample_count")
        .rename(columns={"split": "custom_split"})
    )
    matrix.to_csv(report_dir / "source_to_custom_split_matrix.csv", index=False)

    output_paths = list(paths.values()) + [processed_dir / "rejected.csv"]
    new_sha = manifest_hash(output_paths)
    manifest_path = processed_dir / "dataset_manifest.json"
    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    old_recorded_sha = dataset_manifest.get("manifest_sha256")
    if old_recorded_sha and old_recorded_sha != new_sha:
        dataset_manifest["previous_manifest_sha256"] = old_recorded_sha
    elif previous_sha != new_sha:
        dataset_manifest["previous_manifest_sha256"] = previous_sha
    dataset_manifest.update(
        {
            "split_protocol": PROTOCOL,
            "official_split_preserved": False,
            "benchmark_comparable_to_official_30vnfoods": False,
            "manifest_sha256": new_sha,
            "split_counts": {split: int(len(frame)) for split, frame in frames.items()},
            "class_order": labels,
        }
    )
    write_json(manifest_path, dataset_manifest)
    print(f"previous_manifest_sha256={dataset_manifest.get('previous_manifest_sha256')}")
    print(f"manifest_sha256={new_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
