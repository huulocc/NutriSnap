from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ai_training.src.config import find_repo_root, resolve_project_path
from ai_training.src.utils import ensure_dirs, setup_logging, sha256_file, write_json


DATASET_SLUG = "quandang/vietnamese-foods"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract the 30VNFoods Kaggle dataset.")
    parser.add_argument("--output", default="ai_training/data/source")
    parser.add_argument("--downloads-dir", default="ai_training/data/downloads")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def has_kaggle_credentials() -> bool:
    env_ok = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    file_ok = (Path.home() / ".kaggle" / "kaggle.json").exists()
    return env_ok or file_ok


def kaggle_command() -> list[str] | None:
    executable = shutil.which("kaggle")
    if executable:
        return [executable]
    try:
        subprocess.run([sys.executable, "-c", "import kaggle"], capture_output=True, text=True, check=True)
        return [sys.executable, "-m", "kaggle"]
    except subprocess.CalledProcessError:
        return None


def safe_extract_zip(archive: Path, output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            target = (output_dir / member.filename).resolve()
            if not str(target).startswith(str(output_dir)):
                raise RuntimeError(f"Unsafe archive member path detected: {member.filename}")
        zip_file.extractall(output_dir)


def main() -> int:
    args = parse_args()
    setup_logging()
    repo_root = find_repo_root()
    output_dir = resolve_project_path(args.output, repo_root)
    downloads_dir = resolve_project_path(args.downloads_dir, repo_root)
    ensure_dirs(output_dir, downloads_dir)

    command_base = kaggle_command()
    if command_base is None:
        logging.error("Kaggle CLI/package not found. Install it with: pip install kaggle")
        return 1
    if not has_kaggle_credentials():
        logging.error(
            "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY, or create ~/.kaggle/kaggle.json. "
            "No dataset was downloaded."
        )
        return 1

    archive_path = downloads_dir / f"{DATASET_SLUG.split('/')[-1]}.zip"
    if archive_path.exists() and not args.force:
        logging.info("Archive already exists, skipping download: %s", archive_path)
    else:
        command = [
            *command_base,
            "datasets",
            "download",
            "-d",
            DATASET_SLUG,
            "-p",
            str(downloads_dir),
            "--force" if args.force else "",
        ]
        command = [part for part in command if part]
        logging.info("Downloading Kaggle dataset %s to %s", DATASET_SLUG, downloads_dir)
        subprocess.run(command, check=True)

    if not archive_path.exists():
        candidates = sorted(downloads_dir.glob("*.zip"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"No downloaded zip archive found in {downloads_dir}")
        archive_path = candidates[0]

    archive_sha256 = sha256_file(archive_path)
    safe_extract_zip(archive_path, output_dir)
    metadata = {
        "dataset_slug": DATASET_SLUG,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha256,
        "extracted_path": str(output_dir),
    }
    write_json(downloads_dir / "download_metadata.json", metadata)
    logging.info("Dataset extracted to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
