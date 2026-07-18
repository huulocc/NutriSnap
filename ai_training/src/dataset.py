from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_MANIFEST_COLUMNS = {
    "filepath",
    "class_index",
    "model_label",
    "display_name",
    "split",
    "sha256",
}


def preprocess_image_tensor(image, config: dict[str, Any], training: bool = False, augmenter=None):
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    model_cfg = config["model"]
    target_height = int(model_cfg["input_height"])
    target_width = int(model_cfg["input_width"])

    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_pad(image, target_height, target_width)
    image = tf.ensure_shape(image, [target_height, target_width, 3])
    image = image * 255.0
    if training and augmenter is not None:
        image = augmenter(image, training=True)
    return preprocess_input(image)


def load_image_for_inference(image_path: str | Path, config: dict[str, Any]):
    import tensorflow as tf

    raw = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    return preprocess_image_tensor(image, config, training=False)


def _load_and_preprocess(path, label, config: dict[str, Any], training: bool, augmenter):
    import tensorflow as tf

    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image = preprocess_image_tensor(image, config, training=training, augmenter=augmenter)
    label = tf.cast(label, tf.int32)
    return image, label


def read_manifest(path: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    missing = REQUIRED_MANIFEST_COLUMNS.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest {path} is missing columns: {sorted(missing)}")
    return manifest


def build_dataset(
    manifest_path: str | Path,
    config: dict[str, Any],
    *,
    training: bool,
    batch_size: int | None = None,
    cache: bool | None = None,
):
    import tensorflow as tf

    from ai_training.src.augmentation import build_augmentation

    manifest = read_manifest(manifest_path)
    paths = manifest["filepath"].astype(str).tolist()
    labels = manifest["class_index"].astype("int32").tolist()
    batch = int(batch_size or config["training"]["batch_size"])
    use_cache = bool(config["training"].get("cache_dataset", False) if cache is None else cache)
    augmenter = build_augmentation(config) if training else None

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(buffer_size=len(paths), seed=int(config["project"]["seed"]), reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda path, label: _load_and_preprocess(path, label, config, training, augmenter),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if use_cache:
        dataset = dataset.cache()
    return dataset.batch(batch).prefetch(tf.data.AUTOTUNE)
