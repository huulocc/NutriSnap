from __future__ import annotations

from typing import Any


def build_augmentation(config: dict[str, Any]):
    import tensorflow as tf

    aug = config.get("augmentation", {})
    layers = []
    if aug.get("random_flip"):
        layers.append(tf.keras.layers.RandomFlip(aug["random_flip"]))
    if aug.get("random_rotation", 0):
        layers.append(tf.keras.layers.RandomRotation(float(aug["random_rotation"])))
    if aug.get("random_zoom", 0):
        layers.append(tf.keras.layers.RandomZoom(float(aug["random_zoom"])))
    if aug.get("random_translation_height", 0) or aug.get("random_translation_width", 0):
        layers.append(
            tf.keras.layers.RandomTranslation(
                height_factor=float(aug.get("random_translation_height", 0)),
                width_factor=float(aug.get("random_translation_width", 0)),
            )
        )
    if aug.get("random_contrast", 0):
        layers.append(tf.keras.layers.RandomContrast(float(aug["random_contrast"])))
    return tf.keras.Sequential(layers, name="train_augmentation")
