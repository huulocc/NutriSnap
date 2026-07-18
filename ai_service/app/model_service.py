from __future__ import annotations

import asyncio
import json
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ai_service.app.config import Settings


LOGGER = logging.getLogger("nutrisnap.ai")


class ModelService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.tf = None
        self.model = None
        self.labels: list[str] = []
        self.class_names: dict[int, dict[str, str]] = {}
        self.input_shape: list[int | None] = []
        self.output_shape: list[int | None] = []
        self.visible_gpus: list[Any] = []
        self.memory_growth: dict[str, bool | None] = {}
        self.model_load_count = 0
        self.semaphore = asyncio.Semaphore(settings.inference_concurrency)

    @property
    def ready(self) -> bool:
        return self.model is not None and bool(self.labels)

    def load(self) -> None:
        if self.model is not None:
            return

        import tensorflow as tf

        self.tf = tf
        self.visible_gpus = tf.config.list_physical_devices("GPU")
        for gpu in self.visible_gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                self.memory_growth[gpu.name] = bool(tf.config.experimental.get_memory_growth(gpu))
            except Exception:
                LOGGER.exception("Could not set memory growth for %s", gpu.name)
                self.memory_growth[gpu.name] = None
        if self.visible_gpus:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        else:
            tf.keras.mixed_precision.set_global_policy("float32")

        self.labels = self._load_labels(self.settings.labels_path)
        self.class_names = self._load_class_names(self.settings.class_names_path)
        self.model = tf.keras.models.load_model(self.settings.model_path, compile=False, safe_mode=True)
        self.model_load_count += 1
        self.input_shape = self._shape_list(self.model.input_shape)
        self.output_shape = self._shape_list(self.model.output_shape)
        if self.output_shape[-1] != len(self.labels):
            raise RuntimeError(f"Model output classes {self.output_shape[-1]} do not match labels {len(self.labels)}")
        dummy = tf.zeros([1, 224, 224, 3], dtype=tf.float32)
        probs = self.model(dummy, training=False).numpy()
        if probs.shape[-1] != len(self.labels) or np.isnan(probs).any() or np.isinf(probs).any():
            raise RuntimeError("Dummy inference failed model readiness checks")

        LOGGER.info("TensorFlow version: %s", tf.__version__)
        LOGGER.info("Physical GPU exposed to process: %s GPU(s)", len(self.visible_gpus))
        LOGGER.info("TensorFlow logical GPU count: %s", len(tf.config.list_logical_devices("GPU")))
        LOGGER.info("GPU devices: %s", [gpu.name for gpu in self.visible_gpus])
        LOGGER.info("Memory growth status: %s", self.memory_growth)
        LOGGER.info("Mixed precision policy: %s", tf.keras.mixed_precision.global_policy().name)
        LOGGER.info("Model input shape: %s", self.input_shape)
        LOGGER.info("Model output shape: %s", self.output_shape)
        LOGGER.info("Class count: %s", len(self.labels))
        LOGGER.info("Uvicorn workers: %s", self.settings.workers)
        LOGGER.info("Model load count: %s", self.model_load_count)

    async def predict(self, image_bytes: bytes, top_k: int) -> tuple[list[dict[str, Any]], dict[str, float]]:
        async with self.semaphore:
            return await asyncio.to_thread(self._predict_sync, image_bytes, top_k)

    def _predict_sync(self, image_bytes: bytes, top_k: int) -> tuple[list[dict[str, Any]], dict[str, float]]:
        if self.model is None or self.tf is None:
            raise RuntimeError("Model is not loaded")
        start = time.perf_counter()
        tensor = self._preprocess(image_bytes)
        after_preprocess = time.perf_counter()
        probabilities = self.model(tensor, training=False).numpy()[0]
        after_inference = time.perf_counter()
        if np.isnan(probabilities).any() or np.isinf(probabilities).any():
            raise RuntimeError("Model returned NaN or Inf probabilities")
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        predictions = [self._prediction_item(int(index), float(probabilities[index])) for index in top_indices]
        timing = {
            "preprocessing_ms": (after_preprocess - start) * 1000,
            "inference_ms": (after_inference - after_preprocess) * 1000,
            "total_ms": (after_inference - start) * 1000,
        }
        return predictions, timing

    def _preprocess(self, image_bytes: bytes):
        tf = self.tf
        if tf is None:
            raise RuntimeError("TensorFlow is not initialized")
        raw = tf.convert_to_tensor(image_bytes)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_with_pad(image, 224, 224)
        image = tf.ensure_shape(image, [224, 224, 3])
        image = image * 255.0
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return tf.expand_dims(image, 0)

    def validate_image(self, image_bytes: bytes, content_type: str | None) -> None:
        if content_type not in self.settings.allowed_types:
            raise ValueError(f"Unsupported image content type: {content_type}")
        if len(image_bytes) > self.settings.max_image_size_bytes:
            raise ValueError("Image exceeds maximum size")
        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.size
            if width > self.settings.max_image_width or height > self.settings.max_image_height:
                raise ValueError("Image dimensions exceed maximum allowed size")
            image.verify()

    def classes(self) -> list[dict[str, Any]]:
        return [
            {
                "class_index": index,
                "model_label": label,
                "display_name": self.class_names[index]["display_name"],
            }
            for index, label in enumerate(self.labels)
        ]

    def _prediction_item(self, index: int, confidence: float) -> dict[str, Any]:
        return {
            "class_index": index,
            "model_label": self.labels[index],
            "display_name": self.class_names[index]["display_name"],
            "confidence": confidence,
        }

    @staticmethod
    def _load_labels(path: Path) -> list[str]:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    @staticmethod
    def _load_class_names(path: Path) -> dict[int, dict[str, str]]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {int(key): value for key, value in raw.items()}

    @staticmethod
    def _shape_list(shape) -> list[int | None]:
        return [int(dim) if dim is not None else None for dim in tuple(shape)]
