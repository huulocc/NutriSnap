from __future__ import annotations

from typing import Any


def get_sparse_smoothing_loss(num_classes: int, label_smoothing: float):
    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package="NutriSnap")
    class SparseCategoricalCrossentropyWithLabelSmoothing(tf.keras.losses.Loss):
        def __init__(self, num_classes: int, label_smoothing: float, name: str = "sparse_ce_label_smoothing"):
            super().__init__(name=name)
            self.num_classes = int(num_classes)
            self.label_smoothing = float(label_smoothing)

        def call(self, y_true, y_pred):
            y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
            y_true = tf.one_hot(y_true, depth=self.num_classes)
            return tf.keras.losses.categorical_crossentropy(
                y_true,
                y_pred,
                label_smoothing=self.label_smoothing,
            )

        def get_config(self):
            return {
                "num_classes": self.num_classes,
                "label_smoothing": self.label_smoothing,
                "name": self.name,
            }

    return SparseCategoricalCrossentropyWithLabelSmoothing(num_classes, label_smoothing)


def build_mobilenetv2(config: dict[str, Any], *, train_backbone: bool = False):
    import tensorflow as tf

    model_cfg = config["model"]
    input_shape = (
        int(model_cfg["input_height"]),
        int(model_cfg["input_width"]),
        int(model_cfg["input_channels"]),
    )
    inputs = tf.keras.Input(
        shape=input_shape,
        name="image",
    )
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet" if model_cfg.get("imagenet_weights", True) else None,
        name="mobilenetv2_backbone",
    )
    backbone.trainable = train_backbone
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(float(model_cfg["dropout"]), name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        int(model_cfg["num_classes"]),
        activation="softmax",
        dtype="float32",
        name="predictions",
    )(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="nutrisnap_mobilenetv2")


def compile_model(config: dict[str, Any], model, learning_rate: float):
    import tensorflow as tf

    smoothing = float(config["training"].get("label_smoothing", 0.0))
    num_classes = int(config["model"]["num_classes"])

    if smoothing > 0:
        loss = get_sparse_smoothing_loss(num_classes, smoothing)
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=float(config["training"].get("gradient_clipnorm", 0.0)) or None,
        ),
        loss=loss,
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    )
    return model


def get_backbone(model):
    for layer in model.layers:
        if "mobilenetv2_backbone" in layer.name:
            return layer
    raise ValueError("MobileNetV2 backbone layer not found.")


def freeze_backbone(model) -> None:
    backbone = get_backbone(model)
    backbone.trainable = False


def set_fine_tuning(model, fine_tune_last_layers: int, freeze_batch_normalization: bool = True) -> None:
    import tensorflow as tf

    backbone = get_backbone(model)
    backbone.trainable = True
    freeze_until = max(0, len(backbone.layers) - int(fine_tune_last_layers))
    for layer in backbone.layers[:freeze_until]:
        layer.trainable = False
    for layer in backbone.layers[freeze_until:]:
        layer.trainable = True
    if freeze_batch_normalization:
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False


def parameter_report(model) -> dict[str, int]:
    trainable = int(sum(tf_count_params(weight) for weight in model.trainable_weights))
    non_trainable = int(sum(tf_count_params(weight) for weight in model.non_trainable_weights))
    layers = list(iter_layers(model))
    return {
        "total_parameters": trainable + non_trainable,
        "trainable_parameters": trainable,
        "non_trainable_parameters": non_trainable,
        "trainable_layers": sum(1 for layer in layers if layer.trainable),
        "frozen_layers": sum(1 for layer in layers if not layer.trainable),
    }


def iter_layers(model):
    for layer in model.layers:
        yield layer
        if hasattr(layer, "layers"):
            yield from iter_layers(layer)


def tf_count_params(weight) -> int:
    import tensorflow as tf

    return int(tf.keras.backend.count_params(weight))
