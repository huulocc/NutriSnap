from __future__ import annotations

import os

os.environ.setdefault("MODEL_PATH", "ai_training/outputs/runs/20260717_162321_mobilenetv2_continuation/checkpoints/best_continued.keras")
os.environ.setdefault("LABELS_PATH", "ai_training/metadata/food_labels.txt")
os.environ.setdefault("CLASS_NAMES_PATH", "ai_training/metadata/class_names.json")
os.environ.setdefault("API_KEY", "test-key")

from ai_service.app.main import app, health


def test_health_is_public() -> None:
    response = health()
    assert response.status == "ok"


def test_required_routes_exist() -> None:
    routes = {route.path for route in app.routes}
    assert "/health" in routes
    assert "/ready" in routes
    assert "/predict" in routes
    assert "/api/v1/predict" in routes
    assert "/api/v1/model" in routes


def test_predict_method_is_post() -> None:
    methods_by_path = {route.path: getattr(route, "methods", set()) for route in app.routes}
    assert "POST" in methods_by_path["/predict"]
    assert "POST" in methods_by_path["/api/v1/predict"]
