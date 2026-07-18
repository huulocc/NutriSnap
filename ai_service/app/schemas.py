from __future__ import annotations

from pydantic import BaseModel


class ClassInfo(BaseModel):
    class_index: int
    model_label: str
    display_name: str


class PredictionItem(ClassInfo):
    confidence: float


class Timing(BaseModel):
    preprocessing_ms: float
    inference_ms: float
    total_ms: float


class HealthResponse(BaseModel):
    status: str
    service: str


class ReadyResponse(BaseModel):
    ready: bool
    model_version: str
    model_loaded: bool
    gpu_count: int


class ModelResponse(BaseModel):
    model_version: str
    input_shape: list[int | None]
    output_shape: list[int | None]
    class_count: int
    classes: list[ClassInfo]
    confidence_threshold: float


class PredictionResponse(BaseModel):
    request_id: str
    model_version: str
    prediction: PredictionItem
    top_k: list[PredictionItem]
    low_confidence: bool
    confidence_threshold: float
    timing: Timing


class SimplePredictionResponse(BaseModel):
    request_id: str
    prediction: PredictionItem
