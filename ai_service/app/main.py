from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status

from ai_service.app.config import get_settings
from ai_service.app.model_service import ModelService
from ai_service.app.schemas import HealthResponse, ModelResponse, PredictionResponse, ReadyResponse, SimplePredictionResponse
from ai_service.app.security import require_api_key


settings = get_settings()
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger("nutrisnap.ai")
model_service = ModelService(settings)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    model_service.load()
    yield


app = FastAPI(
    title=settings.app_name,
    docs_url="/docs" if settings.enable_docs else None,
    redoc_url="/redoc" if settings.enable_docs else None,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service=settings.app_name)


@app.get("/ready", response_model=ReadyResponse, dependencies=[Depends(require_api_key)])
def ready() -> ReadyResponse:
    if not model_service.ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not ready")
    return ReadyResponse(
        ready=True,
        model_version=settings.model_version,
        model_loaded=True,
        gpu_count=len(model_service.visible_gpus),
    )


@app.get(f"{settings.api_prefix}/model", response_model=ModelResponse, dependencies=[Depends(require_api_key)])
def model() -> ModelResponse:
    if not model_service.ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not ready")
    return ModelResponse(
        model_version=settings.model_version,
        input_shape=model_service.input_shape,
        output_shape=model_service.output_shape,
        class_count=len(model_service.labels),
        classes=model_service.classes(),
        confidence_threshold=settings.confidence_threshold,
    )


@app.post(f"{settings.api_prefix}/predict", response_model=PredictionResponse, dependencies=[Depends(require_api_key)])
async def predict(file: UploadFile = File(...), top_k: int = Form(default=None)) -> PredictionResponse:
    if not model_service.ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not ready")
    effective_top_k = top_k or settings.default_top_k
    if effective_top_k < 1 or effective_top_k > settings.max_top_k:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"top_k must be between 1 and {settings.max_top_k}")
    start = time.perf_counter()
    image_bytes = await file.read()
    try:
        model_service.validate_image(image_bytes, file.content_type)
        predictions, timing = await model_service.predict(image_bytes, effective_top_k)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction failed") from exc

    prediction = predictions[0]
    low_confidence = prediction["confidence"] < settings.confidence_threshold
    timing["total_ms"] = (time.perf_counter() - start) * 1000
    logger.info(
        "request_id=%s model_version=%s model_label=%s confidence=%.6f low_confidence=%s preprocessing_ms=%.3f inference_ms=%.3f total_ms=%.3f status_code=200",
        request_id := str(uuid.uuid4()),
        settings.model_version,
        prediction["model_label"],
        prediction["confidence"],
        low_confidence,
        timing["preprocessing_ms"],
        timing["inference_ms"],
        timing["total_ms"],
    )
    return PredictionResponse(
        request_id=request_id,
        model_version=settings.model_version,
        prediction=prediction,
        top_k=predictions,
        low_confidence=low_confidence,
        confidence_threshold=settings.confidence_threshold,
        timing=timing,
    )


@app.post("/predict", response_model=SimplePredictionResponse)
async def predict_top1(file: UploadFile = File(...)) -> SimplePredictionResponse:
    if not model_service.ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not ready")
    image_bytes = await file.read()
    try:
        model_service.validate_image(image_bytes, file.content_type)
        predictions, timing = await model_service.predict(image_bytes, 1)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction failed") from exc

    prediction = predictions[0]
    request_id = str(uuid.uuid4())
    logger.info(
        "request_id=%s model_version=%s model_label=%s confidence=%.6f low_confidence=%s preprocessing_ms=%.3f inference_ms=%.3f total_ms=%.3f status_code=200",
        request_id,
        settings.model_version,
        prediction["model_label"],
        prediction["confidence"],
        prediction["confidence"] < settings.confidence_threshold,
        timing["preprocessing_ms"],
        timing["inference_ms"],
        timing["total_ms"],
    )
    return SimplePredictionResponse(request_id=request_id, prediction=prediction)
