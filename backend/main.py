"""Serving layer for FaceKYC with explicit liveness/readiness semantics."""

from __future__ import annotations

import io
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from facekyc.artifacts import ArtifactError, load_bundle
from facekyc.config import Settings, load_settings
from facekyc.pipeline import FaceKYCPipeline

LOGGER = logging.getLogger("facekyc.api")


def _candidate_mode_enabled() -> bool:
    return os.getenv("FACEKYC_ALLOW_CANDIDATE", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _metadata(settings: Settings) -> dict[str, Any]:
    configured_bundle = os.getenv("FACEKYC_BUNDLE_PATH", settings.artifact.bundle_path)
    bundle_path = settings.resolve(configured_bundle)
    try:
        bundle = load_bundle(bundle_path, verify_weights=False)
    except ArtifactError as exc:
        return {
            "ready": False,
            "project_version": settings.project.version,
            "artifact_status": "unavailable",
            "detail": str(exc),
        }
    return {
        "ready": bundle["deployment_status"] == "approved",
        "project_version": settings.project.version,
        "model_version": bundle["model_version"],
        "created_at": bundle["created_at"],
        "deployment_status": bundle["deployment_status"],
        "verification": bundle["verification"],
        "liveness": {
            key: value
            for key, value in bundle["liveness"].items()
            if key not in {"weights_path", "weights_sha256"}
        },
        "metrics": bundle["metrics"],
        "data_protocols": bundle["data_protocols"],
        "governance": bundle["governance"],
    }


async def _read_image(upload: UploadFile, settings: Settings, field_name: str) -> Image.Image:
    content_type = (upload.content_type or "").lower()
    if content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail=f"{field_name}: unsupported media type")
    payload = await upload.read(settings.input.max_file_bytes + 1)
    if len(payload) > settings.input.max_file_bytes:
        raise HTTPException(status_code=413, detail=f"{field_name}: file exceeds size limit")
    if not payload:
        raise HTTPException(status_code=422, detail=f"{field_name}: empty file")
    try:
        image = Image.open(io.BytesIO(payload))
        width, height = image.size
        if width > settings.input.max_width or height > settings.input.max_height:
            raise HTTPException(
                status_code=413, detail=f"{field_name}: image dimensions exceed limit"
            )
        image.load()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=422, detail=f"{field_name}: invalid image") from exc
    if (image.format or "").upper() not in settings.input.allowed_formats:
        raise HTTPException(status_code=415, detail=f"{field_name}: unsupported image format")
    return image.convert("RGB")


def create_app(
    *,
    pipeline: Any | None = None,
    settings: Settings | None = None,
    initialize_models: bool = True,
) -> FastAPI:
    resolved_settings = settings or load_settings(os.getenv("FACEKYC_CONFIG_PATH"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline
        app.state.startup_error = None
        if pipeline is None and initialize_models:
            try:
                configured_bundle = os.getenv("FACEKYC_BUNDLE_PATH")
                app.state.pipeline = await run_in_threadpool(
                    FaceKYCPipeline.from_artifacts,
                    resolved_settings.config_path,
                    configured_bundle,
                    require_approved=not _candidate_mode_enabled(),
                )
            except Exception as exc:  # readiness captures optional dependency/artifact failures
                app.state.startup_error = str(exc)
                LOGGER.warning("FaceKYC is live but not ready: %s", exc)
        yield
        app.state.pipeline = None

    app = FastAPI(
        title="FaceKYC biometric decision-support API",
        version=resolved_settings.project.version,
        description=(
            "Research-grade 1:1 face verification plus passive PAD. "
            "The response is a review signal, not legal proof of identity."
        ),
        lifespan=lifespan,
    )
    origins = [
        value.strip()
        for value in os.getenv(
            "FACEKYC_CORS_ORIGINS", "http://localhost:8504,http://localhost:8501"
        ).split(",")
        if value.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        LOGGER.info(
            "request_id=%s method=%s path=%s status=%s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
        )
        return response

    @app.get("/live")
    def live() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/ready")
    def ready(request: Request):
        if getattr(request.app.state, "pipeline", None) is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "detail": getattr(request.app.state, "startup_error", None)
                    or "Model pipeline is not initialized",
                },
            )
        return {"status": "ready"}

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        return {
            "live": True,
            "ready": getattr(request.app.state, "pipeline", None) is not None,
            "model": _metadata(resolved_settings),
        }

    @app.get("/model")
    def model() -> dict[str, Any]:
        return _metadata(resolved_settings)

    @app.post("/api/v1/verify")
    async def verify(
        request: Request,
        id_image: Annotated[UploadFile, File(...)],
        selfie_image: Annotated[UploadFile, File(...)],
    ) -> dict[str, Any]:
        runtime_pipeline = getattr(request.app.state, "pipeline", None)
        if runtime_pipeline is None:
            raise HTTPException(status_code=503, detail="Model pipeline is not ready")
        id_pil = await _read_image(id_image, resolved_settings, "id_image")
        selfie_pil = await _read_image(selfie_image, resolved_settings, "selfie_image")
        try:
            result = await run_in_threadpool(runtime_pipeline.verify, id_pil, selfie_pil)
        except Exception as exc:
            LOGGER.exception("Pipeline failure")
            raise HTTPException(status_code=500, detail="Biometric pipeline failed") from exc
        result["request_id"] = request.state.request_id
        return result

    return app


app = create_app()
