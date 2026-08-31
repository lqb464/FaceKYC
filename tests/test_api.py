from __future__ import annotations

from dataclasses import replace

from conftest import make_pipeline
from fastapi.testclient import TestClient

from backend.main import _candidate_mode_enabled, create_app


def test_candidate_mode_requires_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("FACEKYC_ALLOW_CANDIDATE", raising=False)
    assert _candidate_mode_enabled() is False
    monkeypatch.setenv("FACEKYC_ALLOW_CANDIDATE", "true")
    assert _candidate_mode_enabled() is True


def test_health_model_and_verify_endpoints(settings, image_bytes):
    app = create_app(pipeline=make_pipeline(settings), settings=settings, initialize_models=False)
    with TestClient(app) as client:
        assert client.get("/live").status_code == 200
        assert client.get("/ready").json() == {"status": "ready"}
        response = client.post(
            "/api/v1/verify",
            files={
                "id_image": ("id.jpg", image_bytes, "image/jpeg"),
                "selfie_image": ("selfie.jpg", image_bytes, "image/jpeg"),
            },
        )
        assert response.status_code == 200
        assert response.json()["decision"] == "verified"
        assert response.json()["request_id"] == response.headers["X-Request-ID"]
        assert response.headers["Cache-Control"] == "no-store"
        assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_api_rejects_non_image_payload(settings):
    app = create_app(pipeline=make_pipeline(settings), settings=settings, initialize_models=False)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/verify",
            files={
                "id_image": ("id.txt", b"not-image", "text/plain"),
                "selfie_image": ("selfie.txt", b"not-image", "text/plain"),
            },
        )
        assert response.status_code == 415


def test_api_rejects_missing_selfie(settings, image_bytes):
    app = create_app(pipeline=make_pipeline(settings), settings=settings, initialize_models=False)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/verify",
            files={"id_image": ("id.jpg", image_bytes, "image/jpeg")},
        )
        assert response.status_code == 422


def test_api_rejects_oversized_dimensions(settings, image_bytes):
    limited = replace(
        settings,
        input=replace(settings.input, max_width=100, max_height=100),
    )
    app = create_app(pipeline=make_pipeline(limited), settings=limited, initialize_models=False)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/verify",
            files={
                "id_image": ("id.jpg", image_bytes, "image/jpeg"),
                "selfie_image": ("selfie.jpg", image_bytes, "image/jpeg"),
            },
        )
        assert response.status_code == 413


def test_readiness_fails_closed_without_artifact(settings):
    app = create_app(settings=settings, initialize_models=False)
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 503
        response = client.post(
            "/api/v1/verify",
            files={
                "id_image": ("id.jpg", b"unused", "image/jpeg"),
                "selfie_image": ("selfie.jpg", b"unused", "image/jpeg"),
            },
        )
        assert response.status_code == 503
