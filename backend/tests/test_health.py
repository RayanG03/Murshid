"""Smoke tests for the Stage 1 skeleton."""

from fastapi.testclient import TestClient

from src.main import app


def test_health_returns_ok() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_cors_allows_angular_dev_origin() -> None:
    client = TestClient(app)
    response = client.get(
        "/health", headers={"Origin": "http://localhost:4200"}
    )
    assert response.headers["access-control-allow-origin"] == "http://localhost:4200"
