from credit_scoring import __version__
from credit_scoring.api import health


async def test_healthz(client):
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_version_matches_package_metadata(client):
    response = await client.get("/api/v1/version")
    assert response.status_code == 200
    assert response.json()["version"] == __version__


async def test_health_reports_versions_and_latency(client, state, monkeypatch):
    async def version(_url: str) -> str:
        return "1.2.3"

    for name in ("postgres_version", "rabbitmq_version", "mlflow_version"):
        monkeypatch.setattr(health, name, version)

    response = await client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model"]["version"] == "7"
    for component in ("postgres", "rabbitmq", "mlflow"):
        assert data["components"][component]["version"] == "1.2.3"
        assert data["components"][component]["latency_ms"] >= 0


async def test_health_is_degraded_when_component_is_down(client, monkeypatch):
    async def version(_url: str) -> str:
        return "1.2.3"

    async def broken(_url: str) -> str:
        raise ConnectionError("db is down")

    monkeypatch.setattr(health, "postgres_version", broken)
    monkeypatch.setattr(health, "rabbitmq_version", version)
    monkeypatch.setattr(health, "mlflow_version", version)

    response = await client.get("/api/v1/health")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "degraded"
    assert data["components"]["postgres"]["status"] == "unhealthy"
    assert "db is down" not in response.text
    assert data["components"]["rabbitmq"]["status"] == "healthy"
