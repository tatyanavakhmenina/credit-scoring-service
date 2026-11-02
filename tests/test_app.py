import pytest
from credit_scoring.main import app
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_healthz():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_version():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.get("/api/v1/version")
    assert response.status_code == 200
    assert "version" in response.json()


@pytest.mark.asyncio
async def test_health_e2e():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "components" in data
    assert "database" in data["components"]
