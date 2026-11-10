from credit_scoring.api.v1 import tasks
from credit_scoring.broker import QueueOverloadedError
from credit_scoring.config import Settings, settings


async def test_create_task_keeps_pending_after_confirm(
    client, state, features, monkeypatch
):
    published = []

    async def publish(_broker, message, _settings):
        published.append(message)

    monkeypatch.setattr(tasks, "publish_task", publish)

    response = await client.post("/api/v1/tasks", json=features)

    assert response.status_code == 202
    assert response.json()["status"] == "PENDING"
    assert len(published) == 1
    assert len(state.pool.rows) == 1


async def test_overload_returns_503_and_rolls_back_pending(
    client, state, features, monkeypatch
):
    async def publish(_broker, _message, _settings):
        raise QueueOverloadedError("nack")

    monkeypatch.setattr(tasks, "publish_task", publish)

    response = await client.post("/api/v1/tasks", json=features)

    assert response.status_code == 503
    assert response.headers["Retry-After"] == str(settings.retry_after_seconds)
    assert state.pool.rows == []


async def test_missing_dependencies_return_503(client, state, features):
    state.broker = None

    response = await client.post("/api/v1/tasks", json=features)

    assert response.status_code == 503
    assert "Retry-After" in response.headers


def test_retry_after_is_time_to_drain_full_queue():
    cfg = Settings(
        queue_max_length=200, worker_concurrency=4, inference_delay_seconds=0.2
    )
    assert cfg.retry_after_seconds == 10


def test_prefetch_defaults_to_concurrency():
    assert Settings(worker_concurrency=8, prefetch_count=None).effective_prefetch == 8
    assert Settings(prefetch_count=0).effective_prefetch == 0
