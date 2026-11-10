from uuid import uuid4

import pytest
from faststream.rabbit import TestRabbitBroker

from credit_scoring.config import settings
from credit_scoring.pipeline import worker
from tests.conftest import FakeModel


@pytest.fixture
def stored(monkeypatch):
    saved, failed = [], []

    async def save_result(_pool, task_id, probability, version, _created):
        saved.append((task_id, probability, version))

    async def mark_failed(_pool, task_id, error, _created):
        failed.append((task_id, error))

    monkeypatch.setattr(worker, "save_result", save_result)
    monkeypatch.setattr(worker, "mark_failed", mark_failed)
    monkeypatch.setattr(worker.settings, "inference_delay_seconds", 0)
    worker.state.model = FakeModel()
    yield saved, failed
    worker.state.model = None


def _message(features: dict) -> dict:
    return {
        "task_id": str(uuid4()),
        "features": features,
        "created_at": "2026-11-06T10:00:00+00:00",
    }


async def test_handler_saves_result_with_model_version(stored, features):
    saved, _ = stored
    async with TestRabbitBroker(worker.broker) as broker:
        await broker.publish(_message(features), queue=settings.queue_name)

    assert len(saved) == 1
    assert saved[0][1:] == (0.9, "7")


async def test_failure_before_limit_does_not_mark_failed(stored, features, monkeypatch):
    _, failed = stored
    monkeypatch.setattr(worker, "score", _raise)
    async with TestRabbitBroker(worker.broker) as broker:
        with pytest.raises(RuntimeError):
            await broker.publish(
                _message(features),
                queue=settings.queue_name,
                headers={"x-delivery-count": 1},
            )

    assert failed == []


async def test_last_delivery_marks_task_failed(stored, features, monkeypatch):
    saved, failed = stored
    monkeypatch.setattr(worker, "score", _raise)
    async with TestRabbitBroker(worker.broker) as broker:
        with pytest.raises(RuntimeError):
            await broker.publish(
                _message(features),
                queue=settings.queue_name,
                headers={"x-delivery-count": settings.delivery_limit},
            )

    assert saved == []
    assert len(failed) == 1
    assert failed[0][1] == "model failed"


def _raise(_features: dict) -> float:
    raise RuntimeError("model failed")
