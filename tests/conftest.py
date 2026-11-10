from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from credit_scoring.main import app
from credit_scoring.ml.model import ModelInfo

FEATURES = {
    "limit_bal": 200000,
    "education": 2,
    "marriage": 1,
    "age": 35,
    "pay_0": 2,
    "pay_2": 2,
    "pay_3": 0,
    "pay_4": 0,
    "pay_5": 0,
    "pay_6": 0,
    "bill_amt1": 50000,
    "bill_amt2": 48000,
    "bill_amt3": 46000,
    "bill_amt4": 42000,
    "bill_amt5": 40000,
    "bill_amt6": 38000,
    "pay_amt1": 0,
    "pay_amt2": 2000,
    "pay_amt3": 1500,
    "pay_amt4": 1500,
    "pay_amt5": 1500,
    "pay_amt6": 1500,
}


class FakeModel:
    info = ModelInfo("credit-default", "champion", "7", "run-1", threshold=0.5)

    def predict_proba(self, features: dict) -> float:
        return 0.9 if features["pay_0"] > 0 else 0.1

    def decide(self, probability: float) -> str:
        return "reject" if probability >= self.info.threshold else "approve"


class FakeConnection:
    def __init__(self, pool: "FakePool") -> None:
        self.pool = pool

    @asynccontextmanager
    async def transaction(self):
        snapshot = list(self.pool.rows)
        try:
            yield
        except Exception:
            self.pool.rows = snapshot
            raise

    async def execute(self, _query: str, *args) -> None:
        self.pool.rows.append(args)


class FakePool:
    def __init__(self) -> None:
        self.rows: list[tuple] = []

    def acquire(self, timeout: float | None = None):
        @asynccontextmanager
        async def _acquire():
            yield FakeConnection(self)

        return _acquire()


@pytest.fixture
def features() -> dict:
    return dict(FEATURES)


@pytest.fixture
def state():
    app.state.pool = FakePool()
    app.state.broker = SimpleNamespace()
    app.state.model = FakeModel()
    yield app.state
    app.state.pool = app.state.broker = app.state.model = None


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
