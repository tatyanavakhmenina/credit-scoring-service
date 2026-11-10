import json
from datetime import UTC, datetime
from uuid import UUID

import asyncpg

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id             uuid PRIMARY KEY,
    status              text NOT NULL,
    probability_default double precision,
    model_version       text,
    error               text,
    created_at          timestamptz NOT NULL,
    finished_at         timestamptz
);
"""

# Идемпотентность держится на PRIMARY KEY: повторная доставка не создаёт вторую строку,
# а одновременная вставка ждёт коммита первой. Схема без "проверить, затем вставить".
SAVE_RESULT = """
INSERT INTO tasks (
    task_id, status, probability_default, model_version, created_at, finished_at
)
VALUES ($1, 'DONE', $2, $3, $4, $5)
ON CONFLICT (task_id) DO UPDATE SET
    status = 'DONE',
    probability_default = EXCLUDED.probability_default,
    model_version = EXCLUDED.model_version,
    error = NULL,
    finished_at = EXCLUDED.finished_at
WHERE tasks.status <> 'DONE'
"""


def _dsn(url: str) -> str:
    return url.replace("postgresql+asyncpg://", "postgresql://")


async def create_pool(url: str, max_size: int = 10) -> asyncpg.Pool:
    pool = await asyncpg.create_pool(_dsn(url), min_size=1, max_size=max_size)
    await pool.execute(SCHEMA)
    return pool


async def postgres_version(url: str) -> str:
    conn = await asyncpg.connect(_dsn(url), timeout=2.0)
    try:
        return await conn.fetchval("SHOW server_version")
    finally:
        await conn.close()


async def insert_pending(conn, task_id: UUID, created_at: datetime) -> None:
    await conn.execute(
        "INSERT INTO tasks (task_id, status, created_at) VALUES ($1, 'PENDING', $2)",
        task_id,
        created_at,
    )


async def save_result(
    pool: asyncpg.Pool,
    task_id: UUID,
    probability: float,
    model_version: str,
    created_at: datetime,
) -> None:
    await pool.execute(
        SAVE_RESULT,
        task_id,
        probability,
        model_version,
        created_at,
        datetime.now(UTC),
    )


async def mark_failed(
    pool: asyncpg.Pool, task_id: UUID, error: str, created_at: datetime
) -> None:
    await pool.execute(
        """
        INSERT INTO tasks (task_id, status, error, created_at, finished_at)
        VALUES ($1, 'FAILED', $2, $3, $4)
        ON CONFLICT (task_id) DO UPDATE SET
            status = 'FAILED', error = EXCLUDED.error,
            finished_at = EXCLUDED.finished_at
        WHERE tasks.status <> 'DONE'
        """,
        task_id,
        error,
        created_at,
        datetime.now(UTC),
    )


async def get_task(pool: asyncpg.Pool, task_id: UUID) -> dict | None:
    row = await pool.fetchrow("SELECT * FROM tasks WHERE task_id = $1", task_id)
    return json.loads(json.dumps(dict(row), default=str)) if row else None
