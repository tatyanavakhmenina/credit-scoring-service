"""Проверки надёжности под нагрузкой на реальном стенде: каждая печатает PASS/FAIL."""

import asyncio
import json
import subprocess
import time
from pathlib import Path

import aio_pika
import httpx

from experiments.common import compose, connect_db, delete_queue, queue_info
from tests.conftest import FEATURES

QUEUE = "exp.reliability"
ENV = {"QUEUE_NAME": QUEUE, "QUEUE_MAX_LENGTH": "200", "PREFETCH_COUNT": "4"}
RESULTS = Path("experiments/results/reliability.json")
report: dict[str, dict] = {}


def check(name: str, passed: bool, **details) -> None:
    report[name] = {"passed": passed, **details}
    print("PASS" if passed else "FAIL", name, details)


def worker_containers() -> list[str]:
    output = subprocess.run(
        ["docker", "compose", "ps", "-q", "worker"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return output.split()


async def wait_until_drained(db, expected: int, timeout: float = 240) -> int:
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        done = await db.fetchval("SELECT count(*) FROM tasks WHERE status = 'DONE'")
        if done >= expected and queue_info(QUEUE).get("messages", 1) == 0:
            return done
        await asyncio.sleep(1)
    return await db.fetchval("SELECT count(*) FROM tasks WHERE status = 'DONE'")


async def run_batch(db, tasks: int, batch: str, action) -> None:
    await db.execute("TRUNCATE tasks")
    env = {**ENV, "SOURCE_MAX_TASKS": str(tasks), "SOURCE_BATCH_ID": batch}
    compose("up", "-d", "--no-deps", "source", env=env)
    await asyncio.sleep(8)
    await action()
    done = await wait_until_drained(db, tasks)
    pending = await db.fetchval("SELECT count(*) FROM tasks WHERE status <> 'DONE'")
    check(
        batch,
        done == tasks and pending == 0,
        expected=tasks,
        rows_done=done,
        not_done=pending,
        dlq=queue_info(f"{QUEUE}.dlq").get("messages", 0),
    )


async def sigterm_under_load() -> None:
    first = worker_containers()[0]
    started = time.monotonic()
    subprocess.run(
        ["docker", "stop", "-t", "40", first], check=True, capture_output=True
    )
    report["sigterm_stop_seconds"] = {"value": round(time.monotonic() - started, 1)}
    logs = subprocess.run(
        ["docker", "logs", "--tail", "15", first],
        capture_output=True,
        text=True,
        check=False,
    )
    print((logs.stdout + logs.stderr)[-700:])


async def sigkill_under_load() -> None:
    subprocess.run(
        ["docker", "kill", worker_containers()[0]], check=True, capture_output=True
    )


async def health_during_load() -> None:
    samples = []
    worker = worker_containers()[0]
    for _ in range(10):
        output = subprocess.run(
            [
                "docker",
                "exec",
                worker,
                "python",
                "-c",
                "import time,urllib.request as u;s=time.perf_counter();"
                "u.urlopen('http://localhost:8001/healthz',timeout=2);"
                "print(round((time.perf_counter()-s)*1000,1))",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        samples.append(float(output))
    unacked = queue_info(QUEUE).get("messages_unacknowledged", 0)
    check(
        "worker_health_responds_during_inference",
        max(samples) < 500 and unacked > 0,
        max_ms=max(samples),
        unacked_at_end=unacked,
    )


async def poison_goes_to_dlq() -> None:
    before = queue_info(f"{QUEUE}.dlq").get("messages", 0)
    connection = await aio_pika.connect("amqp://guest:guest@localhost:5672/")
    async with connection:
        channel = await connection.channel()
        await channel.default_exchange.publish(
            aio_pika.Message(b'{"not": "a task"}', delivery_mode=2),
            routing_key=QUEUE,
        )
    await asyncio.sleep(5)
    after = queue_info(f"{QUEUE}.dlq").get("messages", 0)
    check(
        "poison_message_in_dlq", after == before + 1, dlq_before=before, dlq_after=after
    )


async def broker_queue_depth(name: str) -> int:
    """Прямой ответ брокера: статистика Management API у новой очереди отстаёт."""
    connection = await aio_pika.connect("amqp://guest:guest@localhost:5672/")
    async with connection:
        channel = await connection.channel()
        queue = await channel.declare_queue(name, passive=True)
        return queue.declaration_result.message_count


async def api_sheds_load(db) -> None:
    await db.execute("TRUNCATE tasks")
    compose("rm", "-sf", "worker", "source")
    compose("up", "-d", "--no-deps", "--wait", "app", env=ENV)
    statuses, retry_after, durations = [], set(), []
    async with httpx.AsyncClient(base_url="http://localhost:8000", timeout=5) as client:
        for _ in range(260):
            started = time.perf_counter()
            response = await client.post("/api/v1/tasks", json=FEATURES)
            durations.append((time.perf_counter() - started) * 1000)
            statuses.append(response.status_code)
            if response.status_code == 503:
                retry_after.add(response.headers.get("Retry-After"))
    accepted, rejected = statuses.count(202), statuses.count(503)
    rows = await db.fetchval("SELECT count(*) FROM tasks")
    in_queue = await broker_queue_depth(QUEUE)
    rejected_ms = [d for d, s in zip(durations, statuses, strict=True) if s == 503]
    check(
        "api_503_with_retry_after_and_no_pending_leak",
        rejected > 0 and rows == accepted == in_queue and retry_after == {"10"},
        accepted=accepted,
        rejected=rejected,
        pending_rows=rows,
        in_queue=in_queue,
        retry_after=sorted(retry_after),
        max_503_ms=round(max(rejected_ms), 1) if rejected_ms else None,
    )


async def main() -> None:
    db = await connect_db()
    compose("rm", "-sf", "source", "worker")
    delete_queue(QUEUE)
    delete_queue(f"{QUEUE}.dlq")
    compose("up", "-d", "--no-deps", "--wait", "--scale", "worker=2", "worker", env=ENV)

    await run_batch(
        db, 600, "graceful_sigterm_no_loss_no_duplicates", sigterm_under_load
    )
    compose("up", "-d", "--no-deps", "--wait", "--scale", "worker=2", "worker", env=ENV)
    await run_batch(
        db, 600, "sigkill_redelivers_without_duplicates", sigkill_under_load
    )
    compose("up", "-d", "--no-deps", "--wait", "--scale", "worker=2", "worker", env=ENV)
    await run_batch(db, 300, "health_batch", health_during_load)
    await poison_goes_to_dlq()
    await api_sheds_load(db)

    await db.close()
    RESULTS.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    compose("rm", "-sf", "source", "worker")
    delete_queue(QUEUE)
    delete_queue(f"{QUEUE}.dlq")
    compose("up", "-d", "--wait", "app", "worker")


if __name__ == "__main__":
    asyncio.run(main())
