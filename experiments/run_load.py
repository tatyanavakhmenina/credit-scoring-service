"""Сравнение конвейера с ограничениями и без них. Результаты: experiments/results."""

import asyncio
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.common import (
    compose,
    connect_db,
    container_memory_mb,
    delete_queue,
    queue_info,
)

RESULTS = Path("experiments/results")
SAMPLE_SECONDS = 3
RATE_WINDOW_SAMPLES = 4

BOUNDED = {"QUEUE_MAX_LENGTH": "200", "PREFETCH_COUNT": "4"}
SCENARIOS = {
    "no_limits": {
        "title": "Без ограничений (очередь и prefetch без лимита)",
        "env": {"QUEUE_MAX_LENGTH": "0", "PREFETCH_COUNT": "0"},
        "queue": "exp.no-limits",
        "duration": 90,
    },
    "unbounded_queue": {
        "title": "Очередь без лимита, prefetch=4",
        "env": {"QUEUE_MAX_LENGTH": "0", "PREFETCH_COUNT": "4"},
        "queue": "exp.unbounded",
        "duration": 90,
    },
    "backpressure_1w": {
        "title": "Backpressure, 1 воркер",
        "env": BOUNDED,
        "queue": "exp.bp-one",
        "duration": 90,
    },
    "backpressure_2w": {
        "title": "Backpressure, второй воркер на 60-й секунде",
        "env": BOUNDED,
        "queue": "exp.bp-two",
        "duration": 150,
        "scale_at": 60,
    },
}


def stop_pipeline() -> None:
    compose("rm", "-sf", "source", "worker")


async def run_scenario(name: str, spec: dict) -> dict:
    queue = spec["queue"]
    stop_pipeline()
    delete_queue(queue)
    delete_queue(f"{queue}.dlq")
    env = {
        **spec["env"],
        "QUEUE_NAME": queue,
        "SOURCE_BATCH_ID": f"{name}-{int(time.time())}",
        "SOURCE_MAX_TASKS": "30000",
    }
    db = await connect_db()
    await db.execute("TRUNCATE tasks")

    compose("up", "-d", "--no-deps", "--wait", "worker", env=env)
    compose("up", "-d", "--no-deps", "source", env=env)

    rows, scaled = [], False
    start = time.monotonic()
    while (elapsed := time.monotonic() - start) < spec["duration"]:
        if not scaled and elapsed >= spec.get("scale_at", 1e9):
            compose(
                "up",
                "-d",
                "--no-deps",
                "--no-recreate",
                "--scale",
                "worker=2",
                "worker",
                env=env,
            )
            scaled = True
        info = queue_info(queue)
        done = await db.fetchval("SELECT count(*) FROM tasks WHERE status = 'DONE'")
        latency = await db.fetchval(
            "SELECT avg(extract(epoch FROM finished_at - created_at)) FROM tasks "
            "WHERE finished_at > now() - interval '5 seconds'"
        )
        memory = container_memory_mb()
        rows.append(
            {
                "t": round(elapsed, 1),
                "depth": info.get("messages", 0),
                "ready": info.get("messages_ready", 0),
                "unacked": info.get("messages_unacknowledged", 0),
                "done": done,
                "accepted": done + info.get("messages", 0),
                "task_seconds": round(float(latency or 0), 3),
                "rabbit_mb": round(memory.get("rabbitmq", 0), 1),
                "worker_mb": round(memory.get("worker", 0), 1),
                "source_mb": round(memory.get("source", 0), 1),
            }
        )
        await asyncio.sleep(SAMPLE_SECONDS)
    await db.close()
    stop_pipeline()

    add_rates(rows)
    with (RESULTS / f"{name}.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summarize(spec, rows)


def add_rates(rows: list[dict]) -> None:
    for i, row in enumerate(rows):
        base = rows[max(0, i - RATE_WINDOW_SAMPLES)]
        dt = max(row["t"] - base["t"], 1e-6)
        row["source_rate"] = round((row["accepted"] - base["accepted"]) / dt, 1)
        row["worker_rate"] = round((row["done"] - base["done"]) / dt, 1)


def _mean(rows: list[dict], key: str) -> float:
    return round(sum(r[key] for r in rows) / max(len(rows), 1), 1)


def _steady(rows: list[dict], start: float, end: float) -> dict:
    window = [r for r in rows if start <= r["t"] <= end]
    return {
        "source_rate": _mean(window, "source_rate"),
        "worker_rate": _mean(window, "worker_rate"),
        "task_seconds": _mean(window, "task_seconds"),
        "mean_depth": _mean(window, "depth"),
    }


def summarize(spec: dict, rows: list[dict]) -> dict:
    summary = {
        "title": spec["title"],
        "max_depth": max(r["depth"] for r in rows),
        "max_rabbit_mb": max(r["rabbit_mb"] for r in rows),
        "max_worker_mb": max(r["worker_mb"] for r in rows),
        "max_source_mb": max(r["source_mb"] for r in rows),
        "done": rows[-1]["done"],
    }
    scale_at = spec.get("scale_at")
    if scale_at:
        summary["before_scale"] = _steady(rows, scale_at - 30, scale_at)
        summary["after_scale"] = _steady(rows, spec["duration"] - 40, spec["duration"])
    else:
        summary["steady"] = _steady(rows, 30, spec["duration"])
    return summary


def plot_all(names: list[str]) -> None:
    metrics = [
        ("depth", "Глубина очереди"),
        ("source_rate", "Скорость источника, задач/с"),
        ("worker_rate", "Скорость воркеров, задач/с"),
        ("task_seconds", "Время задачи, с"),
        ("rabbit_mb", "Память RabbitMQ, МБ"),
        ("worker_mb", "Память воркеров, МБ"),
    ]
    figure, axes = plt.subplots(3, 2, figsize=(12, 10))
    for ax, (key, title) in zip(axes.flat, metrics, strict=True):
        for name in names:
            with (RESULTS / f"{name}.csv").open() as file:
                data = list(csv.DictReader(file))
            ax.plot(
                [float(r["t"]) for r in data],
                [float(r[key]) for r in data],
                label=name,
            )
        ax.set_title(title)
        ax.set_xlabel("секунды")
        if key == "depth":
            ax.set_yscale("symlog", linthresh=300)
        ax.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(RESULTS / "comparison.png", dpi=110)


async def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name, spec in SCENARIOS.items():
        summaries[name] = await run_scenario(name, spec)
        print(name, json.dumps(summaries[name], ensure_ascii=False))
    (RESULTS / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_all(list(SCENARIOS))
    compose("up", "-d", "--no-deps", "--wait", "worker")


if __name__ == "__main__":
    asyncio.run(main())
