import json
import os
import subprocess
import urllib.parse

import asyncpg
import httpx

MANAGEMENT = "http://localhost:15672/api"
AUTH = ("guest", "guest")
DATABASE = "postgresql://postgres:postgres@localhost:5433/credit_scoring"


def compose(*args: str, env: dict[str, str] | None = None) -> None:
    subprocess.run(
        ["docker", "compose", "--profile", "load", *args],
        env={**os.environ, **(env or {})},
        check=True,
        capture_output=True,
    )


def queue_info(name: str) -> dict:
    encoded = urllib.parse.quote(name, safe="")
    response = httpx.get(f"{MANAGEMENT}/queues/%2F/{encoded}", auth=AUTH, timeout=5)
    return response.json() if response.status_code == 200 else {}


def delete_queue(name: str) -> None:
    encoded = urllib.parse.quote(name, safe="")
    httpx.delete(f"{MANAGEMENT}/queues/%2F/{encoded}", auth=AUTH, timeout=5)


def container_memory_mb() -> dict[str, float]:
    """Память контейнеров по сервисам compose (несколько воркеров суммируются)."""
    output = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{json .}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    memory: dict[str, float] = {}
    units = {"KiB": 1 / 1024, "MiB": 1.0, "GiB": 1024.0, "B": 1 / 1024 / 1024}
    for line in output.splitlines():
        row = json.loads(line)
        value = row["MemUsage"].split("/")[0].strip()
        unit = next(u for u in ("KiB", "MiB", "GiB", "B") if value.endswith(u))
        amount = float(value.removesuffix(unit)) * units[unit]
        parts = row["Name"].split("-")
        if len(parts) >= 3:
            service = "-".join(parts[2:-1])
            memory[service] = memory.get(service, 0.0) + amount
    return memory


async def connect_db() -> asyncpg.Connection:
    return await asyncpg.connect(DATABASE)
