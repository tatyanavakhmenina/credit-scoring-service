import csv

import pytest

from credit_scoring.broker import QueueOverloadedError
from credit_scoring.pipeline.source import publish_with_backoff, read_tasks


async def test_backoff_grows_and_stops_after_success():
    delays = []
    attempts = iter([True, True, True, False])

    async def publish():
        if next(attempts):
            raise QueueOverloadedError("full")

    async def sleep(delay):
        delays.append(delay)

    rejections = await publish_with_backoff(publish, 0.1, 1.0, sleep=sleep)

    assert rejections == 3
    assert len(delays) == 3
    assert delays[2] > delays[0]
    assert max(delays) <= 1.0


async def test_backoff_is_capped():
    delays = []
    attempts = iter([True] * 10 + [False])

    async def publish():
        if next(attempts):
            raise QueueOverloadedError("full")

    async def sleep(delay):
        delays.append(delay)

    await publish_with_backoff(publish, 0.1, 0.4, sleep=sleep)

    assert max(delays) <= 0.4


async def test_backoff_does_not_hide_other_errors():
    async def publish():
        raise ValueError("bug")

    with pytest.raises(ValueError, match="bug"):
        await publish_with_backoff(publish, 0.1, 1.0)


def test_read_tasks_ids_are_deterministic_and_limit_loops_file(tmp_path, features):
    path = tmp_path / "data.csv"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[*features, "default"])
        writer.writeheader()
        writer.writerow({**features, "default": 1})

    first = list(read_tasks(str(path), "b", 3))
    second = list(read_tasks(str(path), "b", 3))

    assert len(first) == 3
    assert len({t.task_id for t in first}) == 3
    assert [t.task_id for t in first] == [t.task_id for t in second]
