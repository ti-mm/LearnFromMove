from __future__ import annotations

import os
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Iterable, Iterator, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

MEASURED_LOCAL_WORKER_CAP = 8
MEASURED_SCAN_WORKER_CAP = 4


def recommended_local_workers(*, cpu_count: int | None = None) -> int:
    """Return the measured-safe worker count for local image work."""

    detected = os.cpu_count() if cpu_count is None else cpu_count
    return max(1, min(MEASURED_LOCAL_WORKER_CAP, int(detected or 1)))


def recommended_scan_workers(*, cpu_count: int | None = None) -> int:
    """Return the measured optimum for GroundCUA parquet/image scanning."""

    detected = os.cpu_count() if cpu_count is None else cpu_count
    return max(1, min(MEASURED_SCAN_WORKER_CAP, int(detected or 1)))


def ordered_bounded_map(
    function: Callable[[InputT], OutputT],
    items: Iterable[InputT],
    *,
    max_workers: int,
    max_pending: int | None = None,
) -> Iterator[OutputT]:
    """Run work concurrently with bounded memory and deterministic output order."""

    max_workers = int(max_workers)
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if max_pending is None:
        max_pending = max_workers * 2
    max_pending = int(max_pending)
    if max_pending < max_workers:
        raise ValueError("max_pending must be at least max_workers")

    iterator = iter(items)
    if max_workers == 1:
        for item in iterator:
            yield function(item)
        return

    pending: deque[Future[OutputT]] = deque()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _ in range(max_pending):
            try:
                item = next(iterator)
            except StopIteration:
                break
            pending.append(executor.submit(function, item))

        try:
            while pending:
                future = pending.popleft()
                yield future.result()
                try:
                    item = next(iterator)
                except StopIteration:
                    continue
                pending.append(executor.submit(function, item))
        except BaseException:
            for future in pending:
                future.cancel()
            raise
