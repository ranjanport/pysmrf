"""Parallel execution utilities, thread/process pools, and spatial partitioning for PySMRF."""

from __future__ import annotations

import os
import math
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Any, Callable, Iterable, List, Optional, Tuple, TypeVar
import numpy as np

T = TypeVar("T")


def get_worker_count(workers: Optional[int] = 1) -> int:
    """Resolve worker count.

    - None or 0 or 1 -> 1 worker (serial).
    - -1 or < 0 -> os.cpu_count() cores.
    - > 1 -> specified number of workers.
    """
    if workers is None or workers == 0:
        return 1
    if workers < 0:
        count = os.cpu_count()
        return count if count is not None and count > 0 else 1
    return max(1, int(workers))


def chunk_indices(total_size: int, n_chunks: int) -> List[Tuple[int, int]]:
    """Split an integer range [0, total_size) into approximately equal contiguous chunks."""
    if total_size <= 0 or n_chunks <= 0:
        return []
    n_chunks = min(n_chunks, total_size)
    chunk_size = math.ceil(total_size / n_chunks)
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(total_size, (i + 1) * chunk_size)
        if start < end:
            chunks.append((start, end))
    return chunks


def parallel_map(
    func: Callable[[T], Any],
    items: Iterable[T],
    workers: int = 1,
    use_processes: bool = False,
) -> List[Any]:
    """Execute a function over items in parallel using threads or processes."""
    workers = get_worker_count(workers)
    items_list = list(items)
    if len(items_list) == 0:
        return []
    if workers <= 1 or len(items_list) == 1:
        return [func(item) for item in items_list]

    executor_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    with executor_cls(max_workers=min(workers, len(items_list))) as executor:
        return list(executor.map(func, items_list))
