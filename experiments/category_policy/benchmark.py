"""Warm loaded-service runtime and memory benchmark for category policy V1.1."""

import asyncio
import gc
import resource
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median
from typing import Any

from app.services.categorized_recommendation_service import (
    CategorizedRecommendationService,
)
from experiments.category_policy.evaluate import _atomic_json


def run_warm_category_benchmark(
    *,
    user_ids: tuple[int, ...],
    repetitions: int = 2,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Measure loading and sequential warm requests in one service process."""
    if not user_ids or any(user_id <= 0 for user_id in user_ids):
        raise ValueError("benchmark user IDs must be positive")
    if repetitions <= 0:
        raise ValueError("benchmark repetitions must be positive")
    report = asyncio.run(_run(user_ids, repetitions))
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(report, destination)
    return report


async def _run(user_ids: tuple[int, ...], repetitions: int) -> dict[str, Any]:
    service = CategorizedRecommendationService()
    initial_rss = _current_rss_mib()
    artifacts_started = time.perf_counter()
    if not service.load_candidate_artifacts():
        raise RuntimeError("candidate artifacts are unavailable")
    artifact_load_ms = (time.perf_counter() - artifacts_started) * 1000
    artifact_rss = _current_rss_mib()
    catalog_started = time.perf_counter()
    if not await service.load_policy_catalog():
        raise RuntimeError("policy catalog is unavailable")
    catalog_load_ms = (time.perf_counter() - catalog_started) * 1000
    gc.collect()
    steady_rss = _current_rss_mib()
    try:
        warmup_started = time.perf_counter()
        await service.recommend(user_ids[0])
        warmup_ms = (time.perf_counter() - warmup_started) * 1000
        latencies: list[float] = []
        by_user: dict[int, list[float]] = defaultdict(list)
        for _ in range(repetitions):
            for user_id in user_ids:
                started = time.perf_counter()
                result = await service.recommend(user_id)
                elapsed = (time.perf_counter() - started) * 1000
                latencies.append(elapsed)
                by_user[user_id].append(elapsed)
                del result
        gc.collect()
        post_request_rss = _current_rss_mib()

        traced_peaks: list[float] = []
        tracemalloc.start()
        try:
            for user_id in user_ids:
                gc.collect()
                baseline, _ = tracemalloc.get_traced_memory()
                tracemalloc.reset_peak()
                result = await service.recommend(user_id)
                _, peak = tracemalloc.get_traced_memory()
                traced_peaks.append(max(0, peak - baseline) / 1024 / 1024)
                del result
        finally:
            tracemalloc.stop()
    finally:
        service.unload_resources()
    return {
        "user_ids": list(user_ids),
        "repetitions": repetitions,
        "resource_loading": {
            "artifact_load_ms": artifact_load_ms,
            "policy_catalog_load_ms": catalog_load_ms,
            "initial_rss_mib": initial_rss,
            "rss_after_artifacts_mib": artifact_rss,
            "steady_state_rss_after_load_mib": steady_rss,
            "artifact_rss_delta_mib": artifact_rss - initial_rss,
            "catalog_rss_delta_mib": steady_rss - artifact_rss,
        },
        "warmup_ms": warmup_ms,
        "warm_request_latency_ms": _summary(latencies),
        "warm_request_latency_by_user_ms": {
            str(user_id): _summary(values)
            for user_id, values in sorted(by_user.items())
        },
        "post_request_rss_mib": post_request_rss,
        "peak_temporary_python_request_mib": _summary(traced_peaks),
        "process_high_water_rss_mib": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        ),
        "measurement_notes": [
            "latency pass runs without tracemalloc",
            "temporary-memory pass is separate and profiler-instrumented",
            "RSS is read from the current Linux process after garbage collection",
        ],
    }


def _current_rss_mib() -> float:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("current process RSS is unavailable")


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "mean": fmean(values),
        "median": median(values),
        "max": max(values),
    }
