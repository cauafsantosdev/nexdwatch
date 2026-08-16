"""Loaded-service profiling and exact semantic fingerprints for category V1.1."""

import asyncio
import cProfile
import gc
import hashlib
import json
import os
import platform
import pstats
import sys
import time
import tracemalloc
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from statistics import fmean, median
from typing import Any

import faiss
import numpy as np
import sqlalchemy
from sqlalchemy import event

from app.core.database import engine
from app.policy.catalog import PolicyCatalog, PolicyEntity
from app.policy.request_metrics import (
    CURRENT_CATEGORY_STAGE,
    CategoryRequestProfile,
)
from app.services.categorized_recommendation_service import (
    CategorizedRecommendationService,
)
from experiments.category_policy.evaluate import _atomic_json

DEFAULT_PERFORMANCE_USERS = (3318, 3569, 3155, 2825, 3953, 2504, 2474, 2994, 3724)


def run_serving_performance_profile(
    *,
    user_ids: tuple[int, ...] = DEFAULT_PERFORMANCE_USERS,
    repetitions: int = 3,
    rss_requests: int = 100,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run isolated loaded-service latency, stage, CPU, SQL, and memory passes."""
    if not user_ids or any(user_id <= 0 for user_id in user_ids):
        raise ValueError("performance user IDs must be positive")
    if repetitions <= 0 or rss_requests < 0:
        raise ValueError("repetitions must be positive and RSS requests non-negative")
    report = asyncio.run(_run(user_ids, repetitions, rss_requests))
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(report, destination)
    return report


async def _run(
    user_ids: tuple[int, ...], repetitions: int, rss_requests: int
) -> dict[str, Any]:
    service = CategorizedRecommendationService()
    startup_queries: list[dict[str, Any]] = []
    rss_initial = _current_rss_mib()
    with _sql_recorder(startup_queries, "service_startup"):
        artifact_started = time.perf_counter()
        if not service.load_candidate_artifacts():
            raise RuntimeError("candidate artifacts are unavailable")
        artifact_ms = (time.perf_counter() - artifact_started) * 1000
        rss_after_artifacts = _current_rss_mib()
        catalog_started = time.perf_counter()
        if not await service.load_policy_catalog():
            raise RuntimeError("policy catalog is unavailable")
        catalog_ms = (time.perf_counter() - catalog_started) * 1000
    gc.collect()
    rss_after_load = _current_rss_mib()
    catalog = service._catalog
    if catalog is None:
        raise RuntimeError("policy catalog was not retained")
    catalog_memory = _catalog_memory(catalog)

    try:
        await service.recommend(user_ids[0])
        fingerprints = {}
        latency_values: list[float] = []
        latency_by_user: dict[int, list[float]] = defaultdict(list)
        for _ in range(repetitions):
            for user_id in user_ids:
                started = time.perf_counter()
                result = await service.recommend(user_id)
                elapsed = (time.perf_counter() - started) * 1000
                latency_values.append(elapsed)
                latency_by_user[user_id].append(elapsed)
                fingerprints.setdefault(str(user_id), semantic_fingerprint(result))
                del result

        stage_profiles: list[tuple[int, CategoryRequestProfile]] = []
        for _ in range(repetitions):
            for user_id in user_ids:
                request_profile = CategoryRequestProfile()
                result = await service.recommend(user_id, profiler=request_profile)
                stage_profiles.append((user_id, request_profile))
                del result

        request_queries: list[dict[str, Any]] = []
        with _sql_recorder(request_queries, "request_unscoped"):
            sql_profile = CategoryRequestProfile()
            sql_result = await service.recommend(3953, profiler=sql_profile)
            del sql_result

        cpu_profiler = cProfile.Profile()
        cpu_profiler.enable()
        for user_id in user_ids:
            cpu_result = await service.recommend(user_id)
            del cpu_result
        cpu_profiler.disable()

        allocation_peaks = []
        allocation_snapshots: list[dict[str, Any]] = []
        tracemalloc.start(25)
        try:
            for user_id in user_ids:
                gc.collect()
                baseline, _ = tracemalloc.get_traced_memory()
                tracemalloc.reset_peak()
                allocation_result = await service.recommend(user_id)
                _, peak = tracemalloc.get_traced_memory()
                allocation_peaks.append((peak - baseline) / 1024 / 1024)
                snapshot = tracemalloc.take_snapshot()
                allocation_snapshots.extend(_snapshot_rows(snapshot, user_id, 8))
                del allocation_result
        finally:
            tracemalloc.stop()

        gc.collect()
        rss_before_long_run = _current_rss_mib()
        rss_checkpoints = [{"requests": 0, "rss_mib": rss_before_long_run}]
        for request_number in range(1, rss_requests + 1):
            rss_result = await service.recommend(
                user_ids[(request_number - 1) % len(user_ids)]
            )
            del rss_result
            if request_number % 25 == 0 or request_number == rss_requests:
                gc.collect()
                rss_checkpoints.append(
                    {"requests": request_number, "rss_mib": _current_rss_mib()}
                )

        concurrency = {}
        for width in (1, 2, 4):
            selected = tuple(user_ids[index % len(user_ids)] for index in range(width))
            started = time.perf_counter()
            concurrent_results = await asyncio.gather(
                *(service.recommend(user_id) for user_id in selected)
            )
            wall_ms = (time.perf_counter() - started) * 1000
            concurrency[str(width)] = {
                "user_ids": list(selected),
                "batch_wall_ms": wall_ms,
                "amortized_wall_ms_per_request": wall_ms / width,
                "rss_mib_after_gc": None,
            }
            del concurrent_results
            gc.collect()
            concurrency[str(width)]["rss_mib_after_gc"] = _current_rss_mib()
    finally:
        service.unload_resources()

    return {
        "runtime_context": _runtime_context(),
        "user_ids": list(user_ids),
        "repetitions": repetitions,
        "resource_loading": {
            "artifact_load_ms": artifact_ms,
            "catalog_load_ms": catalog_ms,
            "initial_rss_mib": rss_initial,
            "rss_after_artifacts_mib": rss_after_artifacts,
            "rss_after_load_gc_mib": rss_after_load,
            "artifact_rss_delta_mib": rss_after_artifacts - rss_initial,
            "catalog_rss_delta_mib": rss_after_load - rss_after_artifacts,
        },
        "catalog_memory": catalog_memory,
        "startup_sql": _sql_report(startup_queries),
        "request_sql_user_3953": _sql_report(request_queries),
        "latency_ms": _summary(latency_values),
        "latency_by_user_ms": {
            str(user_id): _summary(values)
            for user_id, values in sorted(latency_by_user.items())
        },
        "stage_profile": _stage_report(stage_profiles),
        "request_shape_by_user": _shape_report(stage_profiles),
        "cpu_profile": _cpu_report(cpu_profiler, 30),
        "temporary_python_allocation_mib": _summary(allocation_peaks),
        "allocation_hotspots": sorted(
            allocation_snapshots,
            key=lambda value: value["size_mib"],
            reverse=True,
        )[:30],
        "long_rss": {
            "rotating_requests": rss_requests,
            "checkpoints": rss_checkpoints,
            "growth_mib": (
                rss_checkpoints[-1]["rss_mib"] - rss_checkpoints[0]["rss_mib"]
            ),
        },
        "concurrency_smoke": concurrency,
        "fingerprints": fingerprints,
        "measurement_notes": [
            "latency pass is uninstrumented and excludes startup",
            "stage, SQL, cProfile, tracemalloc, RSS, and concurrency passes are separate",
            "concurrency is a local service-level smoke test, not throughput evidence",
        ],
    }


def semantic_fingerprint(result: Any) -> dict[str, Any]:
    """Canonicalize every deterministic recommendation output and hash it."""
    canonical = _canonical(result)
    encoded = json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "result": canonical}


def compare_fingerprint_reports(
    baseline_path: str | Path, optimized_path: str | Path
) -> dict[str, Any]:
    """Compare representative-user semantic fingerprints exactly."""
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    optimized = json.loads(Path(optimized_path).read_text(encoding="utf-8"))
    baseline_values = baseline["fingerprints"]
    optimized_values = optimized["fingerprints"]
    all_users = sorted(set(baseline_values) | set(optimized_values), key=int)
    differences = [
        user_id
        for user_id in all_users
        if baseline_values.get(user_id) != optimized_values.get(user_id)
    ]
    return {
        "equal": not differences,
        "users": all_users,
        "different_user_ids": differences,
        "baseline_sha256": {
            key: value["sha256"] for key, value in baseline_values.items()
        },
        "optimized_sha256": {
            key: value["sha256"] for key, value in optimized_values.items()
        },
    }


@contextmanager
def _sql_recorder(
    records: list[dict[str, Any]], fallback_purpose: str
) -> Iterator[None]:
    starts: list[float] = []

    def before_cursor_execute(*_args) -> None:
        starts.append(time.perf_counter())

    def after_cursor_execute(_conn, _cursor, statement, _parameters, context, _many):
        started = starts.pop()
        records.append(
            {
                "purpose": CURRENT_CATEGORY_STAGE.get() or fallback_purpose,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "rows": context.cursor.rowcount
                if context.cursor.rowcount >= 0
                else None,
                "operation": statement.lstrip().split(None, 1)[0].upper(),
                "statement_shape": " ".join(statement.split())[:240],
            }
        )

    event.listen(engine.sync_engine, "before_cursor_execute", before_cursor_execute)
    event.listen(engine.sync_engine, "after_cursor_execute", after_cursor_execute)
    try:
        yield
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", before_cursor_execute)
        event.remove(engine.sync_engine, "after_cursor_execute", after_cursor_execute)


def _canonical(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field.name: _canonical(getattr(value, field.name))
            for field in fields(value)
            # Poster/link metadata is presentation-only and must not redefine the
            # canonical recommendation identity, ordering, or explanation fingerprint.
            if field.name not in {"tmdb_id", "slug", "preference_context"}
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_canonical(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "mean": fmean(ordered),
        "median": median(ordered),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def _percentile(ordered: list[float], quantile: float) -> float:
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _stage_report(
    profiles: list[tuple[int, CategoryRequestProfile]],
) -> dict[str, Any]:
    values: dict[str, list[float]] = defaultdict(list)
    total_values = []
    for _, profile in profiles:
        milliseconds = profile.milliseconds()
        total = milliseconds.get("total_request", 0.0)
        total_values.append(total)
        for name, elapsed in milliseconds.items():
            values[name].append(elapsed)
    total_mean = fmean(total_values) if total_values else 0.0
    return {
        name: {
            **_summary(elapsed),
            "mean_share_of_total": fmean(elapsed) / total_mean if total_mean else 0.0,
        }
        for name, elapsed in sorted(values.items())
    }


def _shape_report(
    profiles: list[tuple[int, CategoryRequestProfile]],
) -> dict[str, Any]:
    by_user: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for user_id, profile in profiles:
        by_user[user_id].append(dict(sorted(profile.counters.items())))
    return {str(user_id): values[0] for user_id, values in sorted(by_user.items())}


def _sql_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "query_count": len(records),
        "total_elapsed_ms": sum(value["elapsed_ms"] for value in records),
        "queries": records,
    }


def _cpu_report(profiler: cProfile.Profile, limit: int) -> dict[str, Any]:
    stats = pstats.Stats(profiler)
    rows = []
    for (filename, line, function), (
        primitive,
        calls,
        self_time,
        cumulative,
        _,
    ) in stats.stats.items():
        rows.append(
            {
                "function": f"{Path(filename).name}:{line}({function})",
                "primitive_calls": primitive,
                "calls": calls,
                "self_seconds": self_time,
                "cumulative_seconds": cumulative,
            }
        )
    return {
        "total_calls": stats.total_calls,
        "total_seconds": stats.total_tt,
        "top_cumulative": sorted(
            rows, key=lambda value: value["cumulative_seconds"], reverse=True
        )[:limit],
        "top_self": sorted(rows, key=lambda value: value["self_seconds"], reverse=True)[
            :limit
        ],
    }


def _snapshot_rows(snapshot: tracemalloc.Snapshot, user_id: int, limit: int):
    rows = []
    for statistic in snapshot.statistics("lineno")[:limit]:
        frame = statistic.traceback[0]
        rows.append(
            {
                "user_id": user_id,
                "location": f"{frame.filename}:{frame.lineno}",
                "size_mib": statistic.size / 1024 / 1024,
                "count": statistic.count,
            }
        )
    return rows


def _catalog_memory(catalog: PolicyCatalog) -> dict[str, Any]:
    films = tuple(catalog.films.values())
    entities = {
        id(entity): entity
        for film in films
        for family in ("director", "genre", "country", "language")
        for entity in film.entities(family)
    }
    title_bytes = sum(sys.getsizeof(film.title) for film in films)
    tuple_bytes = {
        family: sum(sys.getsizeof(film.entities(family)) for film in films)
        for family in ("director", "genre", "country", "language")
    }
    entity_bytes = sum(
        sys.getsizeof(entity) + sys.getsizeof(entity.name)
        for entity in entities.values()
        if isinstance(entity, PolicyEntity)
    )
    return {
        "film_count": len(films),
        "unique_entity_count": len(entities),
        "policy_film_shallow_mib": sum(sys.getsizeof(film) for film in films)
        / 1024
        / 1024,
        "title_strings_mib": title_bytes / 1024 / 1024,
        "relation_tuples_mib": {
            key: value / 1024 / 1024 for key, value in tuple_bytes.items()
        },
        "unique_entities_and_names_mib": entity_bytes / 1024 / 1024,
        "film_dictionary_shallow_mib": sys.getsizeof(catalog.films) / 1024 / 1024,
        "artifact_id_set_shallow_mib": sys.getsizeof(catalog.artifact_film_ids)
        / 1024
        / 1024,
        "measurement": "sys.getsizeof components; shared referents counted once only where stated",
    }


def _runtime_context() -> dict[str, Any]:
    cpu_model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {
        "timestamp_timezone": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "faiss": getattr(faiss, "__version__", "unknown"),
        "sqlalchemy": sqlalchemy.__version__,
    }


def _current_rss_mib() -> float:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("current process RSS is unavailable")
