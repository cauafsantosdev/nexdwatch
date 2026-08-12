"""User-clustered uncertainty for repeated seeded test appearances."""

from statistics import fmean
from typing import Any

import numpy as np


def user_clustered_ndcg_at_20_comparison(
    strategy_rows: list[dict[str, int | None]],
    baseline_rows: list[dict[str, int | None]],
    *,
    bootstrap_seed: int = 20260810,
    bootstrap_repetitions: int = 5000,
) -> dict[str, Any]:
    """Cluster paired rank deltas by historical user before bootstrapping."""
    if len(strategy_rows) != len(baseline_rows):
        raise ValueError("paired ranking observations differ in length")
    deltas_by_user: dict[int, list[float]] = {}
    for strategy, baseline in zip(strategy_rows, baseline_rows, strict=True):
        if strategy["user_id"] != baseline["user_id"]:
            raise RuntimeError("paired ranker rows have different users")
        strategy_rank = strategy["target_rank"]
        baseline_rank = baseline["target_rank"]
        strategy_ndcg = (
            1 / np.log2(strategy_rank + 1)
            if strategy_rank and strategy_rank <= 20
            else 0.0
        )
        baseline_ndcg = (
            1 / np.log2(baseline_rank + 1)
            if baseline_rank and baseline_rank <= 20
            else 0.0
        )
        deltas_by_user.setdefault(int(strategy["user_id"]), []).append(
            float(strategy_ndcg - baseline_ndcg)
        )
    clustered_deltas = np.asarray(
        [fmean(values) for _, values in sorted(deltas_by_user.items())],
        dtype=np.float64,
    )
    rng = np.random.default_rng(bootstrap_seed)
    boot = (
        np.asarray(
            [
                float(
                    rng.choice(
                        clustered_deltas,
                        size=len(clustered_deltas),
                        replace=True,
                    ).mean()
                )
                for _ in range(bootstrap_repetitions)
            ]
        )
        if len(clustered_deltas)
        else np.zeros(1)
    )
    return {
        "user_clustered_ndcg_at_20_delta": (
            float(clustered_deltas.mean()) if len(clustered_deltas) else 0.0
        ),
        "user_clustered_bootstrap_95_percent_ci": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "unique_historical_users": len(clustered_deltas),
        "repeated_seed_observations": len(strategy_rows),
        "bootstrap_clusters": "historical_user_id",
    }


def paired_user_clustered_comparisons(
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bootstrap unique users after averaging their repeated-seed deltas."""
    if "full" not in reports[0]["models"]:
        return {}
    comparisons: dict[str, Any] = {}
    for baseline in ("rrf", "positive_weighted_svd"):
        strategy_rows: list[dict[str, int | None]] = []
        baseline_rows: list[dict[str, int | None]] = []
        for report in reports:
            full_rows = report["models"]["full"]["test"]["per_user_global"]
            base_rows = report["baselines"]["test"][baseline]["per_user_global"]
            strategy_rows.extend(full_rows)
            baseline_rows.extend(base_rows)
        comparisons[baseline] = user_clustered_ndcg_at_20_comparison(
            strategy_rows, baseline_rows
        )
    return comparisons
