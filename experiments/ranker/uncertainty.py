"""User-clustered uncertainty for repeated seeded test appearances."""

from statistics import fmean
from typing import Any

import numpy as np


def paired_user_clustered_comparisons(
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bootstrap unique users after averaging their repeated-seed deltas."""
    if "full" not in reports[0]["models"]:
        return {}
    comparisons: dict[str, Any] = {}
    for baseline in ("rrf", "positive_weighted_svd"):
        deltas_by_user: dict[int, list[float]] = {}
        observation_count = 0
        for report in reports:
            full_rows = report["models"]["full"]["test"]["per_user_global"]
            base_rows = report["baselines"]["test"][baseline]["per_user_global"]
            for full, base in zip(full_rows, base_rows, strict=True):
                if full["user_id"] != base["user_id"]:
                    raise RuntimeError("paired ranker rows have different users")
                full_rank = full["target_rank"]
                base_rank = base["target_rank"]
                full_ndcg = (
                    1 / np.log2(full_rank + 1) if full_rank and full_rank <= 20 else 0.0
                )
                base_ndcg = (
                    1 / np.log2(base_rank + 1) if base_rank and base_rank <= 20 else 0.0
                )
                deltas_by_user.setdefault(int(full["user_id"]), []).append(
                    float(full_ndcg - base_ndcg)
                )
                observation_count += 1
        clustered_deltas = np.asarray(
            [fmean(values) for _, values in sorted(deltas_by_user.items())],
            dtype=np.float64,
        )
        rng = np.random.default_rng(20260810)
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
                    for _ in range(5000)
                ]
            )
            if len(clustered_deltas)
            else np.zeros(1)
        )
        comparisons[baseline] = {
            "user_clustered_ndcg_at_20_delta": (
                float(clustered_deltas.mean()) if len(clustered_deltas) else 0.0
            ),
            "user_clustered_bootstrap_95_percent_ci": [
                float(np.quantile(boot, 0.025)),
                float(np.quantile(boot, 0.975)),
            ],
            "unique_historical_users": len(clustered_deltas),
            "repeated_seed_observations": observation_count,
            "bootstrap_clusters": "historical_user_id",
        }
    return comparisons
