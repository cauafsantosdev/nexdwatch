"""Deterministic V1/V1.1 qualitative category previews for review, not scoring."""

import asyncio
from pathlib import Path
from typing import Any

from app.policy.config import DEFAULT_POLICY_CONFIG, V1_POLICY_CONFIG
from app.services.categorized_recommendation_service import (
    CategorizedRecommendationService,
)
from experiments.category_policy.evaluate import _atomic_json

CHANGED_CATEGORY_KEYS = {
    "because_you_liked",
    "classic_cinema",
    "world_cinema",
    "outside_usual",
    "directors_you_love",
}


def run_qualitative_previews(
    *, user_ids: tuple[int, ...], output_path: str | Path
) -> dict[str, Any]:
    """Render safe deterministic examples for both policy versions."""
    if not user_ids or any(user_id <= 0 for user_id in user_ids):
        raise ValueError("preview user IDs must be positive")
    report = asyncio.run(_run(user_ids))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(report, destination)
    return report


async def _run(user_ids: tuple[int, ...]) -> dict[str, Any]:
    versions = {}
    for name, config in (
        ("v1", V1_POLICY_CONFIG),
        ("v1_1", DEFAULT_POLICY_CONFIG),
    ):
        service = CategorizedRecommendationService(config=config)
        try:
            if not await service.load_resources():
                raise RuntimeError("category resources are unavailable")
            versions[name] = {
                str(user_id): _safe_result(await service.recommend(user_id))
                for user_id in user_ids
            }
        finally:
            service.unload_resources()
    return {
        "purpose": "deterministic qualitative review; not user-satisfaction evidence",
        "user_ids": list(user_ids),
        "versions": versions,
    }


def _safe_result(result) -> dict[str, Any]:
    categories = {}
    for category in result.categories:
        if category.key not in CHANGED_CATEGORY_KEYS:
            continue
        categories[category.key] = {
            "title": category.title,
            "size": len(category.items),
            "strata": _strata(category.items),
            "items": [
                {
                    "film_id": item.film_id,
                    "title": item.title,
                    "year": item.year,
                    "reason": item.reason.code,
                    "entity_name": item.reason.entity_name,
                    "support_count": item.reason.support_count,
                    "popularity_stratum": item.popularity_stratum,
                    "source_membership": item.source_membership,
                    "rrf_rank": item.rrf_rank,
                }
                for item in category.items[:10]
            ],
        }
    return {
        "category_order": [category.key for category in result.categories],
        "history_depth_band": result.diagnostics["history_depth_band"],
        "categories": categories,
        "anchor": result.diagnostics["anchor"],
        "director_pool": result.diagnostics["director_pool"],
        "allocation": result.diagnostics["allocation"],
    }


def _strata(items) -> dict[str, int]:
    return {
        stratum: sum(item.popularity_stratum == stratum for item in items)
        for stratum in ("HEAD", "MID", "TAIL")
    }
