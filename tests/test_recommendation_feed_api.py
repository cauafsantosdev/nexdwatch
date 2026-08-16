"""Public categorized-feed mapping and route behavior."""

import asyncio
from dataclasses import replace

import httpx
import pytest
from fastapi import FastAPI

from app.api.mappers.recommendations import map_recommendation_feed
from app.api.routes.recommendations import (
    get_categorized_recommendation_service,
    router,
)
from app.domain.categorized_recommendations import (
    CategorizedRecommendation,
    CategorizedRecommendationResult,
    CategoryPreferenceContext,
    CategoryRole,
    RecommendationCategory,
    RecommendationReason,
    RecommendationReasonCode,
)
from app.services.categorized_recommendation_service import (
    CategoryPolicyResourcesUnavailableError,
    RecommendationUserNotFoundError,
)


class _FeedService:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.is_loaded = True
        self.calls: list[int] = []

    async def recommend(self, user_id: int):
        self.calls.append(user_id)
        if self.error is not None:
            raise self.error
        return self.result


def _item(
    film_id: int,
    reason: RecommendationReason,
) -> CategorizedRecommendation:
    return CategorizedRecommendation(
        film_id=film_id,
        title=f"Film {film_id}",
        year=2000 + film_id,
        directors=("Director",),
        tmdb_id=1000 + film_id if film_id % 2 else None,
        slug=f"film-{film_id}",
        reason=reason,
        rrf_rank=film_id,
        popularity_stratum="TAIL",
        source_membership="svd_only",
    )


def _category(
    key: str,
    title: str,
    *items: CategorizedRecommendation,
    preference_context: CategoryPreferenceContext | None = None,
) -> RecommendationCategory:
    return RecommendationCategory(
        key=key,
        family="test-internal-family",
        role=CategoryRole.PERSONALIZED,
        title=title,
        items=items,
        evidence_tier="strong",
        evidence_support=99,
        preference_context=preference_context,
    )


def _result() -> CategorizedRecommendationResult:
    anchor_reason = RecommendationReason(
        RecommendationReasonCode.ANCHOR_SIMILARITY,
        anchor_film_id=44,
        anchor_title="Anchor Film",
        popularity_stratum="TAIL",
        retrieved_by_both=True,
    )
    director_reason = RecommendationReason(
        RecommendationReasonCode.DIRECTOR_AFFINITY,
        entity_family="director",
        entity_name="Andrei Tarkovsky",
        support_count=8,
        high_rating_count=6,
    )
    return CategorizedRecommendationResult(
        user_id=3953,
        categories=(
            _category(
                "top_picks",
                "Top Picks",
                _item(1, RecommendationReason(RecommendationReasonCode.GLOBAL_RRF)),
            ),
            _category(
                "because_you_liked",
                "Because You Liked Anchor Film",
                _item(2, anchor_reason),
            ),
            _category(
                "directors_you_love",
                "From Directors You Love",
                _item(3, director_reason),
            ),
            _category(
                "outside_usual",
                "Outside Your Usual Picks",
                _item(
                    4,
                    RecommendationReason(
                        RecommendationReasonCode.LATENT_MATCH_METADATA_NOVELTY
                    ),
                ),
            ),
        ),
        diagnostics={"raw_internal_policy": "must not escape"},
    )


def _application(service: _FeedService | None = None) -> FastAPI:
    application = FastAPI()
    application.include_router(router)
    if service is not None:

        async def override_service():
            return service

        application.dependency_overrides[get_categorized_recommendation_service] = (
            override_service
        )
    return application


def _get(path: str, service: _FeedService | None = None) -> httpx.Response:
    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=_application(service))
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.get(path)

    return asyncio.run(request())


def test_successful_feed_preserves_order_and_exposes_only_product_fields() -> None:
    service = _FeedService(_result())

    response = _get("/recommendations/3953/feed", service)

    assert response.status_code == 200
    payload = response.json()
    assert service.calls == [3953]
    assert [category["key"] for category in payload["categories"]] == [
        "top_picks",
        "because_you_liked",
        "directors_you_love",
        "outside_usual",
    ]
    assert [category["experimental"] for category in payload["categories"]] == [
        False,
        False,
        False,
        True,
    ]
    assert set(payload) == {"user_id", "categories"}
    assert set(payload["categories"][0]) == {
        "key",
        "title",
        "experimental",
        "items",
    }
    assert set(payload["categories"][0]["items"][0]) == {
        "film_id",
        "title",
        "year",
        "directors",
        "tmdb_id",
        "slug",
        "reason",
    }
    assert payload["categories"][0]["items"][0]["tmdb_id"] == 1001
    assert payload["categories"][0]["items"][0]["slug"] == "film-1"
    assert "preference_context" not in payload["categories"][0]
    serialized = response.text
    for internal_name in (
        "rrf_rank",
        "rrf_score",
        "svd_score",
        "popularity_stratum",
        "source_membership",
        "evidence_support",
        "support_count",
        "entity_id",
        "diagnostics",
    ):
        assert internal_name not in serialized


def test_anchor_and_entity_reasons_are_compact_and_named() -> None:
    payload = map_recommendation_feed(_result()).model_dump(exclude_none=True)

    anchor = payload["categories"][1]["items"][0]["reason"]
    director = payload["categories"][2]["items"][0]["reason"]
    assert anchor == {
        "code": "ANCHOR_SIMILARITY",
        "anchor": {"film_id": 44, "title": "Anchor Film"},
    }
    assert director == {
        "code": "DIRECTOR_AFFINITY",
        "entity": {"type": "director", "name": "Andrei Tarkovsky"},
    }


@pytest.mark.parametrize(
    ("code", "family", "name"),
    [
        (RecommendationReasonCode.GENRE_AFFINITY, "genre", "Western"),
        (RecommendationReasonCode.DECADE_AFFINITY, "decade", "1970s"),
    ],
)
def test_genre_and_decade_reason_mapping(code, family, name) -> None:
    reason = RecommendationReason(code, entity_family=family, entity_name=name)
    result = replace(
        _result(),
        categories=(_category("preference", "Preference", _item(9, reason)),),
    )

    public_reason = map_recommendation_feed(result).model_dump(exclude_none=True)[
        "categories"
    ][0]["items"][0]["reason"]

    assert public_reason == {
        "code": code.value,
        "entity": {"type": family, "name": name},
    }


def test_preference_context_is_public_only_for_preference_categories() -> None:
    original = _result()
    genre = _category(
        "favorite_genre",
        "Western Picks for You",
        _item(
            9,
            RecommendationReason(
                RecommendationReasonCode.GENRE_AFFINITY,
                entity_family="genre",
                entity_name="Western",
            ),
        ),
        preference_context=CategoryPreferenceContext(4.25, 9),
    )
    decade = _category(
        "favorite_decade",
        "1990s Films for You",
        _item(
            10,
            RecommendationReason(
                RecommendationReasonCode.DECADE_AFFINITY,
                entity_family="decade",
                entity_name="1990s",
            ),
        ),
        preference_context=CategoryPreferenceContext(4.125, 24),
    )
    result = replace(original, categories=(original.categories[0], genre, decade))

    payload = map_recommendation_feed(result).model_dump()

    assert [category["key"] for category in payload["categories"]] == [
        "top_picks",
        "favorite_genre",
        "favorite_decade",
    ]
    assert [
        item["film_id"]
        for category in payload["categories"]
        for item in category["items"]
    ] == [1, 9, 10]
    assert payload["categories"][0]["preference_context"] is None
    assert payload["categories"][1]["preference_context"] == {
        "average_rating": 4.25,
        "rated_count": 9,
    }
    assert payload["categories"][2]["preference_context"] == {
        "average_rating": 4.125,
        "rated_count": 24,
    }


@pytest.mark.parametrize(
    "code",
    [
        RecommendationReasonCode.NON_HEAD_TASTE_MATCH,
        RecommendationReasonCode.BRAZILIAN_CINEMA_DISCOVERY,
        RecommendationReasonCode.WORLD_CINEMA_DISCOVERY,
        RecommendationReasonCode.CLASSIC_CINEMA_DISCOVERY,
        RecommendationReasonCode.LATENT_MATCH_METADATA_NOVELTY,
    ],
)
def test_discovery_reason_mapping_exposes_only_the_safe_code(code) -> None:
    reason = RecommendationReason(
        code,
        additional_codes=(RecommendationReasonCode.SOURCE_AGREEMENT,),
        popularity_stratum="MID",
        evidence_tier="strong",
        retrieved_by_both=True,
    )
    result = replace(
        _result(),
        categories=(_category("discovery", "Discovery", _item(8, reason)),),
    )

    public_reason = map_recommendation_feed(result).model_dump(exclude_none=True)[
        "categories"
    ][0]["items"][0]["reason"]

    assert public_reason == {"code": code.value}


def test_mapping_is_deterministic_and_omits_inactive_empty_categories() -> None:
    result = replace(
        _result(),
        categories=(*_result().categories, _category("inactive", "Inactive")),
    )

    first = map_recommendation_feed(result)
    second = map_recommendation_feed(result)

    assert first == second
    assert all(category.key != "inactive" for category in first.categories)


@pytest.mark.parametrize(
    ("error", "status_code", "detail"),
    [
        (RecommendationUserNotFoundError(99), 404, "User not found."),
        (
            CategoryPolicyResourcesUnavailableError(),
            503,
            "Categorized recommendations are temporarily unavailable.",
        ),
        (
            RuntimeError("private failure"),
            500,
            "Categorized recommendation generation failed.",
        ),
    ],
)
def test_feed_errors_are_safe(error, status_code, detail) -> None:
    response = _get("/recommendations/99/feed", _FeedService(error=error))

    assert response.status_code == status_code
    assert response.json() == {"detail": detail}
    assert "private failure" not in response.text


def test_missing_application_resource_returns_503() -> None:
    response = _get("/recommendations/3953/feed")

    assert response.status_code == 503


def test_openapi_uses_product_schema_and_keeps_existing_route() -> None:
    application = FastAPI()
    application.include_router(router)
    schema = application.openapi()

    assert "/recommendations/{user_id}/feed" in schema["paths"]
    assert "/users/{user_id}/recommendations" in schema["paths"]
    feed_schema = schema["components"]["schemas"]["RecommendationFeedResponse"]
    serialized = str(feed_schema)
    assert "Internal" not in serialized
    assert "RankedCandidate" not in str(schema)
    assert "CategoryProposal" not in str(schema)
    assert "SOURCE_AGREEMENT" not in str(
        schema["components"]["schemas"]["RecommendationReasonResponse"]
    )
