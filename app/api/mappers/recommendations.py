"""Map categorized recommendation domain results to the public feed contract."""

from dataclasses import dataclass
from types import MappingProxyType

from app.api.schemas.recommendations import (
    RecommendationAnchorResponse,
    RecommendationCategoryResponse,
    RecommendationEntityResponse,
    RecommendationFeedItemResponse,
    RecommendationFeedResponse,
    RecommendationReasonResponse,
)
from app.domain.categorized_recommendations import (
    CategorizedRecommendationResult,
    RecommendationReason,
)


@dataclass(frozen=True, slots=True)
class CategoryProductMetadata:
    """Frontend-facing rollout metadata independent of policy implementation."""

    experimental: bool = False


DEFAULT_CATEGORY_PRODUCT_METADATA = CategoryProductMetadata()
CATEGORY_PRODUCT_METADATA = MappingProxyType(
    {"outside_usual": CategoryProductMetadata(experimental=True)}
)


def map_recommendation_feed(
    result: CategorizedRecommendationResult,
) -> RecommendationFeedResponse:
    """Preserve policy order while removing model and diagnostic implementation data.

    Only product-approved reason fields cross this boundary. RRF scores, ranks,
    affinities, support diagnostics, and internal category roles remain private.

    Args:
        result: Internal categorized domain result after policy allocation.

    Returns:
        RecommendationFeedResponse: Policy-ordered, non-empty shelves containing only
            approved display and reason fields.
    """
    return RecommendationFeedResponse(
        user_id=result.user_id,
        categories=[
            RecommendationCategoryResponse(
                key=category.key,
                title=category.title,
                experimental=CATEGORY_PRODUCT_METADATA.get(
                    category.key, DEFAULT_CATEGORY_PRODUCT_METADATA
                ).experimental,
                items=[
                    RecommendationFeedItemResponse(
                        film_id=item.film_id,
                        title=item.title,
                        year=item.year,
                        directors=list(item.directors),
                        reason=_map_reason(item.reason),
                    )
                    for item in category.items
                ],
            )
            for category in result.categories
            if category.items
        ],
    )


def _map_reason(reason: RecommendationReason) -> RecommendationReasonResponse:
    """Project an internal reason to approved code, anchor, and entity fields.

    Partial anchors/entities are omitted as a pair so inconsistent internal optional
    metadata cannot leak a malformed public object. Support and ranking evidence are
    intentionally discarded.
    """
    anchor = None
    if reason.anchor_film_id is not None and reason.anchor_title is not None:
        anchor = RecommendationAnchorResponse(
            film_id=reason.anchor_film_id,
            title=reason.anchor_title,
        )
    entity = None
    if reason.entity_family is not None and reason.entity_name is not None:
        entity = RecommendationEntityResponse(
            type=reason.entity_family,
            name=reason.entity_name,
        )
    return RecommendationReasonResponse(
        code=reason.code.value,
        anchor=anchor,
        entity=entity,
    )
