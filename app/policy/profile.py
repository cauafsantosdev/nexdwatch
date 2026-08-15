"""Request-scoped support-aware user preference construction."""

from dataclasses import dataclass
from statistics import fmean

from app.domain.categorized_recommendations import (
    AnchorPreference,
    EntityFamily,
    EntityPreferenceRecord,
    UserCategoryProfile,
)
from app.policy.catalog import PolicyCatalog, PolicyEntity
from app.policy.config import DEFAULT_POLICY_CONFIG, CategoryPolicyConfig
from app.policy.request_metrics import (
    CategoryRequestProfile,
    request_stage,
)
from app.repositories.interactions import RecommendationHistory


@dataclass(slots=True)
class _PreferenceAccumulator:
    entity: PolicyEntity
    ratings: list[float]


def build_user_category_profile(
    user_id: int,
    history: RecommendationHistory,
    catalog: PolicyCatalog,
    *,
    config: CategoryPolicyConfig = DEFAULT_POLICY_CONFIG,
    profiler: CategoryRequestProfile | None = None,
) -> UserCategoryProfile:
    """Build one support-aware profile from explicit ratings and watched depth.

    Only catalog-resolved explicit ratings inform affinity. Each entity family's
    deviation from the user's own mean is confidence-smoothed, while anchors require
    model-indexed positive evidence. Watched-but-unrated films affect only history
    depth and never preference strength.

    Args:
        user_id: Persisted identity carried into policy output.
        history: Single request snapshot of watched and rated interactions.
        catalog: Immutable policy metadata bounded to the model vocabulary.
        config: Frozen support, rating, and history-band thresholds.
        profiler: Optional request-local measurements.

    Returns:
        UserCategoryProfile: Deterministically ordered preferences, anchors, rating
            counts, and the sparse/established/deep history band.
    """
    # Restrict evidence to metadata-resolved explicit ratings; watched-only entries
    # remain useful solely for history depth and candidate exclusion upstream.
    with request_stage(profiler, "profile_rating_aggregation"):
        rated = tuple(
            interaction
            for interaction in history.rated_interactions
            if interaction.film_id in catalog.films
        )
        ratings = [interaction.rating for interaction in rated]
        user_mean = fmean(ratings) if ratings else None
    # Build each family independently so support cannot leak between directors,
    # genres, decades, countries, and languages.
    preferences: dict[EntityFamily, tuple[EntityPreferenceRecord, ...]] = {}
    for family in ("director", "genre", "decade", "country", "language"):
        with request_stage(profiler, f"profile_{family}_preferences"):
            preferences[family] = _family_preferences(
                family,
                rated,
                user_mean,
                catalog,
                config,
            )

    # Anchors must exist in the vector vocabulary because their later neighborhood
    # proposal computes exact item-vector similarities.
    anchors = tuple(
        sorted(
            (
                AnchorPreference(
                    film_id=interaction.film_id,
                    title=catalog.films[interaction.film_id].title,
                    rating=interaction.rating,
                )
                for interaction in rated
                if interaction.rating >= config.anchor_fallback_rating
                and interaction.film_id in catalog.artifact_film_ids
            ),
            key=lambda anchor: (-anchor.rating, anchor.film_id),
        )
    )
    positive_count = sum(value >= config.positive_rating for value in ratings)
    indexed_positive_count = sum(
        interaction.rating >= config.positive_rating
        and interaction.film_id in catalog.artifact_film_ids
        for interaction in rated
    )
    watched_count = len(set(history.watched_film_ids))
    # History bands alter portfolio role priority, not underlying affinity scores.
    if len(rated) < config.sparse_rated_threshold:
        history_band = "sparse"
    elif watched_count >= config.deep_watched_threshold:
        history_band = "deep"
    else:
        history_band = "established"
    profile = UserCategoryProfile(
        user_id=user_id,
        watched_count=watched_count,
        rated_count=len(rated),
        user_mean_rating=user_mean,
        positive_count=positive_count,
        indexed_positive_count=indexed_positive_count,
        high_count=sum(value >= config.high_rating for value in ratings),
        negative_count=sum(value <= config.negative_rating for value in ratings),
        anchors=anchors,
        preferences=preferences,
        history_depth_band=history_band,
    )
    if profiler is not None:
        profiler.count("profile_rated_count", len(rated))
        profiler.count("eligible_anchor_count", len(anchors))
        profiler.count(
            "maximum_rating_anchor_count",
            sum(anchor.rating == anchors[0].rating for anchor in anchors)
            if anchors
            else 0,
        )
        profiler.count("maximum_anchor_rating", anchors[0].rating if anchors else None)
    return profile


def _family_preferences(
    family: EntityFamily,
    rated: tuple,
    user_mean: float | None,
    catalog: PolicyCatalog,
    config: CategoryPolicyConfig,
) -> tuple[EntityPreferenceRecord, ...]:
    """Aggregate and deterministically rank one metadata family's preferences."""
    if user_mean is None:
        return ()
    accumulators: dict[int, _PreferenceAccumulator] = {}
    for interaction in rated:
        film = catalog.films[interaction.film_id]
        for entity in film.entities(family):
            accumulator = accumulators.setdefault(
                entity.id, _PreferenceAccumulator(entity=entity, ratings=[])
            )
            accumulator.ratings.append(interaction.rating)
    smoothing = (
        config.director_smoothing
        if family == "director"
        else config.broad_entity_smoothing
    )
    records = [
        _preference_record(family, accumulator, user_mean, smoothing, config)
        for accumulator in accumulators.values()
    ]
    return tuple(
        sorted(
            records,
            key=lambda value: (
                -value.affinity,
                -value.support_count,
                -value.mean_rating,
                value.entity_id,
            ),
        )
    )


def _preference_record(
    family: EntityFamily,
    accumulator: _PreferenceAccumulator,
    user_mean: float,
    smoothing: float,
    config: CategoryPolicyConfig,
) -> EntityPreferenceRecord:
    """Create one confidence-shrunk affinity relative to the user's mean rating."""
    ratings = accumulator.ratings
    support = len(ratings)
    mean_rating = fmean(ratings)
    raw_preference = fmean(rating - user_mean for rating in ratings)
    confidence = support / (support + smoothing)
    positive_count = sum(rating >= config.positive_rating for rating in ratings)
    high_count = sum(rating >= config.high_rating for rating in ratings)
    negative_count = sum(rating <= config.negative_rating for rating in ratings)
    return EntityPreferenceRecord(
        family=family,
        entity_id=accumulator.entity.id,
        name=accumulator.entity.name,
        support_count=support,
        mean_rating=mean_rating,
        positive_count=positive_count,
        high_rating_count=high_count,
        negative_count=negative_count,
        positive_fraction=positive_count / support,
        high_rating_fraction=high_count / support,
        raw_preference=raw_preference,
        confidence=confidence,
        affinity=raw_preference * confidence,
    )


def preference_evidence_tier(
    preference: EntityPreferenceRecord,
    *,
    config: CategoryPolicyConfig = DEFAULT_POLICY_CONFIG,
) -> str:
    """Classify qualified preference support under frozen V1.1 thresholds."""
    minimum = (
        config.director_minimum_support
        if preference.family == "director"
        else config.broad_minimum_support
    )
    return "strong" if preference.support_count >= 2 * minimum else "minimum"


def qualifying_preferences(
    profile: UserCategoryProfile,
    family: EntityFamily,
    *,
    config: CategoryPolicyConfig = DEFAULT_POLICY_CONFIG,
) -> tuple[EntityPreferenceRecord, ...]:
    """Return deterministically ordered preferences passing support and strength."""
    return tuple(
        preference
        for preference in profile.preferences[family]
        if _qualifies(preference, config)
    )


def _qualifies(
    preference: EntityPreferenceRecord, config: CategoryPolicyConfig
) -> bool:
    """Apply director-specific or broad-family minimum evidence gates."""
    if preference.affinity <= 0:
        return False
    if preference.family == "director":
        return (
            preference.support_count >= config.director_minimum_support
            and preference.positive_count >= config.director_minimum_positive
            and preference.high_rating_count >= config.director_minimum_high
            and preference.negative_count / preference.support_count
            <= config.director_maximum_negative_fraction
        )
    return (
        preference.support_count >= config.broad_minimum_support
        and preference.positive_fraction >= config.broad_minimum_positive_fraction
        and preference.high_rating_count >= config.broad_minimum_high
    )
