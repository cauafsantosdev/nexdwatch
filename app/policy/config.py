"""Centralized transparent defaults for categorized recommendation policy V1."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CategoryPolicyConfig:
    """Policy thresholds are reviewable defaults, not optimized model parameters."""

    rrf_k: int = 60
    maximum_categories: int = 10
    top_picks_maximum: int = 20
    default_maximum: int = 20

    positive_rating: float = 3.5
    anchor_rating: float = 4.5
    anchor_fallback_rating: float = 4.0
    high_rating: float = 4.5
    negative_rating: float = 2.5

    director_smoothing: float = 5.0
    broad_entity_smoothing: float = 4.0
    director_minimum_support: int = 2
    director_minimum_positive: int = 2
    director_minimum_high: int = 1
    director_maximum_negative_fraction: float = 0.25
    broad_minimum_support: int = 5
    broad_minimum_positive_fraction: float = 0.70
    broad_minimum_high: int = 2

    hidden_minimum: int = 12
    hidden_mid_svd_rank: int = 500
    hidden_tail_svd_rank: int = 250
    hidden_minimum_indexed_positives: int = 3

    brazilian_minimum: int = 8
    brazilian_country_names: tuple[str, ...] = ("brazil", "brasil")

    anchor_minimum: int = 8
    anchor_maximum: int = 20
    anchor_similarity_threshold: float = 0.0
    anchor_similarity_batch_size: int = 256

    directors_minimum: int = 8
    director_pool_size: int = 15
    director_film_cap: int = 3
    genre_minimum: int = 12
    decade_minimum: int = 12

    world_minimum: int = 12
    world_supported_country_pool: int = 5
    world_country_cap: int = 5
    world_discovery_svd_rank: int = 500
    english_language_names: tuple[str, ...] = ("english",)
    english_core_country_names: tuple[str, ...] = (
        "usa",
        "uk",
        "canada",
        "australia",
        "new zealand",
        "ireland",
        "english",
    )
    metadata_none_names: tuple[str, ...] = ("no spoken language",)

    outside_minimum: int = 12
    outside_svd_rank: int = 500
    outside_rrf_rank: int = 1000
    outside_familiar_entities_per_family: int = 1
    outside_minimum_familiar_families: int = 2
    outside_exclude_head: bool = True

    classic_year_boundary: int = 1969
    classic_minimum: int = 8

    category_overlap_threshold: float = 0.70
    maximum_film_appearances: int = 2
    top_picks_reserved: int = 10
    generic_director_cap: int = 3
    generic_decade_cap: int = 6
    generic_genre_cap: int = 8

    sparse_rated_threshold: int = 5
    deep_watched_threshold: int = 2000


DEFAULT_POLICY_CONFIG = CategoryPolicyConfig()
