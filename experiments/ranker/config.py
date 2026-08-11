"""Frozen protocol constants for the first offline LambdaRank benchmark."""

from typing import Final

RANKER_PROTOCOL: Final = "strict_out_of_user_lambdarank_full_pool_v2"
SAMPLED_BENCHMARK_PROTOCOL: Final = "strict_out_of_user_lambdarank_v1"
RANKER_SEEDS: Final = (42, 43, 44)
USER_FOLD_COUNT: Final = 5
TRAINING_HOLDOUT_LIMIT: Final = 8
GROUP_ROW_CAP: Final = 512
CATALOG_REFERENCE_YEAR: Final = 2026
SVD_DEPTH: Final = 2000
POPULARITY_DEPTH: Final = 2000
POSITIVE_RATING_THRESHOLD: Final = 3.5
NEGATIVE_RATING_THRESHOLD: Final = 2.5
ENTITY_FAMILIES: Final = (
    "genre",
    "director",
    "actor",
    "theme",
    "country",
    "language",
    "studio",
    "decade",
)
ENTITY_SMOOTHING: Final = {
    "genre": 2.0,
    "director": 5.0,
    "actor": 5.0,
    "theme": 2.0,
    "country": 2.0,
    "language": 2.0,
    "studio": 5.0,
    "decade": 2.0,
}
SAMPLING_STRATA: Final = (
    "positive",
    "both_source",
    "svd_only",
    "popularity_only",
    "additional_top_500",
    "source_tail",
    "uniform_remainder",
    "deterministic_fill",
    "full_pool_candidate",
)

SOURCE_FEATURES: Final = (
    "svd_score",
    "svd_rank",
    "svd_reciprocal_rank",
    "svd_rank_fraction",
    "log_svd_rank",
    "popularity_positive_count",
    "log_popularity_count",
    "popularity_rank",
    "popularity_reciprocal_rank",
    "popularity_rank_fraction",
    "retrieved_by_svd",
    "retrieved_by_popularity",
    "retrieved_by_both",
    "source_count",
    "source_rank_gap",
    "svd_profile_available",
)
USER_FEATURES: Final = (
    "watched_count",
    "rated_count",
    "log_rated_count",
    "mean_rating",
    "rating_variance",
    "rating_std",
    "positive_count",
    "neutral_count",
    "negative_count",
    "positive_fraction",
    "negative_fraction",
    "high_rating_fraction",
    "rating_range",
    "history_depth_bucket",
    "unrated_watch_fraction",
)
FILM_AGGREGATE_FEATURES: Final = (
    "cohort_rating_count",
    "cohort_positive_count",
    "cohort_negative_count",
    "cohort_mean_rating",
    "cohort_rating_variance",
    "cohort_smoothed_rating",
    "popularity_percentile",
    "is_head",
    "is_mid",
    "is_tail",
)
CATALOG_FEATURES: Final = (
    "release_year",
    "release_decade",
    "film_age",
    "runtime_minutes",
    "log_runtime",
    "year_missing",
    "runtime_missing",
    "genre_count",
    "director_count",
    "actor_count",
    "theme_count",
    "country_count",
    "language_count",
    "studio_count",
)
AFFINITY_FEATURES: Final = tuple(
    f"{family}_{stat}"
    for family in ENTITY_FAMILIES
    for stat in (
        "affinity_mean",
        "affinity_max",
        "affinity_min",
        "support_sum",
        "matched_count",
        "matched_fraction",
    )
)
PREFERENCE_FEATURES: Final = (
    "positive_year_mean",
    "year_distance",
    "year_z_distance",
    "positive_runtime_mean",
    "runtime_distance",
    "runtime_z_distance",
    "preferred_popularity_percentile",
    "popularity_preference_distance",
    "head_positive_fraction",
    "mid_positive_fraction",
    "tail_positive_fraction",
    "candidate_stratum_preference",
)
FEATURE_NAMES: Final = (
    *SOURCE_FEATURES,
    *USER_FEATURES,
    *FILM_AGGREGATE_FEATURES,
    *CATALOG_FEATURES,
    *AFFINITY_FEATURES,
    *PREFERENCE_FEATURES,
)

LABEL_GAIN: Final = (0, 1, 3, 7)
LIGHTGBM_PARAMETERS: Final = {
    "objective": "lambdarank",
    "metric": "None",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "n_estimators": 1000,
    "num_leaves": 31,
    "max_depth": 8,
    "min_child_samples": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "max_bin": 255,
    "lambdarank_truncation_level": 30,
    "lambdarank_norm": True,
    "sigmoid": 1.0,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}
