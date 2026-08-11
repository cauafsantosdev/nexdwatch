import numpy as np

from app.domain.candidates import RecommendationCandidate
from experiments.ranker.artifacts import RankerUserContext, build_fold_artifacts
from experiments.ranker.catalog import IndexedRelation, RankerCatalog
from experiments.ranker.config import (
    AFFINITY_FEATURES,
    ENTITY_FAMILIES,
    FEATURE_NAMES,
    PREFERENCE_FEATURES,
)
from experiments.ranker.features import (
    _affinity_features,
    _catalog_features,
    _film_aggregate_features,
    _preference_features,
    _source_features,
    build_feature_matrix,
    build_personalized_feature_matrix,
    build_user_feature_profile,
)


def _relation(rows: list[list[int]]) -> IndexedRelation:
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    values: list[int] = []
    for row, entities in enumerate(rows):
        values.extend(entities)
        indptr[row + 1] = len(values)
    return IndexedRelation(indptr, np.asarray(values, dtype=np.int32))


def _fixtures():
    film_ids = np.asarray([10, 20, 30, 40], dtype=np.int64)
    artifacts = build_fold_artifacts(
        (
            RankerUserContext(
                1,
                np.asarray([0, 1], dtype=np.int64),
                np.asarray([9, 3], dtype=np.int64),
            ),
        ),
        film_ids,
        seed=42,
        svd_components=1,
    )
    relations = {family: _relation([[0], [1], [0], []]) for family in ENTITY_FAMILIES}
    catalog = RankerCatalog(
        film_ids=film_ids,
        id_to_row={10: 0, 20: 1, 30: 2, 40: 3},
        years=np.asarray([2000, 2010, 2020, np.nan], dtype=np.float32),
        runtimes=np.asarray([100, 110, 120, np.nan], dtype=np.float32),
        relations=relations,
    )
    return artifacts, catalog


def test_fold_aggregates_and_artifact_contributors_are_train_only() -> None:
    artifacts, _ = _fixtures()
    assert artifacts.contributing_user_ids == frozenset({1})
    assert artifacts.contributing_interaction_count == 2
    assert artifacts.rating_counts.tolist() == [1, 1, 0, 0]
    assert artifacts.positive_counts.tolist() == [1, 0, 0, 0]


def test_affinity_formula_source_features_and_missing_metadata() -> None:
    artifacts, catalog = _fixtures()
    profile = build_user_feature_profile(
        np.asarray([0], dtype=np.int64),
        np.asarray([9], dtype=np.int64),
        artifacts,
        catalog,
        history_depth_thresholds=(1.0, 2.0, 3.0),
    )
    candidates = (
        RecommendationCandidate(
            film_id=30,
            svd_score=0.4,
            svd_rank=2,
            retrieved_by_svd=True,
        ),
        RecommendationCandidate(
            film_id=40,
            popularity_score=1,
            popularity_rank=4,
            retrieved_by_popularity=True,
        ),
    )
    matrix = build_feature_matrix(
        candidates,
        profile,
        artifacts,
        catalog,
        svd_profile_available=True,
    )
    position = {name: index for index, name in enumerate(FEATURE_NAMES)}
    assert np.isclose(matrix[0, position["genre_affinity_mean"]], 2 / 3)
    assert matrix[0, position["genre_support_sum"]] == 1
    assert matrix[0, position["genre_matched_fraction"]] == 1
    assert np.isnan(matrix[1, position["svd_score"]])
    assert matrix[1, position["retrieved_by_popularity"]] == 1
    assert matrix[1, position["year_missing"]] == 1
    assert matrix[1, position["runtime_missing"]] == 1
    assert np.isnan(matrix[1, position["year_distance"]])
    assert np.isnan(matrix[1, position["runtime_distance"]])


def test_vectorized_feature_builder_matches_reference_row_formulas() -> None:
    artifacts, catalog = _fixtures()
    profile = build_user_feature_profile(
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([9, 3], dtype=np.int64),
        artifacts,
        catalog,
        history_depth_thresholds=(1.0, 2.0, 3.0),
    )
    candidates = (
        RecommendationCandidate(
            film_id=30,
            svd_score=0.4,
            svd_rank=2,
            popularity_score=8,
            popularity_rank=3,
            retrieved_by_svd=True,
            retrieved_by_popularity=True,
        ),
        RecommendationCandidate(
            film_id=40,
            popularity_score=1,
            popularity_rank=4,
            retrieved_by_popularity=True,
        ),
    )
    actual = build_feature_matrix(
        candidates,
        profile,
        artifacts,
        catalog,
        svd_profile_available=True,
    )
    expected = []
    for candidate in candidates:
        film_row = catalog.id_to_row[candidate.film_id]
        values = _source_features(candidate, svd_profile_available=True)
        values.update(profile.base_features)
        values.update(_film_aggregate_features(film_row, artifacts))
        values.update(_catalog_features(film_row, catalog))
        values.update(_affinity_features(film_row, profile, catalog))
        values.update(_preference_features(film_row, profile, artifacts, catalog))
        expected.append([values[name] for name in FEATURE_NAMES])
    np.testing.assert_allclose(
        actual,
        np.asarray(expected, dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )
    personalized_indexes = [
        FEATURE_NAMES.index(name) for name in (*AFFINITY_FEATURES, *PREFERENCE_FEATURES)
    ]
    personalized = build_personalized_feature_matrix(
        candidates, profile, artifacts, catalog
    )
    np.testing.assert_allclose(
        personalized,
        actual[:, personalized_indexes],
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )
