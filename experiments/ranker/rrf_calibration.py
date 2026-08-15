"""Bounded validation-only calibration of full-pool reciprocal-rank fusion."""

import json
import logging
import os
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import numpy as np
import typer
from numpy.typing import NDArray

from app.core.config import Settings, get_settings
from app.domain.candidates import RecommendationCandidate
from app.ml.historical_interactions import (
    PreparedInteractions,
    UserSplit,
    build_interaction_splits,
    load_historical_interactions,
)
from app.ml.svd_profiles import build_svd_profile
from experiments.catalog import load_catalog_slug_mapping
from experiments.ranker.artifacts import (
    FoldArtifacts,
    RankerUserContext,
    build_fold_artifacts,
)
from experiments.ranker.candidates import generate_fold_candidates, target_source
from experiments.ranker.config import (
    NEGATIVE_RATING_THRESHOLD,
    POPULARITY_DEPTH,
    POSITIVE_RATING_THRESHOLD,
    RANKER_PROTOCOL,
    RANKER_SEEDS,
    SVD_DEPTH,
)
from experiments.ranker.dataset import RankerUserExample, build_user_examples
from experiments.ranker.metrics import target_rank, target_rank_metrics
from experiments.ranker.protocol import (
    build_user_folds,
    select_ranker_training_holdouts,
)
from experiments.ranker.uncertainty import user_clustered_ndcg_at_20_comparison

logger = logging.getLogger(__name__)
app = typer.Typer(no_args_is_help=True)

RRF_CALIBRATION_PROTOCOL = f"{RANKER_PROTOCOL}_rrf_calibration_v1"
RRF_WEIGHTS = (
    (50, 50),
    (60, 40),
    (70, 30),
    (80, 20),
    (40, 60),
    (30, 70),
    (20, 80),
)
RRF_K_VALUES = (20, 60, 100, 200)
VALIDATION_PLATEAU_ABSOLUTE_NDCG = 0.001


@dataclass(frozen=True, slots=True, order=True)
class RRFConfiguration:
    """One normalized deterministic weighted-RRF configuration."""

    svd_weight: int
    popularity_weight: int
    k: int

    @property
    def key(self) -> str:
        """Return a stable human-readable calibration-grid identifier."""
        return f"svd={self.svd_weight}_popularity={self.popularity_weight}_k={self.k}"


RRF_CONFIGURATIONS = tuple(
    RRFConfiguration(svd_weight, popularity_weight, k)
    for svd_weight, popularity_weight in RRF_WEIGHTS
    for k in RRF_K_VALUES
)
DEFAULT_RRF_CONFIGURATION = RRFConfiguration(50, 50, 60)


@dataclass(frozen=True, slots=True)
class RankObservation:
    """One eligible canonical target and its rank under one strategy."""

    user_id: int
    target_rank: int | None
    target_retrieved: bool
    target_stratum: str
    history_depth: str
    target_source: str
    candidate_count: int


@dataclass(frozen=True, slots=True)
class PreparedCalibrationFold:
    """Train-only fold artifacts retained until validation freezes test choices."""

    seed: int
    fold: int
    artifacts: FoldArtifacts
    test_user_ids: frozenset[int]
    history_depth_thresholds: tuple[float, float, float]
    selected_configuration: RRFConfiguration
    validation_reports: dict[RRFConfiguration, dict[str, Any]]


def run_rrf_calibration(
    *,
    csv_path: str | Path,
    output_path: str | Path,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Select on validation, then evaluate frozen RRF choices once on test."""
    started = time.perf_counter()
    active = settings or get_settings()
    mapping = load_catalog_slug_mapping(active)
    data = load_historical_interactions(csv_path, mapping)
    prepared_folds: list[PreparedCalibrationFold] = []
    validation_by_configuration: dict[RRFConfiguration, list[dict[str, Any]]] = {
        configuration: [] for configuration in RRF_CONFIGURATIONS
    }
    splits_by_seed: dict[int, tuple[UserSplit, ...]] = {}

    for seed in RANKER_SEEDS:
        logger.info("Preparing RRF validation folds for seed=%d", seed)
        splits = build_interaction_splits(
            data,
            positive_rating_threshold=POSITIVE_RATING_THRESHOLD,
            negative_rating_threshold=NEGATIVE_RATING_THRESHOLD,
            seed=seed,
        )
        splits_by_seed[seed] = splits
        assignment = build_user_folds(splits, data.film_ids, seed=seed)
        split_by_user = {split.cohort_user_id: split for split in splits}
        for fold in range(5):
            logger.info("Calibrating validation seed=%d fold=%d", seed, fold)
            training_users, validation_users, test_users = assignment.partitions(fold)
            artifacts, thresholds = _build_artifacts(
                data, split_by_user, training_users, seed=seed
            )
            validation_examples = _evaluation_examples(
                splits,
                data,
                validation_users,
                "validation",
                artifacts,
            )
            observations = _evaluate_rrf_configurations(
                validation_examples,
                artifacts,
                RRF_CONFIGURATIONS,
                history_depth_thresholds=thresholds,
            )
            reports = {
                configuration: _strategy_report(values)
                for configuration, values in observations.items()
            }
            selected = select_validation_configuration(reports)
            for configuration, report in reports.items():
                validation_by_configuration[configuration].append(report)
            prepared_folds.append(
                PreparedCalibrationFold(
                    seed=seed,
                    fold=fold,
                    artifacts=artifacts,
                    test_user_ids=frozenset(test_users),
                    history_depth_thresholds=thresholds,
                    selected_configuration=selected,
                    validation_reports=reports,
                )
            )

    aggregate_validation = _aggregate_validation_grid(validation_by_configuration)
    recommended, plateau = recommend_fixed_configuration(aggregate_validation)
    test_observations: dict[str, list[RankObservation]] = {
        "validation_selected_rrf": [],
        "recommended_fixed_rrf": [],
        "conventional_rrf": [],
        "positive_weighted_svd": [],
        "popularity": [],
        "svd_mean_pooling": [],
    }
    fold_test_reports: list[dict[str, Any]] = []
    for prepared in prepared_folds:
        logger.info(
            "Evaluating frozen test seed=%d fold=%d", prepared.seed, prepared.fold
        )
        test_examples = _evaluation_examples(
            splits_by_seed[prepared.seed],
            data,
            set(prepared.test_user_ids),
            "test",
            prepared.artifacts,
        )
        strategies = _evaluate_test_fold(prepared, test_examples, recommended)
        for name, values in strategies.items():
            test_observations[name].extend(values)
        fold_test_reports.append(
            {
                "seed": prepared.seed,
                "fold": prepared.fold,
                "validation_selected_configuration": asdict(
                    prepared.selected_configuration
                ),
                "validation_selection_metrics": prepared.validation_reports[
                    prepared.selected_configuration
                ],
                "test": {
                    name: _strategy_report(values)
                    for name, values in strategies.items()
                },
            }
        )

    test_reports = {
        name: _strategy_report(values) for name, values in test_observations.items()
    }
    aggregate_test = _aggregate_test_folds(fold_test_reports)
    report: dict[str, Any] = {
        "protocol": RRF_CALIBRATION_PROTOCOL,
        "base_protocol": RANKER_PROTOCOL,
        "candidate_policy": {
            "svd_depth": SVD_DEPTH,
            "popularity_depth": POPULARITY_DEPTH,
            "deduplication": "deterministic_no_refill",
            "watched_films_excluded": True,
            "target_injection": False,
        },
        "grid": [asdict(configuration) for configuration in RRF_CONFIGURATIONS],
        "selection": {
            "primary_metric": "validation_global_ndcg_at_20",
            "tie_breaking": [
                "validation_global_recall_at_20",
                "validation_candidate_conditional_ndcg_at_20",
                "default_nearness",
            ],
            "folds": [
                {
                    "seed": prepared.seed,
                    "fold": prepared.fold,
                    "configuration": asdict(prepared.selected_configuration),
                }
                for prepared in prepared_folds
            ],
            "win_counts": dict(
                Counter(
                    prepared.selected_configuration.key for prepared in prepared_folds
                )
            ),
        },
        "aggregate_validation_grid": aggregate_validation,
        "fixed_recommendation": {
            "configuration": asdict(recommended),
            "absolute_ndcg_plateau_threshold": VALIDATION_PLATEAU_ABSOLUTE_NDCG,
            "plateau_configurations": [asdict(value) for value in plateau],
            "selected_without_test_metrics": True,
        },
        "fold_test_reports": fold_test_reports,
        "test": test_reports,
        "aggregate_test_folds": aggregate_test,
        "test_lifts": {
            "validation_selected_rrf_vs_conventional_rrf": _primary_lift(
                aggregate_test["validation_selected_rrf"],
                aggregate_test["conventional_rrf"],
            ),
            "recommended_fixed_rrf_vs_conventional_rrf": _primary_lift(
                aggregate_test["recommended_fixed_rrf"],
                aggregate_test["conventional_rrf"],
            ),
            "validation_selected_rrf_vs_positive_weighted_svd": _primary_lift(
                aggregate_test["validation_selected_rrf"],
                aggregate_test["positive_weighted_svd"],
            ),
        },
        "uncertainty": {
            "validation_selected_vs_conventional_rrf": _clustered_comparison(
                test_observations["validation_selected_rrf"],
                test_observations["conventional_rrf"],
            ),
            "validation_selected_vs_positive_weighted_svd": _clustered_comparison(
                test_observations["validation_selected_rrf"],
                test_observations["positive_weighted_svd"],
            ),
            "recommended_fixed_vs_conventional_rrf": _clustered_comparison(
                test_observations["recommended_fixed_rrf"],
                test_observations["conventional_rrf"],
            ),
            "recommended_fixed_vs_positive_weighted_svd": _clustered_comparison(
                test_observations["recommended_fixed_rrf"],
                test_observations["positive_weighted_svd"],
            ),
        },
        "recommended_fixed_segments": _segment_reports(
            test_observations["recommended_fixed_rrf"]
        ),
        "candidate_inventory": _candidate_inventory(
            test_observations["recommended_fixed_rrf"]
        ),
        "source_summary": asdict(data.summary),
        "runtime_seconds": time.perf_counter() - started,
        "feature_matrices_constructed": False,
        "lightgbm_used": False,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(report, destination)
    return report


def select_validation_configuration(
    reports: dict[RRFConfiguration, dict[str, Any]],
) -> RRFConfiguration:
    """Select one fold configuration without consulting test users."""
    if set(reports) != set(RRF_CONFIGURATIONS):
        raise ValueError("validation selection requires the complete 28-point grid")
    return max(
        reports,
        key=lambda configuration: (
            reports[configuration]["global"]["ndcg_at_20"],
            reports[configuration]["global"]["recall_at_20"],
            reports[configuration]["candidate_conditional"]["ndcg_at_20"],
            *_inverse_simplicity_key(configuration),
        ),
    )


def recommend_fixed_configuration(
    aggregate_validation: dict[str, dict[str, Any]],
) -> tuple[RRFConfiguration, tuple[RRFConfiguration, ...]]:
    """Prefer the default-nearest member of a predeclared validation plateau."""
    mean_by_configuration = {
        configuration: float(
            aggregate_validation[configuration.key]["global"]["ndcg_at_20"]["mean"]
        )
        for configuration in RRF_CONFIGURATIONS
    }
    best = max(mean_by_configuration.values())
    plateau = tuple(
        configuration
        for configuration in RRF_CONFIGURATIONS
        if best - mean_by_configuration[configuration]
        <= VALIDATION_PLATEAU_ABSOLUTE_NDCG
    )
    return min(plateau, key=_simplicity_key), plateau


def _build_artifacts(
    data: PreparedInteractions,
    split_by_user: dict[int, UserSplit],
    training_users: set[int],
    *,
    seed: int,
) -> tuple[FoldArtifacts, tuple[float, float, float]]:
    holdouts = {
        user_id: select_ranker_training_holdouts(split_by_user[user_id], seed=seed)
        for user_id in training_users
    }
    contexts = tuple(
        RankerUserContext(
            user_id=user_id,
            item_rows=holdouts[user_id].context_item_rows,
            rating_buckets=holdouts[user_id].context_rating_buckets,
        )
        for user_id in sorted(training_users)
    )
    artifacts = build_fold_artifacts(contexts, data.film_ids, seed=seed)
    if artifacts.contributing_user_ids != frozenset(training_users):
        raise RuntimeError("non-training user leaked into RRF fold artifacts")
    depths = np.asarray(
        [len(holdouts[user_id].context_item_rows) for user_id in training_users],
        dtype=np.float64,
    )
    thresholds = tuple(float(value) for value in np.quantile(depths, (0.25, 0.5, 0.75)))
    return artifacts, thresholds


def _evaluation_examples(
    splits: tuple[UserSplit, ...],
    data: PreparedInteractions,
    user_ids: set[int],
    partition: str,
    artifacts: FoldArtifacts,
) -> tuple[RankerUserExample, ...]:
    target_rows = {
        split.cohort_user_id: (
            split.validation_target if partition == "validation" else split.test_target
        )
        for split in splits
    }
    strata = {
        user_id: (str(artifacts.popularity_strata[row]) if row is not None else "NONE")
        for user_id, row in target_rows.items()
    }
    return build_user_examples(
        splits,
        data.film_ids,
        user_ids,
        partition,
        {},
        strata,
    )


def _evaluate_rrf_configurations(
    examples: tuple[RankerUserExample, ...],
    artifacts: FoldArtifacts,
    configurations: tuple[RRFConfiguration, ...],
    *,
    history_depth_thresholds: tuple[float, float, float],
) -> dict[RRFConfiguration, list[RankObservation]]:
    observations = {configuration: [] for configuration in configurations}
    for example in examples:
        if example.designated_target_id is None:
            continue
        candidates = _full_candidates(example, artifacts)
        film_ids, svd_ranks, popularity_ranks = _candidate_rank_arrays(candidates)
        retrieved = bool(np.any(film_ids == example.designated_target_id))
        source = target_source(
            next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.film_id == example.designated_target_id
                ),
                None,
            )
        )
        terms = {
            k: (
                np.where(np.isfinite(svd_ranks), 1.0 / (k + svd_ranks), 0.0),
                np.where(
                    np.isfinite(popularity_ranks),
                    1.0 / (k + popularity_ranks),
                    0.0,
                ),
            )
            for k in RRF_K_VALUES
        }
        common = {
            "user_id": example.user_id,
            "target_retrieved": retrieved,
            "target_stratum": example.target_stratum,
            "history_depth": _history_depth(
                len(example.context_item_rows), history_depth_thresholds
            ),
            "target_source": source,
            "candidate_count": len(candidates),
        }
        for configuration in configurations:
            svd_term, popularity_term = terms[configuration.k]
            scores = (
                configuration.svd_weight * svd_term
                + configuration.popularity_weight * popularity_term
            )
            observations[configuration].append(
                RankObservation(
                    target_rank=target_rank(
                        film_ids, scores, example.designated_target_id
                    ),
                    **common,
                )
            )
    return observations


def _evaluate_test_fold(
    prepared: PreparedCalibrationFold,
    test_examples: tuple[RankerUserExample, ...],
    recommended: RRFConfiguration,
) -> dict[str, list[RankObservation]]:
    aliases = {
        "validation_selected_rrf": prepared.selected_configuration,
        "recommended_fixed_rrf": recommended,
        "conventional_rrf": DEFAULT_RRF_CONFIGURATION,
    }
    output = {
        **{name: [] for name in aliases},
        "positive_weighted_svd": [],
        "popularity": [],
        "svd_mean_pooling": [],
    }
    id_to_row = {
        int(film_id): row for row, film_id in enumerate(prepared.artifacts.film_ids)
    }
    for example in test_examples:
        if example.designated_target_id is None:
            continue
        candidates = _full_candidates(example, prepared.artifacts)
        film_ids, svd_ranks, popularity_ranks = _candidate_rank_arrays(candidates)
        candidate_by_id = {candidate.film_id: candidate for candidate in candidates}
        retrieved = example.designated_target_id in candidate_by_id
        common = {
            "user_id": example.user_id,
            "target_retrieved": retrieved,
            "target_stratum": example.target_stratum,
            "history_depth": _history_depth(
                len(example.context_item_rows), prepared.history_depth_thresholds
            ),
            "target_source": target_source(
                candidate_by_id.get(example.designated_target_id)
            ),
            "candidate_count": len(candidates),
        }
        unique_configurations = set(aliases.values())
        ranks_by_configuration = {
            configuration: target_rank(
                film_ids,
                _rrf_scores(svd_ranks, popularity_ranks, configuration),
                example.designated_target_id,
            )
            for configuration in unique_configurations
        }
        for name, configuration in aliases.items():
            output[name].append(
                RankObservation(
                    target_rank=ranks_by_configuration[configuration],
                    **common,
                )
            )

        weighted_svd_scores = np.asarray(
            [
                candidate.svd_score if candidate.svd_score is not None else -np.inf
                for candidate in candidates
            ],
            dtype=np.float64,
        )
        popularity_scores = np.where(
            np.isfinite(popularity_ranks), -popularity_ranks, -np.inf
        )
        mean_query = build_svd_profile(
            prepared.artifacts.item_vectors,
            example.context_item_rows,
            example.context_rating_buckets,
            "svd_mean",
        )
        mean_scores = (
            np.asarray(
                [
                    prepared.artifacts.item_vectors[id_to_row[int(film_id)]]
                    @ mean_query
                    for film_id in film_ids
                ],
                dtype=np.float64,
            )
            if mean_query is not None
            else np.full(len(film_ids), -np.inf, dtype=np.float64)
        )
        for name, scores in (
            ("positive_weighted_svd", weighted_svd_scores),
            ("popularity", popularity_scores),
            ("svd_mean_pooling", mean_scores),
        ):
            output[name].append(
                RankObservation(
                    target_rank=target_rank(
                        film_ids, scores, example.designated_target_id
                    ),
                    **common,
                )
            )
    return output


def _full_candidates(
    example: RankerUserExample,
    artifacts: FoldArtifacts,
) -> tuple[RecommendationCandidate, ...]:
    result = generate_fold_candidates(
        example.user_id,
        example.context_item_rows,
        example.context_rating_buckets,
        artifacts,
    )
    return tuple(
        candidate
        for candidate in result.candidates
        if candidate.film_id not in example.forbidden_positive_ids
    )


def _candidate_rank_arrays(
    candidates: tuple[RecommendationCandidate, ...],
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    film_ids = np.fromiter(
        (candidate.film_id for candidate in candidates),
        dtype=np.int64,
        count=len(candidates),
    )
    svd_ranks = np.fromiter(
        (
            candidate.svd_rank if candidate.svd_rank is not None else np.inf
            for candidate in candidates
        ),
        dtype=np.float64,
        count=len(candidates),
    )
    popularity_ranks = np.fromiter(
        (
            candidate.popularity_rank
            if candidate.popularity_rank is not None
            else np.inf
            for candidate in candidates
        ),
        dtype=np.float64,
        count=len(candidates),
    )
    return film_ids, svd_ranks, popularity_ranks


def _rrf_scores(
    svd_ranks: NDArray[np.float64],
    popularity_ranks: NDArray[np.float64],
    configuration: RRFConfiguration,
) -> NDArray[np.float64]:
    return np.where(
        np.isfinite(svd_ranks),
        configuration.svd_weight / (configuration.k + svd_ranks),
        0.0,
    ) + np.where(
        np.isfinite(popularity_ranks),
        configuration.popularity_weight / (configuration.k + popularity_ranks),
        0.0,
    )


def _strategy_report(observations: list[RankObservation]) -> dict[str, Any]:
    global_ranks = [observation.target_rank for observation in observations]
    conditional_ranks = [
        observation.target_rank
        for observation in observations
        if observation.target_retrieved
    ]
    return {
        "eligible_users": len(observations),
        "candidate_retrieved_users": len(conditional_ranks),
        "candidate_recall": (
            len(conditional_ranks) / len(observations) if observations else 0.0
        ),
        "candidate_conditional": target_rank_metrics(conditional_ranks),
        "global": target_rank_metrics(global_ranks),
    }


def _aggregate_validation_grid(
    by_configuration: dict[RRFConfiguration, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    fold_best = [
        max(
            reports[fold]["global"]["ndcg_at_20"]
            for reports in by_configuration.values()
        )
        for fold in range(15)
    ]
    output: dict[str, dict[str, Any]] = {}
    for configuration, reports in by_configuration.items():
        aggregate: dict[str, Any] = {
            "configuration": asdict(configuration),
            "candidate_recall": _mean_std(
                [float(report["candidate_recall"]) for report in reports]
            ),
        }
        for view in ("candidate_conditional", "global"):
            aggregate[view] = {
                metric: _mean_std([float(report[view][metric]) for report in reports])
                for metric in (
                    "recall_at_10",
                    "recall_at_20",
                    "recall_at_50",
                    "ndcg_at_10",
                    "ndcg_at_20",
                    "mrr_at_10",
                )
            }
        ndcg_values = [float(report["global"]["ndcg_at_20"]) for report in reports]
        aggregate["mean_global_ndcg_at_20_regret"] = fmean(
            best - value for best, value in zip(fold_best, ndcg_values, strict=True)
        )
        aggregate["within_0_001_of_fold_best_count"] = int(
            sum(
                best - value <= VALIDATION_PLATEAU_ABSOLUTE_NDCG
                for best, value in zip(fold_best, ndcg_values, strict=True)
            )
        )
        output[configuration.key] = aggregate
    return output


def _segment_reports(
    observations: list[RankObservation],
) -> dict[str, dict[str, dict[str, Any]]]:
    dimensions = {
        "target_stratum": lambda value: value.target_stratum,
        "history_depth": lambda value: value.history_depth,
        "target_source": lambda value: value.target_source,
    }
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for dimension, classifier in dimensions.items():
        grouped: dict[str, list[RankObservation]] = {}
        for observation in observations:
            grouped.setdefault(classifier(observation), []).append(observation)
        output[dimension] = {
            name: _strategy_report(values) for name, values in sorted(grouped.items())
        }
    return output


def _aggregate_test_folds(
    fold_reports: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    strategy_names = tuple(fold_reports[0]["test"]) if fold_reports else ()
    output: dict[str, dict[str, Any]] = {}
    for strategy in strategy_names:
        reports = [fold["test"][strategy] for fold in fold_reports]
        aggregate: dict[str, Any] = {
            "candidate_recall": _mean_std(
                [float(report["candidate_recall"]) for report in reports]
            )
        }
        for view in ("candidate_conditional", "global"):
            aggregate[view] = {
                metric: _mean_std([float(report[view][metric]) for report in reports])
                for metric in (
                    "recall_at_10",
                    "recall_at_20",
                    "recall_at_50",
                    "ndcg_at_10",
                    "ndcg_at_20",
                    "mrr_at_10",
                )
            }
        output[strategy] = aggregate
    return output


def _primary_lift(
    strategy: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, float | None]:
    strategy_value = float(strategy["global"]["ndcg_at_20"]["mean"])
    baseline_value = float(baseline["global"]["ndcg_at_20"]["mean"])
    absolute = strategy_value - baseline_value
    return {
        "global_ndcg_at_20_absolute": absolute,
        "global_ndcg_at_20_relative": (
            absolute / baseline_value if baseline_value else None
        ),
    }


def _clustered_comparison(
    strategy: list[RankObservation],
    baseline: list[RankObservation],
) -> dict[str, Any]:
    return user_clustered_ndcg_at_20_comparison(
        [
            {"user_id": value.user_id, "target_rank": value.target_rank}
            for value in strategy
        ],
        [
            {"user_id": value.user_id, "target_rank": value.target_rank}
            for value in baseline
        ],
    )


def _candidate_inventory(observations: list[RankObservation]) -> dict[str, Any]:
    counts = [value.candidate_count for value in observations]
    return {
        "eligible_targets": len(observations),
        "candidate_retrieved_targets": sum(
            value.target_retrieved for value in observations
        ),
        "ranked_candidates": {
            "min": min(counts) if counts else 0,
            "mean": fmean(counts) if counts else 0.0,
            "max": max(counts) if counts else 0,
        },
    }


def _history_depth(
    count: int,
    thresholds: tuple[float, float, float],
) -> str:
    bucket = int(np.searchsorted(thresholds, count, side="right"))
    if bucket == 0:
        return "bottom_quartile"
    if bucket == 3:
        return "top_quartile"
    return "middle_50_percent"


def _simplicity_key(configuration: RRFConfiguration) -> tuple[int, int, int, int]:
    return (
        int(configuration != DEFAULT_RRF_CONFIGURATION),
        abs(configuration.svd_weight - 50),
        abs(configuration.k - 60),
        configuration.k,
    )


def _inverse_simplicity_key(
    configuration: RRFConfiguration,
) -> tuple[int, int, int, int]:
    return tuple(-value for value in _simplicity_key(configuration))


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": fmean(values) if values else 0.0,
        "std": pstdev(values) if len(values) > 1 else 0.0,
    }


def _atomic_json(payload: object, destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(temporary, destination)
        destination.chmod(0o644)
    finally:
        temporary.unlink(missing_ok=True)


@app.command()
def run(
    csv_path: Path = typer.Option(..., exists=True, dir_okay=False),  # noqa: B008
    output_path: Path = typer.Option(  # noqa: B008
        Path("notebooks/data/rrf_calibration/full_pool_v2.json")
    ),
) -> None:
    """Run the bounded full-pool RRF calibration."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    report = run_rrf_calibration(csv_path=csv_path, output_path=output_path)
    recommendation = report["fixed_recommendation"]["configuration"]
    typer.echo(f"Recommended fixed RRF: {recommendation}")


if __name__ == "__main__":
    app()
