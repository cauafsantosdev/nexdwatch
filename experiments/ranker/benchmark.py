"""Orchestrate the isolated strict out-of-user LambdaRank benchmark."""

import hashlib
import json
import logging
import os
import tempfile
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import numpy as np

from app.core.config import Settings, get_settings
from app.ml.historical_interactions import (
    PreparedInteractions,
    build_interaction_splits,
    load_historical_interactions,
)
from experiments.catalog import load_catalog_slug_mapping
from experiments.ranker.artifacts import RankerUserContext, build_fold_artifacts
from experiments.ranker.catalog import RankerCatalog, load_ranker_catalog
from experiments.ranker.config import (
    FEATURE_NAMES,
    LIGHTGBM_PARAMETERS,
    NEGATIVE_RATING_THRESHOLD,
    POPULARITY_DEPTH,
    POSITIVE_RATING_THRESHOLD,
    RANKER_PROTOCOL,
    SAMPLED_BENCHMARK_PROTOCOL,
    SVD_DEPTH,
    TRAINING_HOLDOUT_LIMIT,
)
from experiments.ranker.dataset import (
    PartitionDataset,
    build_partition_dataset,
    build_user_examples,
    load_partition_dataset,
    write_partition_dataset,
)
from experiments.ranker.protocol import (
    RankerTrainingHoldouts,
    build_user_folds,
    select_ranker_training_holdouts,
)
from experiments.ranker.training import (
    build_fold_report,
    persist_models_and_report,
    train_ranker_ablations,
)
from experiments.ranker.uncertainty import paired_user_clustered_comparisons

logger = logging.getLogger(__name__)


def run_benchmark(
    *,
    csv_path: str | Path,
    output_root: str | Path,
    seeds: Sequence[int],
    folds: Sequence[int],
    settings: Settings | None = None,
    ablations: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build, fit, evaluate, and persist requested independent fold rankers."""
    active = settings or get_settings()
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    mapping = load_catalog_slug_mapping(active)
    data = load_historical_interactions(csv_path, mapping)
    catalog = load_ranker_catalog(data.film_ids, active)
    manifest = _manifest(data, catalog, seeds, folds)
    _atomic_json(manifest, root / "manifest.json")
    reports: list[dict[str, Any]] = []
    for seed in seeds:
        logger.info("Preparing strict ranker protocol for seed=%d", seed)
        splits = build_interaction_splits(
            data,
            positive_rating_threshold=POSITIVE_RATING_THRESHOLD,
            negative_rating_threshold=NEGATIVE_RATING_THRESHOLD,
            seed=int(seed),
        )
        assignment = build_user_folds(splits, data.film_ids, seed=int(seed))
        for fold in folds:
            existing_path = root / f"seed={seed}" / f"fold={fold}" / "metrics.json"
            if existing_path.is_file():
                logger.info("Reusing completed seed=%d fold=%d", seed, fold)
                report = json.loads(existing_path.read_text(encoding="utf-8"))
                if report.get("fold_metadata", {}).get("protocol") != RANKER_PROTOCOL:
                    raise ValueError(
                        f"incompatible existing fold report: {existing_path}"
                    )
            else:
                logger.info("Running seed=%d fold=%d", seed, fold)
                report = run_fold(
                    data,
                    splits,
                    catalog,
                    assignment=assignment,
                    seed=int(seed),
                    fold=int(fold),
                    output_root=root,
                    ablations=ablations,
                )
            reports.append(report)
            summary = _aggregate_reports(reports)
            _atomic_json(summary, root / "benchmark_summary.json")
    return _aggregate_reports(reports)


def rerun_persisted_ablations(
    output_root: str | Path,
    *,
    seeds: Sequence[int],
    folds: Sequence[int],
    names: tuple[str, ...],
) -> dict[str, Any]:
    """Refit selected ablations from persisted numeric matrices only."""
    root = Path(output_root)
    reports: list[dict[str, Any]] = []
    for seed in seeds:
        for fold in folds:
            fold_root = root / f"seed={seed}" / f"fold={fold}"
            datasets = {
                partition: load_partition_dataset(fold_root, partition)
                for partition in ("train", "validation", "test")
            }
            models = train_ranker_ablations(
                datasets["train"],
                datasets["validation"],
                datasets["test"],
                seed=int(seed),
                names=names,
            )
            update = build_fold_report(models, datasets["validation"], datasets["test"])
            report_path = fold_root / "metrics.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            for field in (
                "lightgbm_version",
                "checkpoint_metric",
                "reported_eval_at",
            ):
                report[field] = update[field]
            for model in models:
                report["models"][model.name] = update["models"][model.name]
                model.model.booster_.save_model(str(fold_root / f"{model.name}.txt"))
            _atomic_json(report, report_path)
            reports.append(report)
    summary = _aggregate_reports(reports)
    _atomic_json(summary, root / "benchmark_summary.json")
    return summary


def run_fold(
    data: PreparedInteractions,
    splits: tuple,
    catalog: RankerCatalog,
    *,
    assignment,
    seed: int,
    fold: int,
    output_root: Path,
    ablations: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Run one fold while asserting the strict train-user artifact boundary."""
    fold_started = time.perf_counter()
    training_users, validation_users, test_users = assignment.partitions(fold)
    if (
        training_users & validation_users
        or training_users & test_users
        or validation_users & test_users
    ):
        raise RuntimeError("ranker user partitions overlap")
    holdouts: dict[int, RankerTrainingHoldouts] = {}
    split_by_user = {split.cohort_user_id: split for split in splits}
    for user_id in training_users:
        holdouts[user_id] = select_ranker_training_holdouts(
            split_by_user[user_id], seed=seed
        )
    contexts = tuple(
        RankerUserContext(
            user_id=user_id,
            item_rows=holdouts[user_id].context_item_rows,
            rating_buckets=holdouts[user_id].context_rating_buckets,
        )
        for user_id in sorted(training_users)
    )
    artifact_started = time.perf_counter()
    logger.info("Building train-only artifacts for seed=%d fold=%d", seed, fold)
    artifacts = build_fold_artifacts(contexts, data.film_ids, seed=seed)
    artifact_seconds = time.perf_counter() - artifact_started
    if artifacts.contributing_user_ids != frozenset(training_users):
        raise RuntimeError("non-training user leaked into fold global artifacts")
    validation_target_strata = {
        split.cohort_user_id: (
            str(artifacts.popularity_strata[split.validation_target])
            if split.validation_target is not None
            else "NONE"
        )
        for split in splits
    }
    test_target_strata = {
        split.cohort_user_id: (
            str(artifacts.popularity_strata[split.test_target])
            if split.test_target is not None
            else "NONE"
        )
        for split in splits
    }
    train_target_strata = {split.cohort_user_id: "NONE" for split in splits}
    train_depths = np.asarray(
        [len(holdouts[user_id].context_item_rows) for user_id in training_users],
        dtype=np.float64,
    )
    depth_thresholds = tuple(
        float(value) for value in np.quantile(train_depths, (0.25, 0.5, 0.75))
    )
    examples = {
        "train": build_user_examples(
            splits,
            data.film_ids,
            training_users,
            "train",
            holdouts,
            train_target_strata,
        ),
        "validation": build_user_examples(
            splits,
            data.film_ids,
            validation_users,
            "validation",
            holdouts,
            validation_target_strata,
        ),
        "test": build_user_examples(
            splits,
            data.film_ids,
            test_users,
            "test",
            holdouts,
            test_target_strata,
        ),
    }
    fold_root = output_root / f"seed={seed}" / f"fold={fold}"
    datasets: dict[str, PartitionDataset] = {}
    dataset_metadata: dict[str, Any] = {}
    dataset_started = time.perf_counter()
    for partition in ("train", "validation", "test"):
        logger.info("Building %s rows for seed=%d fold=%d", partition, seed, fold)
        dataset = build_partition_dataset(
            examples[partition],
            artifacts,
            catalog,
            partition=partition,
            seed=seed,
            fold=fold,
            history_depth_thresholds=depth_thresholds,
        )
        datasets[partition] = dataset
        dataset_metadata[partition] = write_partition_dataset(
            dataset,
            fold_root,
            seed=seed,
            fold=fold,
            partition=partition,
        )
    dataset_seconds = time.perf_counter() - dataset_started
    _assert_canonical_targets_untouched(splits, training_users, holdouts)
    logger.info("Training ranker ablations for seed=%d fold=%d", seed, fold)
    models = train_ranker_ablations(
        datasets["train"],
        datasets["validation"],
        datasets["test"],
        seed=seed,
        names=ablations,
    )
    report = build_fold_report(models, datasets["validation"], datasets["test"])
    holdout_counts = [len(holdouts[user_id].item_rows) for user_id in training_users]
    report["fold_metadata"] = {
        "protocol": RANKER_PROTOCOL,
        "seed": seed,
        "fold": fold,
        "users": {
            "train": len(training_users),
            "validation": len(validation_users),
            "test": len(test_users),
        },
        "train_user_ids_hash": _hash_values(sorted(training_users)),
        "validation_user_ids_hash": _hash_values(sorted(validation_users)),
        "test_user_ids_hash": _hash_values(sorted(test_users)),
        "artifact_contributor_count": len(artifacts.contributing_user_ids),
        "artifact_interaction_count": artifacts.contributing_interaction_count,
        "strict_artifact_boundary_verified": True,
        "history_depth_thresholds": depth_thresholds,
        "training_holdouts": {
            "limit": TRAINING_HOLDOUT_LIMIT,
            "selected_total": sum(holdout_counts),
            "per_user_mean": fmean(holdout_counts) if holdout_counts else 0.0,
            "distribution": dict(Counter(holdout_counts)),
        },
        "dataset": dataset_metadata,
        "runtime_seconds": {
            "fold_artifacts": artifact_seconds,
            "dataset_build": dataset_seconds,
            "model_training": sum(model.runtime_seconds for model in models),
            "total": time.perf_counter() - fold_started,
        },
    }
    persist_models_and_report(models, report, fold_root)
    return report


def _assert_canonical_targets_untouched(
    splits: tuple,
    training_users: set[int],
    holdouts: dict[int, RankerTrainingHoldouts],
) -> None:
    for split in splits:
        if split.cohort_user_id not in training_users:
            continue
        hidden = set(holdouts[split.cohort_user_id].item_rows.tolist())
        canonical = {split.validation_target, split.test_target} - {None}
        if hidden & canonical:
            raise RuntimeError("canonical evaluation target entered ranker fitting")


def _manifest(
    data: PreparedInteractions,
    catalog: RankerCatalog,
    seeds: Sequence[int],
    folds: Sequence[int],
) -> dict[str, Any]:
    policies = {
        "candidate": f"svd={SVD_DEPTH}+popularity={POPULARITY_DEPTH};dedup-no-refill",
        "fold": "test=k;validation=(k+1)%5;train=remaining-3",
        "holdout": "canonical=exact_holdout_v2;train-positive-up-to-8",
        "training_sampler": "cap=512;25/25/20/12.5/10/7.5;deterministic-fill",
        "validation_inventory": "complete-deduplicated-candidate-inventory",
        "test_inventory": "complete-deduplicated-candidate-inventory",
        "checkpoint": "full-pool-global-ndcg@20;candidate-misses-zero",
    }
    return {
        "protocol": RANKER_PROTOCOL,
        "supersedes_sampled_evaluation_protocol": SAMPLED_BENCHMARK_PROTOCOL,
        "seeds": list(seeds),
        "folds": list(folds),
        "source_summary": asdict(data.summary),
        "feature_count": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "feature_schema_hash": _hash_values(FEATURE_NAMES),
        "catalog_mapping_hash": _hash_values(catalog.film_ids.tolist()),
        "policies": policies,
        "policy_hash": _hash_values(
            [f"{key}:{value}" for key, value in policies.items()]
        ),
        "lightgbm_parameters": LIGHTGBM_PARAMETERS,
    }


def _aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        return {"completed_models": 0}
    aggregate: dict[str, Any] = {
        "completed_models": len(reports),
        "models": {},
        "baselines": {},
    }
    model_names = sorted(reports[0]["models"])
    for model_name in model_names:
        aggregate["models"][model_name] = _aggregate_strategy(
            [report["models"][model_name]["test"] for report in reports]
        )
    baseline_names = sorted(reports[0]["baselines"]["test"])
    for baseline_name in baseline_names:
        aggregate["baselines"][baseline_name] = _aggregate_strategy(
            [report["baselines"]["test"][baseline_name] for report in reports]
        )
    aggregate["full_vs_baselines"] = paired_user_clustered_comparisons(reports)
    if "full" in model_names:
        aggregate["full_segments"] = _aggregate_segments(reports)
        aggregate["full_feature_importance_stability"] = _importance_stability(reports)
        best_iterations = [
            int(report["models"]["full"]["best_iteration"]) for report in reports
        ]
        aggregate["checkpoint_stability"] = {
            "best_iterations": [
                {
                    "seed": int(report["fold_metadata"]["seed"]),
                    "fold": int(report["fold_metadata"]["fold"]),
                    "best_iteration": int(report["models"]["full"]["best_iteration"]),
                }
                for report in reports
            ],
            "iteration_1_count": sum(value == 1 for value in best_iterations),
            "previous_sampled_validation_iteration_1_count": 8,
        }
    aggregate["evaluation_inventory"] = "full_candidate_inventory"
    aggregate["candidate_inventory"] = _aggregate_candidate_inventory(reports)
    aggregate["total_runtime_seconds"] = sum(
        report["fold_metadata"]["runtime_seconds"]["total"] for report in reports
    )
    return aggregate


def _aggregate_candidate_inventory(reports: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for partition in ("train", "validation", "test"):
        metadata = [report["fold_metadata"]["dataset"][partition] for report in reports]
        queries = [query for value in metadata for query in value["all_queries"]]
        full_counts = [int(query["full_candidate_count"]) for query in queries]
        ranked_counts = [int(query["ranking_row_count"]) for query in queries]
        output[partition] = {
            "group_construction": metadata[0]["group_construction"],
            "query_count": len(queries),
            "eligible_target_count": sum(
                query["designated_target_id"] is not None for query in queries
            ),
            "persisted_row_count": sum(int(value["row_count"]) for value in metadata),
            "deduplicated_candidate_count_before_protocol_exclusion": (
                _distribution(full_counts)
            ),
            "ranked_candidate_count": _distribution(ranked_counts),
        }
    return output


def _distribution(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values) if values else 0,
        "mean": fmean(values) if values else 0.0,
        "max": max(values) if values else 0,
    }


def _aggregate_segments(reports: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    dimensions = reports[0]["models"]["full"]["test"]["segments"]
    for dimension in dimensions:
        output[dimension] = {}
        all_group_names = sorted(
            {
                name
                for report in reports
                for name in report["models"]["full"]["test"]["segments"].get(
                    dimension, {}
                )
            }
        )
        for name in all_group_names:
            values = [
                report["models"]["full"]["test"]["segments"][dimension][name]
                for report in reports
                if name
                in report["models"]["full"]["test"]["segments"].get(dimension, {})
            ]
            eligible = sum(int(value["eligible_targets"]) for value in values)
            retrieved = sum(
                int(value["candidate_retrieved_targets"]) for value in values
            )
            output[dimension][name] = {
                "eligible_targets": eligible,
                "candidate_retrieved_targets": retrieved,
                "candidate_recall": retrieved / eligible if eligible else 0.0,
                "candidate_recall_fold_mean": _mean_std(
                    [float(value["candidate_recall"]) for value in values]
                ),
                "candidate_conditional": {
                    metric: _mean_std(
                        [
                            float(value["candidate_conditional"][metric])
                            for value in values
                        ]
                    )
                    for metric in ("recall_at_10", "recall_at_20", "ndcg_at_20")
                },
                "global": {
                    metric: _mean_std(
                        [float(value["global"][metric]) for value in values]
                    )
                    for metric in ("recall_at_10", "recall_at_20", "ndcg_at_20")
                },
            }
    return output


def _importance_stability(reports: list[dict[str, Any]]) -> dict[str, Any]:
    by_feature: dict[str, list[tuple[float, int, int]]] = {}
    for report in reports:
        importance = report["models"]["full"]["feature_importance"]
        ordered = sorted(
            importance, key=lambda name: importance[name]["gain"], reverse=True
        )
        ranks = {name: rank for rank, name in enumerate(ordered, start=1)}
        total_gain = sum(float(values["gain"]) for values in importance.values()) or 1.0
        for feature, values in importance.items():
            by_feature.setdefault(feature, []).append(
                (
                    float(values["gain"]) / total_gain,
                    int(values["split"]),
                    ranks[feature],
                )
            )
    return {
        feature: {
            "normalized_gain_mean": fmean(value[0] for value in values),
            "normalized_gain_std": (
                pstdev(value[0] for value in values) if len(values) > 1 else 0.0
            ),
            "split_count_mean": fmean(value[1] for value in values),
            "rank_mean": fmean(value[2] for value in values),
            "rank_std": (
                pstdev(value[2] for value in values) if len(values) > 1 else 0.0
            ),
            "nonzero_gain_folds": sum(value[0] > 0 for value in values),
        }
        for feature, values in sorted(by_feature.items())
    }


def _aggregate_strategy(values: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for view in ("candidate_conditional", "global"):
        output[view] = {}
        for metric in (
            "recall_at_10",
            "recall_at_20",
            "recall_at_50",
            "ndcg_at_10",
            "ndcg_at_20",
            "mrr_at_10",
            "graded_ndcg_at_10",
            "graded_ndcg_at_20",
        ):
            samples = [float(value[view][metric]) for value in values]
            output[view][metric] = {
                "mean": fmean(samples),
                "std": pstdev(samples) if len(samples) > 1 else 0.0,
            }
    recalls = [float(value["candidate_recall"]) for value in values]
    output["candidate_recall"] = {
        "mean": fmean(recalls),
        "std": pstdev(recalls) if len(recalls) > 1 else 0.0,
    }
    return output


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": fmean(values) if values else 0.0,
        "std": pstdev(values) if len(values) > 1 else 0.0,
    }


def _hash_values(values: Sequence[Any]) -> str:
    encoded = json.dumps(list(values), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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
