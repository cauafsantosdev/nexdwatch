"""CPU LightGBM LambdaRank fitting, ablations, and diagnostics."""

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from numpy.typing import NDArray

from experiments.ranker.config import (
    FEATURE_NAMES,
    LABEL_GAIN,
    LIGHTGBM_PARAMETERS,
)
from experiments.ranker.dataset import PartitionDataset
from experiments.ranker.features import feature_group_columns
from experiments.ranker.metrics import baseline_reports, evaluate_ranking_scores


@dataclass(frozen=True, slots=True)
class TrainedRanker:
    """One fitted ablation with predictions and auditable diagnostics."""

    name: str
    model: lgb.LGBMRanker
    feature_names: tuple[str, ...]
    validation_scores: NDArray[np.float64]
    test_scores: NDArray[np.float64]
    best_iteration: int
    training_history: dict[str, dict[str, list[float]]]
    runtime_seconds: float


def train_ranker_ablations(
    train: PartitionDataset,
    validation: PartitionDataset,
    test: PartitionDataset,
    *,
    seed: int,
    names: tuple[str, ...] | None = None,
) -> tuple[TrainedRanker, ...]:
    """Fit the frozen ablation matrix with NDCG@20-only early stopping."""
    if not len(train.labels) or not len(validation.labels):
        raise ValueError("LambdaRank requires non-empty train and validation groups")
    groups = feature_group_columns()
    selected_names = names or tuple(groups)
    trained: list[TrainedRanker] = []
    for name in selected_names:
        columns = groups[name]
        train_x = _ablation_matrix(train, name, columns)
        validation_x = _ablation_matrix(validation, name, columns)
        test_x = _ablation_matrix(test, name, columns)
        feature_names = tuple(FEATURE_NAMES[index] for index in columns)
        history: dict[str, dict[str, list[float]]] = {}
        started = time.perf_counter()
        model = lgb.LGBMRanker(
            **LIGHTGBM_PARAMETERS,
            random_state=seed,
            label_gain=list(LABEL_GAIN),
            n_jobs=-1,
        )
        model.fit(
            train_x,
            train.labels,
            group=train.group_sizes,
            eval_X=validation_x,
            eval_y=validation.labels,
            eval_group=[validation.group_sizes],
            eval_metric=_full_pool_global_ndcg_at_20(
                validation.film_ids,
                np.asarray(
                    [
                        query.designated_target_id is not None
                        for query in validation.queries
                    ],
                    dtype=np.bool_,
                ),
            ),
            eval_names=["validation"],
            callbacks=[
                lgb.early_stopping(100, first_metric_only=True, verbose=False),
                lgb.record_evaluation(history),
            ],
        )
        best_iteration = int(
            model.best_iteration_ or LIGHTGBM_PARAMETERS["n_estimators"]
        )
        trained.append(
            TrainedRanker(
                name=name,
                model=model,
                feature_names=feature_names,
                validation_scores=model.predict(
                    validation_x, num_iteration=best_iteration
                ),
                test_scores=model.predict(test_x, num_iteration=best_iteration),
                best_iteration=best_iteration,
                training_history=history,
                runtime_seconds=time.perf_counter() - started,
            )
        )
    return tuple(trained)


def _full_pool_global_ndcg_at_20(
    film_ids: NDArray[np.int64],
    eligible_queries: NDArray[np.bool_],
):
    """Create the exact full-pool metric, including the protocol tie-break."""

    def metric(
        labels: NDArray[np.floating],
        scores: NDArray[np.floating],
        _weights: NDArray[np.floating] | None,
        group_sizes: NDArray[np.integer],
    ) -> tuple[str, float, bool]:
        values: list[float] = []
        stop = 0
        for query_index, size in enumerate(group_sizes):
            start, stop = stop, stop + int(size)
            if not eligible_queries[query_index]:
                continue
            group_labels = labels[start:stop]
            if not np.any(group_labels > 0):
                values.append(0.0)
                continue
            order = np.lexsort((film_ids[start:stop], -scores[start:stop]))
            target_ranks = np.flatnonzero(group_labels[order] > 0) + 1
            rank = int(target_ranks.min())
            values.append(float(1.0 / np.log2(rank + 1)) if rank <= 20 else 0.0)
        value = float(np.mean(values)) if values else 0.0
        return "full_pool_global_ndcg_at_20", value, True

    return metric


def build_fold_report(
    models: tuple[TrainedRanker, ...],
    validation: PartitionDataset,
    test: PartitionDataset,
) -> dict[str, Any]:
    """Evaluate models/baselines and record non-causal feature diagnostics."""
    model_reports: dict[str, Any] = {}
    for trained in models:
        booster = trained.model.booster_
        split = booster.feature_importance(importance_type="split")
        gain = booster.feature_importance(importance_type="gain")
        model_reports[trained.name] = {
            "best_iteration": trained.best_iteration,
            "runtime_seconds": trained.runtime_seconds,
            "feature_count": len(trained.feature_names),
            "feature_names": list(trained.feature_names),
            "validation": evaluate_ranking_scores(
                validation, trained.validation_scores
            ),
            "test": evaluate_ranking_scores(test, trained.test_scores),
            "feature_importance": {
                feature: {"split": int(split[index]), "gain": float(gain[index])}
                for index, feature in enumerate(trained.feature_names)
            },
            "prediction_distributions": _prediction_distributions(
                test, trained.test_scores
            ),
            "training_history": trained.training_history,
        }
    return {
        "lightgbm_version": lgb.__version__,
        "checkpoint_metric": "validation_full_pool_global_ndcg_at_20",
        "reported_eval_at": [10, 20, 50],
        "models": model_reports,
        "baselines": {
            "validation": baseline_reports(validation),
            "test": baseline_reports(test),
        },
        "ordering_examples": _ordering_examples(models, test),
    }


def persist_models_and_report(
    models: tuple[TrainedRanker, ...], report: dict[str, Any], destination: Path
) -> None:
    """Persist research-only text models and report atomically."""
    destination.mkdir(parents=True, exist_ok=True)
    for trained in models:
        trained.model.booster_.save_model(str(destination / f"{trained.name}.txt"))
    _atomic_json(report, destination / "metrics.json")


def _ablation_matrix(
    dataset: PartitionDataset, name: str, columns: tuple[int, ...]
) -> NDArray[np.float32]:
    values = np.ascontiguousarray(dataset.features[:, columns], dtype=np.float32)
    if name != "shuffled_personalization":
        return values
    personalized = feature_group_columns()["personalized_affinity_only"]
    output = values.copy()
    output[:, personalized] = dataset.shuffled_personalized_features
    return output


def _prediction_distributions(
    dataset: PartitionDataset, scores: NDArray[np.float64]
) -> dict[str, dict[str, float | int]]:
    positions = {name: index for index, name in enumerate(FEATURE_NAMES)}
    svd = dataset.features[:, positions["retrieved_by_svd"]]
    popularity = dataset.features[:, positions["retrieved_by_popularity"]]
    source_indexes = {
        "svd_only": (svd == 1) & (popularity == 0),
        "popularity_only": (svd == 0) & (popularity == 1),
        "both": dataset.features[:, positions["retrieved_by_both"]] == 1,
        "HEAD": dataset.features[:, positions["is_head"]] == 1,
        "MID": dataset.features[:, positions["is_mid"]] == 1,
        "TAIL": dataset.features[:, positions["is_tail"]] == 1,
    }
    return {
        name: {
            "count": int(mask.sum()),
            "mean": float(scores[mask].mean()) if mask.any() else 0.0,
            "std": float(scores[mask].std()) if mask.any() else 0.0,
        }
        for name, mask in source_indexes.items()
    }


def _ordering_examples(
    models: tuple[TrainedRanker, ...], dataset: PartitionDataset
) -> dict[str, list[dict[str, int]]]:
    full = next((model for model in models if model.name == "full"), None)
    if full is None:
        return {"improved_vs_rrf": [], "worsened_vs_rrf": []}
    full_results = evaluate_ranking_scores(dataset, full.test_scores)["per_user"]
    rrf_results = evaluate_ranking_scores(dataset, dataset.baseline_scores[:, 2])[
        "per_user"
    ]
    examples: list[dict[str, int]] = []
    for full_row, rrf_row in zip(full_results, rrf_results, strict=True):
        examples.append(
            {
                "user_id": int(full_row["user_id"]),
                "ranker_rank": int(full_row["target_rank"] or 10**9),
                "rrf_rank": int(rrf_row["target_rank"] or 10**9),
            }
        )
    improved = sorted(examples, key=lambda row: row["ranker_rank"] - row["rrf_rank"])
    worsened = sorted(
        examples, key=lambda row: row["ranker_rank"] - row["rrf_rank"], reverse=True
    )
    return {"improved_vs_rrf": improved[:10], "worsened_vs_rrf": worsened[:10]}


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
