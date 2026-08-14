"""Tests for recommendation artifact CLI commands."""

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from typer.testing import CliRunner

import manage
from app.domain.candidates import CandidateGenerationResult, RecommendationCandidate
from app.domain.maintenance import (
    RetrainingDecision,
    RetrainingDeltas,
    RetrainingReason,
    TrainingStatistics,
)
from app.ml.faiss_index import FaissIndexBuildResult


def test_rollback_dry_run_uses_shared_exact_target_selector(
    monkeypatch,
) -> None:
    from app.ml import model_registry

    target = SimpleNamespace(model_version="20300103T010203Z-00000003")
    selector = Mock(return_value=target)
    monkeypatch.setattr(model_registry, "select_rollback_target", selector)

    result = CliRunner().invoke(manage.app, ["rollback-model", "--dry-run"])

    assert result.exit_code == 0
    assert result.output.strip() == (
        "rollback_target=20300103T010203Z-00000003 dry_run=true"
    )
    selector.assert_called_once()


def test_training_status_and_retrain_dry_run_report_legacy_bootstrap(
    monkeypatch,
) -> None:
    from app.ml import model_lifecycle, model_registry

    stats = TrainingStatistics(datetime.now(UTC), 10, 20, (1, 2))
    decision = RetrainingDecision(
        should_retrain=True,
        reasons=(RetrainingReason.LEGACY_MODEL_BOOTSTRAP,),
        current_stats=stats,
        trained_stats=None,
        deltas=RetrainingDeltas(0, 0, 0, 0.0),
    )
    evaluator = Mock(return_value=decision)
    monkeypatch.setattr(model_lifecycle, "evaluate_retraining", evaluator)
    monkeypatch.setattr(model_registry, "read_current_version", lambda _: None)

    status = CliRunner().invoke(manage.app, ["training-status"])
    dry_run = CliRunner().invoke(manage.app, ["retrain", "--dry-run"])

    assert status.exit_code == 0
    assert dry_run.exit_code == 0
    assert "LEGACY_MODEL_BOOTSTRAP" in status.output
    assert "LEGACY_MODEL_BOOTSTRAP" in dry_run.output
    assert evaluator.call_count == 2


def test_build_index_command_reports_artifact_details(monkeypatch, tmp_path) -> None:
    result = FaissIndexBuildResult(
        film_count=123,
        dimension=32,
        output_path=tmp_path / "retrieval.faiss",
    )
    monkeypatch.setattr(manage, "rebuild_faiss_index", lambda _: result)

    command_result = CliRunner().invoke(manage.app, ["build-index"])

    assert command_result.exit_code == 0
    assert "123 films" in command_result.output
    assert "dimension 32" in command_result.output
    assert str(result.output_path) in command_result.output


def test_build_index_command_returns_nonzero_on_failure(monkeypatch) -> None:
    def fail(_: Path) -> None:
        raise ValueError("invalid artifacts")

    monkeypatch.setattr(manage, "rebuild_faiss_index", fail)

    command_result = CliRunner().invoke(manage.app, ["build-index"])

    assert command_result.exit_code == 1
    assert "Index rebuild failed: invalid artifacts" in command_result.output


def test_manage_does_not_expose_neural_experiment_commands() -> None:
    result = CliRunner().invoke(manage.app, ["--help"])

    assert result.exit_code == 0
    assert "train-ncf" not in result.output
    assert "benchmark-ncf" not in result.output
    assert "build-ncf-index" not in result.output


def test_analyze_candidates_reports_recall_hybrids_and_output_path(
    monkeypatch, tmp_path
) -> None:
    from app.ml import candidate_analysis

    summary = {"mean": 0.2, "population_std": 0.01}
    report = {
        "aggregate_strategy_metrics": {
            "popularity": {
                "recall_at": {
                    cutoff: summary
                    for cutoff in (
                        10,
                        50,
                        100,
                        250,
                        500,
                        750,
                        1000,
                        1500,
                        2000,
                        2500,
                        3000,
                        4000,
                        5000,
                    )
                },
                "profile_coverage_percentage": {
                    "mean": 100.0,
                    "population_std": 0.0,
                },
            }
        },
        "best_svd_profile": "svd_positive_weighted",
        "svd_popularity_hybrids": {
            "shortlist_by_budget": {
                4000: {
                    "selected_configuration": "2000_weighted_2000_popularity",
                    "winner_grid_location": "interior",
                    "selected": {
                        "recall": summary,
                        "mean_deduplicated_candidates": {
                            "mean": 3591.0,
                            "population_std": 2.0,
                        },
                    },
                }
            },
            "allocation_sweep": {
                4000: {
                    "2000_weighted_2000_popularity": {
                        "recall": summary,
                        "mean_deduplicated_candidates": {
                            "mean": 3591.0,
                            "population_std": 2.0,
                        },
                    }
                }
            },
        },
        "neural_retrieval_status": "research-only",
        "recommended_candidate_strategy": {
            "classification": "svd_popularity_hybrid",
            "nominal_budget": 4000,
            "allocations": {"svd": 2000, "popularity": 2000},
        },
    }
    monkeypatch.setattr(
        candidate_analysis, "run_candidate_analysis", lambda *_, **__: report
    )
    output_path = tmp_path / "candidate-analysis.json"

    result = CliRunner().invoke(
        manage.app,
        [
            "analyze-candidates",
            "--seeds",
            "42,43,44",
            "--report-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert "Best SVD profile: svd_positive_weighted" in result.output
    assert "2000_weighted_2000_popularity" in result.output
    assert str(output_path) in result.output


def test_build_popularity_command_reports_artifact(monkeypatch, tmp_path) -> None:
    from app.ml import catalog, historical_interactions, popularity

    artifact = SimpleNamespace(film_count=12, rating_threshold=3.5)
    monkeypatch.setattr(catalog, "load_catalog_slug_mapping", lambda: {"a": 1})
    monkeypatch.setattr(
        historical_interactions,
        "load_historical_interactions",
        lambda *_: object(),
    )
    monkeypatch.setattr(popularity, "build_popularity_artifact", lambda _: artifact)
    monkeypatch.setattr(
        popularity,
        "write_popularity_artifact",
        lambda _, path: path,
    )
    path = tmp_path / "popularity.json"

    result = CliRunner().invoke(
        manage.app, ["build-popularity", "--output-path", str(path)]
    )

    assert result.exit_code == 0
    assert "films=12" in result.output
    assert str(path) in result.output


def test_preview_candidates_reports_provenance(monkeypatch) -> None:
    from app.services import candidate_generation_service

    class FakeService:
        def load_artifacts(self) -> bool:
            return True

        def unload_artifacts(self) -> None:
            return None

    async def preview(*_: object):
        return (
            CandidateGenerationResult(
                user_id=3953,
                candidates=(
                    RecommendationCandidate(
                        film_id=7,
                        svd_score=0.5,
                        svd_rank=1,
                        popularity_score=12,
                        popularity_rank=3,
                        retrieved_by_svd=True,
                        retrieved_by_popularity=True,
                    ),
                ),
                nominal_budget=4000,
                svd_depth=2000,
                popularity_depth=2000,
                svd_profile_available=True,
            ),
            {7: "Film Seven"},
            0,
        )

    monkeypatch.setattr(
        candidate_generation_service,
        "CandidateGenerationService",
        FakeService,
    )
    monkeypatch.setattr(manage, "_preview_candidates_async", preview)

    result = CliRunner().invoke(manage.app, ["preview-candidates", "3953"])

    assert result.exit_code == 0
    assert "unique_candidates=1" in result.output
    assert "source_overlap=1" in result.output
    assert "watched_overlap=0" in result.output
    assert "Film Seven" in result.output
