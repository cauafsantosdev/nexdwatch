#!/usr/bin/env python3

import asyncio
import json
import resource
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from app.db.loaders.load_films import load_films_data
from app.db.loaders.load_logs import load_logs_data
from app.ml.faiss_index import rebuild_faiss_index
from app.ml.train import train_svd_model

app = typer.Typer()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
FILMS_CSV_PATH = str(DATA_DIR / "films_data.csv")
LOGS_CSV_PATH = str(DATA_DIR / "users_data.csv")
CANDIDATE_REPORT_PATH = DATA_DIR / "analysis" / "candidate_analysis.json"
POPULARITY_ARTIFACT_PATH = DATA_DIR / "candidates" / "popularity.json"
CATEGORY_POLICY_REPORT_PATH = DATA_DIR / "analysis" / "category_policy.json"


@app.command()
def load_films() -> None:
    """Loads films from CSV into the database."""
    typer.echo(f"Starting to load films from: {FILMS_CSV_PATH}")
    asyncio.run(load_films_data(FILMS_CSV_PATH))
    typer.echo("Films loaded successfully.")


@app.command()
def load_logs() -> None:
    """Loads user logs from CSV into the database."""
    typer.echo(f"Starting to load logs from: {LOGS_CSV_PATH}")
    asyncio.run(load_logs_data(LOGS_CSV_PATH))
    typer.echo("Logs loaded successfully.")


async def _load_all_async() -> None:
    """Async helper to load both films and logs"""
    typer.echo(f"Starting to load films from: {FILMS_CSV_PATH}")
    await load_films_data(FILMS_CSV_PATH)

    typer.echo(f"Starting to load logs from: {LOGS_CSV_PATH}")
    await load_logs_data(LOGS_CSV_PATH)


@app.command()
def load_all() -> None:
    """Triggers both load_films_data and load_logs_data"""
    asyncio.run(_load_all_async())
    typer.echo("All data loaded successfully.")


@app.command()
def train() -> None:
    """Trains the SVD model and saves artifacts."""
    typer.echo("Starting training pipeline...")
    try:
        result = train_svd_model()
        if result is None:
            raise RuntimeError("training did not produce recommendation artifacts")
        typer.echo(
            "Training finished successfully: "
            f"{result.film_count} films, dimension {result.dimension}, "
            f"index {result.output_path}"
        )
    except Exception as exc:
        typer.echo(f"Training failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("build-index")
def build_index() -> None:
    """Rebuild the exact FAISS index from existing SVD artifacts."""
    try:
        result = rebuild_faiss_index(DATA_DIR)
    except Exception as exc:
        typer.echo(f"Index rebuild failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"Indexed {result.film_count} films with dimension {result.dimension}: "
        f"{result.output_path}"
    )


@app.command("build-popularity")
def build_popularity(
    output_path: Annotated[
        Path, typer.Option(help="Controlled-popularity artifact path.")
    ] = POPULARITY_ARTIFACT_PATH,
) -> None:
    """Build the deterministic controlled-cohort popularity artifact."""
    from app.ml.catalog import load_catalog_slug_mapping
    from app.ml.historical_interactions import load_historical_interactions
    from app.ml.popularity import (
        build_popularity_artifact,
        write_popularity_artifact,
    )

    typer.echo(f"Loading controlled interactions from: {LOGS_CSV_PATH}")
    try:
        mapping = load_catalog_slug_mapping()
        data = load_historical_interactions(LOGS_CSV_PATH, mapping)
        artifact = build_popularity_artifact(data)
        destination = write_popularity_artifact(artifact, output_path)
    except Exception as exc:
        typer.echo(f"Popularity artifact build failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Popularity artifact built: films={artifact.film_count} "
        f"threshold={artifact.rating_threshold:.1f} path={destination}"
    )


@app.command("preview-candidates")
def preview_candidates(
    user_id: Annotated[int, typer.Argument(min=1)],
    sample_size: Annotated[
        int, typer.Option(min=1, max=50, help="Number of candidates to display.")
    ] = 10,
) -> None:
    """Preview the finalized internal candidate inventory without persisting it."""
    from app.services.candidate_generation_service import CandidateGenerationService

    service = CandidateGenerationService()
    if not service.load_artifacts():
        typer.echo("Candidate artifacts are unavailable.", err=True)
        raise typer.Exit(code=1)
    try:
        result, titles, watched_overlap = asyncio.run(
            _preview_candidates_async(service, user_id, sample_size)
        )
    except Exception as exc:
        typer.echo(f"Candidate preview failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        service.unload_artifacts()

    svd_count = sum(candidate.retrieved_by_svd for candidate in result.candidates)
    popularity_count = sum(
        candidate.retrieved_by_popularity for candidate in result.candidates
    )
    overlap = sum(candidate.source_count == 2 for candidate in result.candidates)
    typer.echo(
        f"user_id={user_id} nominal_budget={result.nominal_budget} "
        f"unique_candidates={result.unique_candidate_count} "
        f"svd_depth={result.svd_depth} popularity_depth={result.popularity_depth} "
        f"svd={svd_count} popularity={popularity_count} source_overlap={overlap} "
        f"watched_overlap={watched_overlap} "
        f"svd_profile_available={result.svd_profile_available}"
    )
    typer.echo("Sample:")
    for candidate in result.candidates[:sample_size]:
        typer.echo(
            f"  film_id={candidate.film_id} "
            f"title={titles.get(candidate.film_id, '<unresolved>')!r} "
            f"svd_rank={candidate.svd_rank} svd_score={candidate.svd_score} "
            f"popularity_rank={candidate.popularity_rank} "
            f"popularity_score={candidate.popularity_score}"
        )


async def _preview_candidates_async(
    service: Any,
    user_id: int,
    sample_size: int,
) -> tuple[Any, dict[int, str], int]:
    """Generate candidates and resolve preview titles in one async lifecycle."""
    from app.core.database import SessionLocal
    from app.repositories.films import FilmRepository
    from app.repositories.interactions import InteractionRepository

    result = await service.generate(user_id)
    sample_ids = [candidate.film_id for candidate in result.candidates[:sample_size]]
    async with SessionLocal() as session:
        films = await FilmRepository(session).get_by_ids(sample_ids)
        watched_ids = set(
            await InteractionRepository(session).get_watched_film_ids(result.user_id)
        )
    watched_overlap = len(
        watched_ids.intersection(candidate.film_id for candidate in result.candidates)
    )
    return result, {film.id: film.title for film in films}, watched_overlap


@app.command("preview-categories")
def preview_categories(
    user_id: Annotated[int, typer.Argument(min=1)] = 3953,
) -> None:
    """Preview the internal categorized policy without changing public serving."""
    from app.services.categorized_recommendation_service import (
        CategorizedRecommendationService,
    )

    service = CategorizedRecommendationService()
    started = time.perf_counter()
    try:
        result, load_ms, request_ms = asyncio.run(
            _preview_categories_async(service, user_id)
        )
    except Exception as exc:
        typer.echo(f"Category preview failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        service.unload_resources()
    typer.echo(
        f"user_id={user_id} categories={len(result.categories)} "
        f"load_ms={load_ms:.2f} request_ms={request_ms:.2f} "
        f"total_ms={(time.perf_counter() - started) * 1000:.2f} "
        "peak_process_mib="
        f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.2f}"
    )
    for position, category in enumerate(result.categories, start=1):
        typer.echo(
            f"\n{position}. key={category.key} title={category.title!r} "
            f"size={len(category.items)} role={category.role}"
        )
        for item in category.items:
            typer.echo(
                f"  film_id={item.film_id} title={item.title!r} "
                f"reason={item.reason.code} rrf_rank={item.rrf_rank} "
                f"stratum={item.popularity_stratum} source={item.source_membership}"
            )
    typer.echo("\nPolicy diagnostics:")
    typer.echo(json.dumps(result.diagnostics, indent=2, sort_keys=True))


async def _preview_categories_async(
    service: Any, user_id: int
) -> tuple[Any, float, float]:
    load_started = time.perf_counter()
    if not await service.load_resources():
        raise RuntimeError("categorized recommendation resources are unavailable")
    load_ms = (time.perf_counter() - load_started) * 1000
    request_started = time.perf_counter()
    result = await service.recommend(user_id)
    request_ms = (time.perf_counter() - request_started) * 1000
    return result, load_ms, request_ms


@app.command("evaluate-categories")
def evaluate_categories(
    seeds: str = typer.Option("42,43,44", help="Comma-separated strict-fold seeds."),
    folds: str = typer.Option("0,1,2,3,4", help="Comma-separated test folds."),
    report_path: Annotated[
        Path, typer.Option(help="Non-production JSON report path.")
    ] = CATEGORY_POLICY_REPORT_PATH,
) -> None:
    """Evaluate categorized policy using context-only strict held-out folds."""
    from experiments.category_policy.evaluate import run_category_policy_evaluation

    try:
        parsed_seeds = _parse_non_negative_integers(seeds)
        parsed_folds = _parse_non_negative_integers(folds)
        if any(fold > 4 for fold in parsed_folds):
            raise ValueError
    except ValueError as exc:
        typer.echo(
            "Seeds must be non-negative integers and folds must be in 0..4.",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Evaluating category policy on seeds={parsed_seeds} folds={parsed_folds}..."
    )
    try:
        report = run_category_policy_evaluation(
            csv_path=LOGS_CSV_PATH,
            output_path=report_path,
            seeds=parsed_seeds,
            folds=parsed_folds,
        )
    except Exception as exc:
        typer.echo(f"Category-policy evaluation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    portfolio = report["portfolio"]
    typer.echo(
        "Evaluation complete: "
        f"users={report['evaluated_user_appearances']} "
        f"mean_categories={portfolio['categories_per_user']['mean']:.3f} "
        f"mean_unique_films={portfolio['unique_films_per_response']['mean']:.3f}"
    )
    typer.echo(f"Report: {report_path}")


def _parse_non_negative_integers(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(","))
    if not parsed or any(item < 0 for item in parsed):
        raise ValueError
    return parsed


@app.command("analyze-candidates")
def analyze_candidates(
    seeds: str = typer.Option("42,43,44", help="Comma-separated random seeds."),
    report_path: Annotated[
        Path, typer.Option(help="Non-production JSON report path.")
    ] = CANDIDATE_REPORT_PATH,
) -> None:
    """Analyze offline candidate recall, coverage, overlap, and source unions."""
    from app.ml.candidate_analysis import CANDIDATE_CUTOFFS, run_candidate_analysis

    try:
        parsed_seeds = tuple(int(value.strip()) for value in seeds.split(","))
        if not parsed_seeds or any(seed < 0 for seed in parsed_seeds):
            raise ValueError
    except ValueError as exc:
        typer.echo(
            "Analysis seeds must be comma-separated non-negative integers.",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Analyzing candidate generation on seeds={','.join(map(str, parsed_seeds))}"
    )
    try:
        report = run_candidate_analysis(
            parsed_seeds,
            csv_path=LOGS_CSV_PATH,
            report_path=report_path,
        )
    except Exception as exc:
        typer.echo(f"Candidate analysis failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("\nAggregate global recall (mean ± population std)")
    for name, values in report["aggregate_strategy_metrics"].items():
        coverage = _format_mean_std(values["profile_coverage_percentage"])
        recalls = " ".join(
            f"R@{cutoff}={_format_mean_std(values['recall_at'][cutoff])}"
            for cutoff in CANDIDATE_CUTOFFS
        )
        typer.echo(f"{name}: {recalls} coverage={coverage}")
    typer.echo(f"\nBest SVD profile: {report['best_svd_profile']}")
    typer.echo("Shortlisted SVD + popularity hybrids:")
    hybrids = report["svd_popularity_hybrids"]
    for budget, shortlist in hybrids["shortlist_by_budget"].items():
        values = shortlist["selected"]
        typer.echo(
            f"  nominal_budget={budget} "
            f"configuration={shortlist['selected_configuration']} "
            f"recall={_format_mean_std(values['recall'])} "
            "mean_candidates="
            f"{values['mean_deduplicated_candidates']['mean']:.2f} "
            f"grid_location={shortlist['winner_grid_location']}"
        )
        for label, ratio in hybrids["allocation_sweep"][budget].items():
            typer.echo(
                f"    {label}: recall={_format_mean_std(ratio['recall'])} "
                "mean_candidates="
                f"{ratio['mean_deduplicated_candidates']['mean']:.2f}"
            )
    recommendation = report["recommended_candidate_strategy"]
    typer.echo(
        "Recommended source mix: "
        f"{recommendation['classification']} "
        f"budget={recommendation['nominal_budget']} "
        f"{recommendation['allocations']}"
    )
    typer.echo(f"Report: {report_path}")


def _format_mean_std(summary: Mapping[str, float]) -> str:
    return f"{summary['mean']:.6f}±{summary['population_std']:.6f}"


if __name__ == "__main__":
    app()
