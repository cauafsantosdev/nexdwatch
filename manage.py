"""Provides the stable Typer CLI for data, research, and model operations."""

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
CATEGORY_POLICY_REPORT_PATH = DATA_DIR / "analysis" / "category_policy_v1_1.json"
CATEGORY_REFINEMENT_REPORT_PATH = (
    DATA_DIR / "analysis" / "category_policy_refinement.json"
)
CATEGORY_BENCHMARK_REPORT_PATH = (
    DATA_DIR / "analysis" / "category_policy_benchmark.json"
)
CATEGORY_QUALITATIVE_REPORT_PATH = (
    DATA_DIR / "analysis" / "category_policy_qualitative.json"
)
CATEGORY_SERVING_PROFILE_PATH = (
    DATA_DIR / "analysis" / "category_serving_performance.json"
)


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
    from app.ml.historical_interactions import load_historical_interactions
    from app.ml.popularity import (
        build_popularity_artifact,
        write_popularity_artifact,
    )
    from experiments.catalog import load_catalog_slug_mapping

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


@app.command("analyze-category-refinement")
def analyze_category_refinement(
    sample_stride: Annotated[
        int, typer.Option(min=1, help="Deterministic context-user sample stride.")
    ] = 5,
    report_path: Annotated[
        Path, typer.Option(help="Non-production refinement diagnostics path.")
    ] = CATEGORY_REFINEMENT_REPORT_PATH,
) -> None:
    """Compare bounded V1.1 alternatives without using held-out labels."""
    from experiments.category_policy.refinement import run_refinement_analysis

    typer.echo(
        "Analyzing context-only category refinements "
        f"with sample_stride={sample_stride}..."
    )
    try:
        report = run_refinement_analysis(
            csv_path=LOGS_CSV_PATH,
            output_path=report_path,
            sample_stride=sample_stride,
        )
    except Exception as exc:
        typer.echo(f"Category refinement analysis failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Refinement analysis complete: users={report['sampled_users']} "
        f"runtime_seconds={report['runtime_seconds']:.2f}"
    )
    typer.echo(f"Report: {report_path}")


@app.command("benchmark-categories")
def benchmark_categories(
    user_ids: str = typer.Option(
        "3953", help="Comma-separated persisted user IDs for sequential requests."
    ),
    repetitions: Annotated[
        int, typer.Option(min=1, max=10, help="Uninstrumented passes per user.")
    ] = 2,
    report_path: Annotated[
        Path, typer.Option(help="Non-production benchmark report path.")
    ] = CATEGORY_BENCHMARK_REPORT_PATH,
) -> None:
    """Benchmark warm category-policy latency and steady-state memory."""
    from experiments.category_policy.benchmark import run_warm_category_benchmark

    try:
        parsed_user_ids = tuple(int(value.strip()) for value in user_ids.split(","))
        if not parsed_user_ids or any(value <= 0 for value in parsed_user_ids):
            raise ValueError
    except ValueError as exc:
        typer.echo("Benchmark user IDs must be positive integers.", err=True)
        raise typer.Exit(code=1) from exc
    try:
        report = run_warm_category_benchmark(
            user_ids=parsed_user_ids,
            repetitions=repetitions,
            output_path=report_path,
        )
    except Exception as exc:
        typer.echo(f"Category benchmark failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    resources = report["resource_loading"]
    latency = report["warm_request_latency_ms"]
    typer.echo(
        f"artifact_load_ms={resources['artifact_load_ms']:.2f} "
        f"catalog_load_ms={resources['policy_catalog_load_ms']:.2f} "
        "steady_rss_mib="
        f"{resources['steady_state_rss_after_load_mib']:.2f} "
        f"warm_request_mean_ms={latency['mean']:.2f} "
        f"warm_request_median_ms={latency['median']:.2f}"
    )
    typer.echo(f"Report: {report_path}")


@app.command("preview-category-refinement")
def preview_category_refinement(
    user_ids: str = typer.Option(
        "3318,3569,3155,2825,3953,2504,2474,2994,3724",
        help="Comma-separated shallow, medium, and deep persisted user IDs.",
    ),
    report_path: Annotated[
        Path, typer.Option(help="Non-production qualitative review path.")
    ] = CATEGORY_QUALITATIVE_REPORT_PATH,
) -> None:
    """Produce deterministic V1/V1.1 examples without changing public serving."""
    from experiments.category_policy.qualitative import run_qualitative_previews

    try:
        parsed_user_ids = tuple(int(value.strip()) for value in user_ids.split(","))
        if not parsed_user_ids or any(value <= 0 for value in parsed_user_ids):
            raise ValueError
    except ValueError as exc:
        typer.echo("Preview user IDs must be positive integers.", err=True)
        raise typer.Exit(code=1) from exc
    try:
        report = run_qualitative_previews(
            user_ids=parsed_user_ids, output_path=report_path
        )
    except Exception as exc:
        typer.echo(f"Qualitative category preview failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        "Qualitative previews complete: "
        f"users={len(report['user_ids'])} versions=v1,v1_1"
    )
    typer.echo(f"Report: {report_path}")


@app.command("profile-category-serving")
def profile_category_serving(
    user_ids: str = typer.Option(
        "3318,3569,3155,2825,3953,2504,2474,2994,3724",
        help="Comma-separated representative persisted user IDs.",
    ),
    repetitions: Annotated[
        int, typer.Option(min=1, max=10, help="Latency and stage passes per user.")
    ] = 3,
    rss_requests: Annotated[
        int, typer.Option(min=0, max=500, help="Rotating long-run RSS requests.")
    ] = 100,
    report_path: Annotated[
        Path, typer.Option(help="Ignored internal serving-profile report path.")
    ] = CATEGORY_SERVING_PROFILE_PATH,
) -> None:
    """Profile loaded category serving without changing recommendation output."""
    from experiments.category_policy.serving_performance import (
        run_serving_performance_profile,
    )

    try:
        parsed_user_ids = tuple(int(value.strip()) for value in user_ids.split(","))
        report = run_serving_performance_profile(
            user_ids=parsed_user_ids,
            repetitions=repetitions,
            rss_requests=rss_requests,
            output_path=report_path,
        )
    except (TypeError, ValueError) as exc:
        typer.echo(f"Invalid serving-profile arguments: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        typer.echo(f"Category serving profile failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    latency = report["latency_ms"]
    typer.echo(
        f"requests={latency['count']} mean_ms={latency['mean']:.2f} "
        f"median_ms={latency['median']:.2f} p95_ms={latency['p95']:.2f}"
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
    from experiments.retrieval.candidate_analysis import (
        CANDIDATE_CUTOFFS,
        run_candidate_analysis,
    )

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


@app.command("process-film-queue")
def process_film_queue_command(
    batch_size: Annotated[int | None, typer.Option(min=1, max=1_000)] = None,
    dry_run: Annotated[
        bool, typer.Option(help="Inspect the bounded batch only.")
    ] = False,
) -> None:
    """Run one bounded film-ingestion batch using the established processor."""
    from dataclasses import asdict

    from app.core.config import get_settings
    from app.db.loaders.sync_queue import _get_pending_slugs, sync_film_queue

    selected_batch_size = batch_size or get_settings().FILM_QUEUE_BATCH_SIZE
    try:
        if dry_run:
            from app.core.database import SessionLocal

            slugs = asyncio.run(
                _get_pending_slugs(SessionLocal, batch_size=selected_batch_size)
            )
            typer.echo(
                f"dry_run=true selected={len(slugs)} batch_size={selected_batch_size}"
            )
            return
        from app.infrastructure.maintenance_lock import MaintenanceLock

        settings = get_settings()
        lock = MaintenanceLock(
            settings.MAINTENANCE_REDIS_URL,
            key="film-queue",
            ttl_seconds=settings.MAINTENANCE_LOCK_TTL_SECONDS,
        )
        try:
            with lock.held() as acquired:
                if not acquired:
                    raise RuntimeError("film queue maintenance is already active")
                result = asyncio.run(sync_film_queue(batch_size=selected_batch_size))
        finally:
            lock.close()
    except Exception as exc:
        typer.echo(f"Film queue processing failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(asdict(result), default=str, sort_keys=True))


@app.command("refresh-catalog")
def refresh_catalog_command(
    execution_date: Annotated[
        str | None,
        typer.Option(help="UTC policy date in YYYY-MM-DD form; defaults to today."),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option(help="Select films without scraping/writes.")
    ] = False,
) -> None:
    """Refresh recent-film aggregate fields under the January/July policy."""
    from dataclasses import asdict
    from datetime import UTC, date, datetime

    from app.services.catalog_maintenance import refresh_recent_catalog

    try:
        selected_date = (
            date.fromisoformat(execution_date)
            if execution_date
            else datetime.now(UTC).date()
        )
        if dry_run:
            result = asyncio.run(refresh_recent_catalog(selected_date, dry_run=True))
        else:
            from app.core.config import get_settings
            from app.infrastructure.maintenance_lock import MaintenanceLock

            settings = get_settings()
            lock = MaintenanceLock(
                settings.MAINTENANCE_REDIS_URL,
                key=f"catalog-refresh:{selected_date.year}-{selected_date.month:02d}",
                ttl_seconds=settings.MAINTENANCE_LOCK_TTL_SECONDS,
            )
            try:
                with lock.held() as acquired:
                    if not acquired:
                        raise RuntimeError("catalog refresh is already active")
                    result = asyncio.run(refresh_recent_catalog(selected_date))
            finally:
                lock.close()
    except Exception as exc:
        typer.echo(f"Catalog refresh failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(asdict(result), default=str, sort_keys=True))


def _decision_payload(decision: Any) -> dict[str, Any]:
    trained = decision.trained_stats
    return {
        "current_model": None,
        "trained_at": trained.trained_at.isoformat() if trained else None,
        "current_eligible_users": decision.current_stats.eligible_user_count,
        "trained_eligible_users": (
            trained.eligible_user_count
            if trained and trained.eligible_user_count >= 0
            else None
        ),
        "delta_users": decision.deltas.eligible_users,
        "current_rated_film_count": decision.current_stats.model_film_count,
        "new_model_film_delta": decision.deltas.new_model_films,
        "current_rated_interactions": decision.current_stats.rated_interaction_count,
        "model_age_days": decision.deltas.model_age_days,
        "should_retrain": decision.should_retrain,
        "reasons": [reason.value for reason in decision.reasons],
    }


@app.command("training-status")
def training_status_command() -> None:
    """Inspect precise retraining statistics and threshold decisions without writes."""
    from app.core.config import get_settings
    from app.ml.model_lifecycle import evaluate_retraining
    from app.ml.model_registry import read_current_version

    try:
        settings = get_settings()
        decision = evaluate_retraining(settings=settings)
        payload = _decision_payload(decision)
        payload["current_model"] = (
            read_current_version(settings.ARTIFACT_ROOT) or "legacy-flat"
        )
        payload["thresholds"] = {
            "new_eligible_users": settings.NEW_ELIGIBLE_USERS_THRESHOLD,
            "new_model_films": settings.NEW_MODEL_FILMS_THRESHOLD,
            "max_model_age_days": settings.MAX_MODEL_AGE_DAYS,
        }
    except Exception as exc:
        typer.echo(f"Training status failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("retrain")
def retrain_command(
    force: Annotated[bool, typer.Option(help="Bypass operational thresholds.")] = False,
    dry_run: Annotated[
        bool, typer.Option(help="Evaluate without building/promoting.")
    ] = False,
) -> None:
    """Run the validated build-to-atomic-promotion production lifecycle."""
    from app.ml.model_lifecycle import evaluate_retraining, retrain_and_promote

    try:
        if dry_run:
            typer.echo(
                json.dumps(
                    _decision_payload(evaluate_retraining(force=force)), indent=2
                )
            )
            return
        from app.core.config import get_settings
        from app.infrastructure.maintenance_lock import MaintenanceLock

        settings = get_settings()
        lock = MaintenanceLock(
            settings.MAINTENANCE_REDIS_URL,
            key="retraining",
            ttl_seconds=settings.MAINTENANCE_LOCK_TTL_SECONDS,
        )
        try:
            with lock.held() as acquired:
                if not acquired:
                    raise RuntimeError("retraining is already active")
                result = retrain_and_promote(force=force)
        finally:
            lock.close()
    except Exception as exc:
        typer.echo(f"Retraining failed; current model unchanged: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if result.promotion is None:
        typer.echo("Retraining not required.")
        raise typer.Exit(code=0)
    typer.echo(f"Promoted {result.promotion.model_version}; API activation pending")


@app.command("validate-model")
def validate_model_command(
    model_version: Annotated[
        str | None, typer.Option(help="Version; defaults to current.")
    ] = None,
) -> None:
    """Fully validate a versioned model bundle."""
    from app.core.config import get_settings
    from app.ml.model_registry import (
        model_root,
        read_current_version,
        validate_model_bundle,
    )

    settings = get_settings()
    selected = model_version or read_current_version(settings.ARTIFACT_ROOT)
    if selected is None:
        typer.echo("No versioned current model is configured.", err=True)
        raise typer.Exit(code=1)
    try:
        bundle = validate_model_bundle(model_root(settings.ARTIFACT_ROOT) / selected)
    except Exception as exc:
        typer.echo(f"Model validation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"valid=true version={bundle.manifest.model_version} films={bundle.manifest.film_count}"
    )


@app.command("list-models")
def list_models_command() -> None:
    """List complete valid bundles newest first."""
    from app.core.config import get_settings
    from app.ml.model_registry import list_valid_model_bundles, read_current_version

    settings = get_settings()
    current = read_current_version(settings.ARTIFACT_ROOT)
    for manifest in list_valid_model_bundles(settings.ARTIFACT_ROOT):
        marker = "current" if manifest.model_version == current else "available"
        typer.echo(
            f"{manifest.model_version} {marker} trained_at={manifest.trained_at} "
            f"films={manifest.film_count} users={manifest.eligible_user_count}"
        )


@app.command("current-model")
def current_model_command() -> None:
    """Report the authoritative model pointer without exposing filesystem paths."""
    from app.core.config import get_settings
    from app.ml.model_registry import read_current_version

    current = read_current_version(get_settings().ARTIFACT_ROOT)
    typer.echo(current or "legacy-flat")


@app.command("rollback-model")
def rollback_model_command(
    dry_run: Annotated[bool, typer.Option(help="Report rollback target only.")] = False,
) -> None:
    """Validate and atomically promote the immediately previous valid bundle."""
    from app.core.config import get_settings
    from app.ml.model_registry import (
        rollback_model,
        select_rollback_target,
    )

    settings = get_settings()
    try:
        if dry_run:
            target = select_rollback_target(settings.ARTIFACT_ROOT)
            typer.echo(f"rollback_target={target.model_version} dry_run=true")
            return
        from app.infrastructure.maintenance_lock import MaintenanceLock

        lock = MaintenanceLock(
            settings.MAINTENANCE_REDIS_URL,
            key="retraining",
            ttl_seconds=settings.MAINTENANCE_LOCK_TTL_SECONDS,
        )
        try:
            with lock.held() as acquired:
                if not acquired:
                    raise RuntimeError("model promotion is already active")
                result = rollback_model(settings.ARTIFACT_ROOT)
        finally:
            lock.close()
    except Exception as exc:
        typer.echo(f"Rollback failed; current model unchanged: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Rolled back to {result.model_version}; API activation pending")


if __name__ == "__main__":
    app()
