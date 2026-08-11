"""Standalone commands for reproducing the neural retrieval experiment."""

from pathlib import Path
from statistics import fmean, pstdev

import typer

from experiments.neural_retrieval.artifacts import rebuild_ncf_index
from experiments.neural_retrieval.training import (
    RetrievalMetrics,
    benchmark_ncf_models,
    train_ncf_model,
)

app = typer.Typer()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
USERS_CSV = DATA_DIR / "users_data.csv"
ARTIFACT_ROOT = DATA_DIR / "ncf"


@app.command()
def train() -> None:
    """Train, evaluate, and persist one inductive neural retriever."""
    result = train_ncf_model(csv_path=USERS_CSV, artifact_root=ARTIFACT_ROOT)
    typer.echo(
        f"seed={result.seed} users={result.training_users} "
        f"films={result.candidate_films} best_epoch={result.best_epoch} "
        f"device={result.device} runtime_seconds={result.total_runtime_seconds:.3f}"
    )
    _print_metrics("Popularity", result.popularity_test_metrics)
    _print_metrics("Leakage-free SVD", result.svd_test_metrics)
    _print_metrics("Inductive neural retrieval", result.test_metrics)
    typer.echo(f"Artifacts: {result.artifact_root}")


@app.command()
def benchmark(
    seeds: str = typer.Option("42,43,44", help="Comma-separated random seeds."),
) -> None:
    """Run non-persisted leakage-safe comparisons over explicit seeds."""
    parsed = tuple(int(value.strip()) for value in seeds.split(","))
    results = benchmark_ncf_models(parsed, csv_path=USERS_CSV)
    for result in results:
        typer.echo(f"Seed {result.seed}")
        _print_metrics("Popularity", result.popularity_test_metrics)
        _print_metrics("Leakage-free SVD", result.svd_test_metrics)
        _print_metrics("Inductive neural retrieval", result.test_metrics)
    typer.echo("Aggregate mean ± population standard deviation")
    for label, result_field in (
        ("Popularity", "popularity_test_metrics"),
        ("Leakage-free SVD", "svd_test_metrics"),
        ("Inductive neural retrieval", "test_metrics"),
    ):
        for metric in ("recall_at_10", "recall_at_50", "ndcg_at_10", "mrr_at_10"):
            values = [
                getattr(getattr(result, result_field), metric) for result in results
            ]
            typer.echo(f"{label} {metric}: {fmean(values):.6f} ± {pstdev(values):.6f}")


@app.command("build-index")
def build_index() -> None:
    """Rebuild the experiment's exact FAISS index without retraining."""
    result = rebuild_ncf_index(ARTIFACT_ROOT)
    typer.echo(
        f"Indexed {result.film_count} films with dimension {result.dimension}: "
        f"{result.output_path}"
    )


def _print_metrics(label: str, metrics: RetrievalMetrics) -> None:
    typer.echo(
        f"{label}: Recall@10={metrics.recall_at_10:.6f} "
        f"Recall@50={metrics.recall_at_50:.6f} "
        f"NDCG@10={metrics.ndcg_at_10:.6f} "
        f"MRR@10={metrics.mrr_at_10:.6f}"
    )


if __name__ == "__main__":
    app()
