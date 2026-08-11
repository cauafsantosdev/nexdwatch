"""Command-line interface for the isolated ranking benchmark."""

import logging
from pathlib import Path

import typer

from experiments.ranker.benchmark import run_benchmark
from experiments.ranker.config import RANKER_SEEDS

app = typer.Typer(no_args_is_help=True)


@app.command()
def run(
    csv_path: Path = typer.Option(..., exists=True, dir_okay=False),  # noqa: B008
    output_root: Path = typer.Option(  # noqa: B008
        Path("notebooks/data/ranker_full_pool_v2")
    ),
    seeds: str = typer.Option("42,43,44"),
    folds: str = typer.Option("0,1,2,3,4"),
    ablations: str = typer.Option("full"),
) -> None:
    """Run requested seed/fold rankers and write research-only artifacts."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    selected_seeds = tuple(int(value) for value in seeds.split(",") if value)
    selected_folds = tuple(int(value) for value in folds.split(",") if value)
    if not set(selected_seeds).issubset(RANKER_SEEDS):
        raise typer.BadParameter("seeds must be selected from 42,43,44")
    if any(fold not in range(5) for fold in selected_folds):
        raise typer.BadParameter("folds must be selected from 0,1,2,3,4")
    selected_ablations = (
        None
        if ablations == "all"
        else tuple(value for value in ablations.split(",") if value)
    )
    summary = run_benchmark(
        csv_path=csv_path,
        output_root=output_root,
        seeds=selected_seeds,
        folds=selected_folds,
        ablations=selected_ablations,
    )
    typer.echo(f"Completed {summary['completed_models']} ranker fold(s).")


if __name__ == "__main__":
    app()
