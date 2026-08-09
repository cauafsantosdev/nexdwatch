import asyncio
from pathlib import Path

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


if __name__ == "__main__":
    app()
