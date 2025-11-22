import asyncio
import typer
from pathlib import Path

from app.db.loaders.load_films import load_films_data
from app.db.loaders.load_logs import load_logs_data


app = typer.Typer()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
FILMS_CSV_PATH = str(DATA_DIR / "films_data.csv")
LOGS_CSV_PATH = str(DATA_DIR / "users_data.csv")

@app.command()
def load_films():
    typer.echo(f"Starting to load films from: {FILMS_CSV_PATH}")
    asyncio.run(load_films_data(FILMS_CSV_PATH))
    typer.echo("Films loaded successfully.")


@app.command()
def load_logs():
    typer.echo(f"Starting to load logs from: {LOGS_CSV_PATH}")
    asyncio.run(load_logs_data(LOGS_CSV_PATH))
    typer.echo("Logs loaded successfully.")


@app.command()
def load_all():
    """Triggers both load_films_data and load_logs_data"""
    asyncio.run(load_films_data(FILMS_CSV_PATH))
    asyncio.run(load_logs_data(LOGS_CSV_PATH))
    typer.echo("All data loaded successfully.")

if __name__ == "__main__":
    app()