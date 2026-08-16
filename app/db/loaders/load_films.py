"""Import the externally supplied historical film catalog into PostgreSQL.

The loader resolves repeated relationship names through in-memory ORM caches and
persists the complete CSV in one transaction. It is an administrative bootstrap
path rather than the incremental ``FilmQueue`` ingestion workflow.
"""

import ast
import re

import pandas as pd
import typer
from sqlalchemy import select
from sqlalchemy.orm import Session as AsyncSession

from app.core.database import SessionLocal
from app.models.actor import Actor
from app.models.country import Country
from app.models.director import Director
from app.models.film import Film
from app.models.genre import Genre
from app.models.language import Language
from app.models.studio import Studio
from app.models.theme import Theme


def safe_convert(value, dtype):
    """Normalize one pandas cell to the requested scalar representation.

    Missing values, blank strings, and conversion failures become ``None`` so a
    malformed optional field does not abort the catalog bootstrap.

    Args:
        value: Raw value read from the CSV frame.
        dtype: Target scalar type; integer conversion accepts numeric strings and
            floating-point cells produced by pandas.

    Returns:
        The converted scalar, ``None`` for unusable values, or a string fallback
        for target types outside the loader's explicit conversion set.
    """
    if pd.isna(value):
        return None
    
    try:
        if dtype is int:
            return int(float(value)) 
        if dtype is float:
            return float(value)
        if dtype is str:
            clean_val = str(value).strip()
            return clean_val if clean_val else None
    except (ValueError, TypeError):
        return None
    
    return str(value)

def parse_list(value):
    """Parse one serialized relationship column into clean entity names.

    Historical exports contain Python-list syntax plus inconsistently doubled
    quotes. The parser repairs those known quoting artifacts, accepts only string
    members, and treats any malformed representation as an empty relationship.

    Args:
        value: CSV cell containing a list, list-like string, or missing value.

    Returns:
        list[str]: Non-empty relationship names in source order.
    """
    if value is None:
        return []

    if isinstance(value, list):
        return value

    value = str(value).strip()

    if value in ("", "None", "nan"):
        return []

    value = re.sub(r'""([^"]+)""', r'"\1"', value)
    value = re.sub(r'""([^"]+)""', r'"\1"', value)
    value = value.replace('""', '"')

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [x.strip() for x in parsed if isinstance(x, str) and x.strip()]
        return []
    except Exception:
        return []


async def load_films_data(csv_path):
    """Persist a historical film CSV and its normalized relationships.

    The CSV is loaded into memory, existing relationship tables are cached once,
    and every film plus newly encountered entity is written in a single database
    transaction. An exception therefore rolls back the entire bootstrap import.

    Args:
        csv_path: Semicolon-delimited catalog export to import.

    Returns:
        None: Films and relationships are committed directly to PostgreSQL.

    Raises:
        Exception: Propagates CSV parsing, database, and integrity failures after
            the enclosing transaction has rolled back.
    """
    async with SessionLocal() as session:
        # Parse before opening the transaction so file I/O does not extend the
        # database write-lock window.
        df = pd.read_csv(csv_path, sep=";")

        async with session.begin():
            relation_map = [
                (Director, "director", "directors"),
                (Actor, "actors", "actors"),
                (Genre, "genre", "genres"),
                (Language, "language", "languages"),
                (Country, "country", "countries"),
                (Studio, "studio", "studios"),
                (Theme, "themes", "themes"),
            ]
            
            # Preload shared entities once; per-row relationship resolution must not
            # degrade into a query for every director, actor, or genre.
            caches = {}
            typer.echo("  > Loading existing models from database...")
            
            for model, _, _ in relation_map:
                result = await session.execute(select(model))
                caches[model] = {obj.name: obj for obj in result.scalars().all()}
            
            typer.echo("  > Loading finished.")

            def get_or_create(model, name, session: AsyncSession):
                """Reuse a transaction-local relationship entity or stage a new one."""
                cache = caches[model]

                if name in cache:
                    obj = cache[name]
                    return obj 
                
                # Cache new entities immediately so duplicate names within the same
                # import resolve to one ORM identity before the session is flushed.
                obj = model(name=name)
                session.add(obj)
                cache[name] = obj
                return obj

            for index, row in df.iterrows():
                # Normalize scalar metadata while preserving the established title
                # fallback for incomplete historical rows.
                original_title_value = safe_convert(row["original_title"], str)                
                # If original_title_value is None, uses 'title' as fallback
                if original_title_value is None:
                    original_title_value = safe_convert(row["title"], str)
                
                film = Film(
                    tmdb_id=safe_convert(row["tmdb_id"], int),
                    slug=safe_convert(row["slug"], str),
                    title=safe_convert(row["title"], str),
                    original_title=original_title_value,
                    year=safe_convert(row["year"], int),
                    runtime=safe_convert(row["runtime"], int),
                    synopsis=safe_convert(row["synopsis"], str),
                    tagline=safe_convert(row["tagline"], str),
                    avg_rating=safe_convert(row["avg_rating"], float),
                    total_logs=safe_convert(row["total_logs"], int)
                )
                session.add(film)

                # Attach deduplicated normalized entities from each list-like field.
                for model, column, relation_list in relation_map:
                    items = parse_list(row[column])
                    unique_names = set(items)

                    for name in unique_names:
                        obj = get_or_create(model, name, session)
                        getattr(film, relation_list).append(obj)

                if (index + 1) % 1000 == 0:
                    typer.echo(f"  > Processed Films: {index + 1}/{len(df)}")
