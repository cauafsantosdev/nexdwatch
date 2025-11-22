import re
import ast
import typer
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session as AsyncSession

from app.models.film import Film
from app.models.director import Director
from app.models.actor import Actor
from app.models.genre import Genre
from app.models.country import Country
from app.models.language import Language
from app.models.theme import Theme
from app.models.studio import Studio
from app.core.database import SessionLocal


def safe_convert(value, dtype):
    """
    Converts Pandas values safely
    """
    if pd.isna(value):
        return None
    
    try:
        if dtype == int:
            return int(float(value)) 
        if dtype == float:
            return float(value)
        if dtype == str:
            clean_val = str(value).strip()
            return clean_val if clean_val else None
    except (ValueError, TypeError):
        return None
    
    return str(value)

def parse_list(value):
    """
    Converts CSV fields to lists of strings safely.
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
    """Loads a CSV of film details into the database"""
    async with SessionLocal() as session:
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
            
            # Cache Dictionary: {Model: {Name: ORM Object}}
            caches = {}
            typer.echo("  > Loading existing models from database...")
            
            for model, _, _ in relation_map:
                # Selects all existent objects for this model in one query
                result = await session.execute(select(model))
                # Creates a cache for the model
                caches[model] = {obj.name: obj for obj in result.scalars().all()}
            
            typer.echo("  > Loading finished.")

            def get_or_create(model, name, session: AsyncSession):
                """Gets in cache or creates ORM object."""
                cache = caches[model]

                if name in cache:
                    obj = cache[name]
                    return obj 
                
                # If it's a new object, it's created and added to session and cache
                obj = model(name=name)
                session.add(obj)
                cache[name] = obj
                return obj

            for index, row in df.iterrows():

                original_title_value = safe_convert(row["original_title"], str)                
                # If original_title_value is None, uses 'title' as fallback
                if original_title_value is None:
                    original_title_value = safe_convert(row["title"], str)
                
                # Creates Film object
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

                # Adds relationships using cache
                for model, column, relation_list in relation_map:
                    items = parse_list(row[column])
                    unique_names = set(items)

                    for name in unique_names:
                        obj = get_or_create(model, name, session)
                        getattr(film, relation_list).append(obj)

                # Progress feedback
                if (index + 1) % 1000 == 0:
                    typer.echo(f"  > Processed Films: {index + 1}/{len(df)}")