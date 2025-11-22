from sqlalchemy import Table, Column, Integer, ForeignKey
from app.core.database import Base


film_directors = Table(
    "film_directors",
    Base.metadata,
    Column("film_id", Integer, ForeignKey("films.id"), primary_key=True),
    Column("director_id", Integer, ForeignKey("directors.id"), primary_key=True)
)

film_actors = Table(
    "film_actors",
    Base.metadata,
    Column("film_id", Integer, ForeignKey("films.id"), primary_key=True),
    Column("actor_id", Integer, ForeignKey("actors.id"), primary_key=True)
)

film_genres = Table(
    "film_genres",
    Base.metadata,
    Column("film_id", Integer, ForeignKey("films.id"), primary_key=True),
    Column("genre_id", Integer, ForeignKey("genres.id"), primary_key=True)
)

film_countries = Table(
    "film_countries",
    Base.metadata,
    Column("film_id", Integer, ForeignKey("films.id"), primary_key=True),
    Column("country_id", Integer, ForeignKey("countries.id"), primary_key=True)
)

film_languages = Table(
    "film_languages",
    Base.metadata,
    Column("film_id", Integer, ForeignKey("films.id"), primary_key=True),
    Column("language_id", Integer, ForeignKey("languages.id"), primary_key=True)
)

film_studios = Table(
    "film_studios",
    Base.metadata,
    Column("film_id", Integer, ForeignKey("films.id"), primary_key=True),
    Column("studio_id", Integer, ForeignKey("studios.id"), primary_key=True)
)

film_themes = Table(
    "film_themes",
    Base.metadata,
    Column("film_id", Integer, ForeignKey("films.id"), primary_key=True),
    Column("theme_id", Integer, ForeignKey("themes.id"), primary_key=True)
)