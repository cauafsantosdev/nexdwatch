from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy import Integer, String, Text, Float
from app.core.database import Base


class Film(Base):
    """Films table"""
    __tablename__ = "films"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tmdb_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=True)
    slug: Mapped[str] = mapped_column(String(255), unique=True, index=True)

    title: Mapped[str] = mapped_column(String(255))
    original_title: Mapped[str] = mapped_column(String(255))
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    runtime: Mapped[int | None] = mapped_column(Integer, nullable=True)

    synopsis: Mapped[str | None] = mapped_column(Text, nullable=True)
    tagline: Mapped[str | None] = mapped_column(Text, nullable=True)

    directors: Mapped[list["Director"]] = relationship(
        secondary="film_directors",
        back_populates="films",
        lazy="selectin"
    )
    actors: Mapped[list["Actor"]] = relationship(
        secondary="film_actors",
        back_populates="films",
        lazy="selectin"
    )
    genres: Mapped[list["Genre"]] = relationship(
        secondary="film_genres",
        back_populates="films",
        lazy="selectin"
    )
    countries: Mapped[list["Country"]] = relationship(
        secondary="film_countries",
        back_populates="films",
        lazy="selectin"
    )
    languages: Mapped[list["Language"]] = relationship(
        secondary="film_languages",
        back_populates="films",
        lazy="selectin"
    )
    studios: Mapped[list["Studio"]] = relationship(
        secondary="film_studios",
        back_populates="films",
        lazy="selectin"
    )
    themes: Mapped[list["Theme"]] = relationship(
        secondary="film_themes",
        back_populates="films",
        lazy="selectin"
    )
    avg_rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_logs: Mapped[int | None] = mapped_column(Integer, nullable=True)

    logs: Mapped[list["Log"]] = relationship(back_populates="film", lazy="selectin") 