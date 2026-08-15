"""Defines the persisted country entity and its film relationship."""

from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Country(Base):
    """Normalized production-country identity used by film and policy metadata."""
    __tablename__ = "countries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)

    films: Mapped[list["Film"]] = relationship(
        secondary="film_countries",
        back_populates="countries",
        lazy="selectin"
    )
