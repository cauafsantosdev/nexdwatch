"""Defines the persisted actor entity and its film relationship."""

from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Actor(Base):
    """Normalized actor identity shared across film cast relationships."""
    __tablename__ = "actors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)

    films: Mapped[list["Film"]] = relationship(
        secondary="film_actors",
        back_populates="actors",
        lazy="selectin"
    )
