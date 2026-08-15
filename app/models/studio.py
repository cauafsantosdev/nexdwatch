"""Defines the persisted studio entity and its film relationship."""

from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Studio(Base):
    """Normalized production-studio identity attached to catalog films."""
    __tablename__ = "studios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)

    films: Mapped[list["Film"]] = relationship(
        secondary="film_studios",
        back_populates="studios",
        lazy="selectin"
    )
