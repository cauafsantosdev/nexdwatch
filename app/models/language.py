"""Defines the persisted language entity and its film relationship."""

from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Language(Base):
    """Normalized film-language identity used by cultural preference policy."""
    __tablename__ = "languages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)

    films: Mapped[list["Film"]] = relationship(
        secondary="film_languages",
        back_populates="languages",
        lazy="selectin"
    )
