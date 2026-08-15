"""Defines persisted Letterboxd users and their resolved interactions."""

from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class User(Base):
    """Local profile identity keyed by the synchronized Letterboxd username."""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(15), unique=True)

    logs: Mapped[list["Log"]] = relationship(back_populates="user", lazy="selectin")
