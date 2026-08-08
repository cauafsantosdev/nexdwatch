from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Enum

from app.core.database import Base
from app.models.status import Status


class FilmQueue(Base):
    """Film scraping queue with terminal outcome metadata."""

    __tablename__ = "films_queue"
    __table_args__ = (UniqueConstraint("film_slug", name="uq_films_queue_film_slug"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    film_slug: Mapped[str] = mapped_column(String(255))
    status: Mapped[Status] = mapped_column(
        Enum(Status, name="status"),
        default=Status.PENDING,
        server_default=Status.PENDING.value,
        nullable=False,
    )
    attempts: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
