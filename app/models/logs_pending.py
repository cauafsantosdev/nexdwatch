"""Defines interactions awaiting resolution of an unknown film slug."""

from sqlalchemy import Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Enum

from app.core.database import Base
from app.models.status import Status


class LogPending(Base):
    """Interaction awaiting film catalog ingestion."""

    __tablename__ = "logs_pending"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(15))
    film_slug: Mapped[str] = mapped_column(String(255))
    rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[Status] = mapped_column(
        Enum(Status, name="status"),
        default=Status.PENDING,
        server_default=Status.PENDING.value,
        nullable=False,
    )
