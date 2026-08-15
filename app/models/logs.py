"""Defines resolved user-film interactions used by serving and training."""

from sqlalchemy import Float, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Log(Base):
    """Persisted rated or unrated user-film interaction."""

    __tablename__ = "logs"
    __table_args__ = (
        UniqueConstraint("user_id", "film_id", name="uq_logs_user_id_film_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    user: Mapped["User"] = relationship(back_populates="logs", lazy="selectin")

    film_id: Mapped[int] = mapped_column(ForeignKey("films.id"))
    film: Mapped["Film"] = relationship(back_populates="logs", lazy="selectin")

    rating: Mapped[float | None] = mapped_column(Float, nullable=True)
