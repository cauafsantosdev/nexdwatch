import enum
from sqlalchemy import Integer, String
from sqlalchemy.types import Enum
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base


class Status(enum.Enum):
    PENDING = "PENDING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"

class FilmQueue(Base):
    """Film Scraping Queue table"""
    __tablename__ = "films_queue"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    film_slug: Mapped[str] = mapped_column(String(255))
    status: Mapped[Status] = mapped_column(Enum(Status), default=Status.PENDING, 
                                           server_default=Status.PENDING.value, nullable=False
    )