import enum
from sqlalchemy import Integer, String, Float
from sqlalchemy.types import Enum
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base


class Status(enum.Enum):
    PENDING = "PENDING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"

class LogPending(Base):
    """Logs Pending Scraping table"""
    __tablename__ = "logs_pending"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(15))
    film_slug: Mapped[str] = mapped_column(String(255))
    rating: Mapped[float] = mapped_column(Float)
    status: Mapped[Status] = mapped_column(Enum(Status), default=Status.PENDING, 
                                           server_default=Status.PENDING.value, nullable=False
    )