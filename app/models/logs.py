from sqlalchemy import Integer, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base


class Log(Base):
    """Users Logs table"""
    __tablename__ = "logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    user: Mapped["User"] = relationship(back_populates="logs", lazy="selectin")

    film_id: Mapped[int] = mapped_column(ForeignKey("films.id"))
    film: Mapped["Film"] = relationship(back_populates="logs", lazy="selectin")
    
    rating: Mapped[int] = mapped_column(Integer)