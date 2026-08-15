"""Creates the process-wide asynchronous PostgreSQL engine and session factory."""

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from .config import get_settings

settings = get_settings()

class Base(DeclarativeBase):
    """Declarative base shared by all persisted NexdWatch models."""

engine = create_async_engine(settings.DATABASE_URL, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
