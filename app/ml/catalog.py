"""Controlled catalog identity loading for offline ML workflows."""

from sqlalchemy import create_engine, text

from app.core.config import Settings, get_settings


def load_catalog_slug_mapping(settings: Settings | None = None) -> dict[str, int]:
    """Load the catalog slug-to-film-ID mapping in one read-only query."""
    active_settings = settings or get_settings()
    sync_db_url = (
        "postgresql+psycopg2://"
        f"{active_settings.POSTGRES_USER}:{active_settings.POSTGRES_PASSWORD}"
        f"@{active_settings.POSTGRES_HOST}:{active_settings.POSTGRES_PORT}/"
        f"{active_settings.POSTGRES_DB}"
    )
    engine = create_engine(sync_db_url)
    try:
        with engine.connect() as connection:
            rows = connection.execute(text("SELECT slug, id FROM films")).all()
    finally:
        engine.dispose()
    mapping = {str(slug): int(film_id) for slug, film_id in rows}
    if len(mapping) != len(rows):
        raise ValueError("database film slugs are not unique")
    return mapping
