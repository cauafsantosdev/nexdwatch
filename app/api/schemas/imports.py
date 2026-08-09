"""Transport schemas for offline Letterboxd imports."""

from pydantic import BaseModel, ConfigDict


class UnresolvedImportFilm(BaseModel):
    """A bounded unresolved-film sample item."""

    model_config = ConfigDict(from_attributes=True)

    name: str
    year: int | None
    uri: str
    reason: str


class LetterboxdImportResponse(BaseModel):
    """Offline import summary."""

    user_id: int | None
    watched_in_export: int
    rated_in_export: int
    imported: int
    unresolved: int
    unresolved_sample: list[UnresolvedImportFilm]
