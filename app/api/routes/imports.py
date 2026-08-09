"""Offline Letterboxd export route."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Path, UploadFile, status

from app.api.schemas.imports import LetterboxdImportResponse
from app.importers.letterboxd_export import LetterboxdExportError
from app.services.letterboxd_import_service import (
    LetterboxdImportService,
    NoResolvedFilmsError,
    get_letterboxd_import_service,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/users/{username}/import", response_model=LetterboxdImportResponse)
async def import_letterboxd_export(
    username: Annotated[str, Path(min_length=1, max_length=15)],
    file: Annotated[UploadFile, File(description="Official Letterboxd export ZIP")],
    service: Annotated[LetterboxdImportService, Depends(get_letterboxd_import_service)],
) -> LetterboxdImportResponse:
    """Import watched films from an official Letterboxd export ZIP."""
    normalized_username = username.strip()
    if not normalized_username:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Username cannot be blank.",
        )

    try:
        archive = await file.read()
        result = await service.import_export(normalized_username, archive)
    except LetterboxdExportError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except NoResolvedFilmsError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception(
            "Letterboxd export import failed for username=%s", normalized_username
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Letterboxd export import failed.",
        ) from exc
    finally:
        await file.close()

    return LetterboxdImportResponse(
        user_id=result.user_id,
        watched_in_export=result.watched_in_export,
        rated_in_export=result.rated_in_export,
        imported=result.imported,
        unresolved=result.unresolved,
        unresolved_sample=list(result.unresolved_films[:20]),
    )
