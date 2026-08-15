"""Expose bounded offline Letterboxd ZIP ingestion as a synchronous fallback."""

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
    """Import an official export and return resolution-safe aggregate results.

    The upload is read once and delegated to the archive/service boundaries. Invalid
    archives return 400, valid exports with no resolvable catalog films return 422,
    and unexpected persistence failures return a sanitized 500. The upload handle is
    closed for every outcome.
    """
    normalized_username = username.strip()
    if not normalized_username:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Username cannot be blank.",
        )

    # Read and validate the complete bounded archive before opening service-owned
    # database work; parser limits constrain the in-memory request body.
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

    # Expose only a bounded unresolved sample; internal matching diagnostics and the
    # complete unresolved inventory remain outside the public response.
    return LetterboxdImportResponse(
        user_id=result.user_id,
        watched_in_export=result.watched_in_export,
        rated_in_export=result.rated_in_export,
        imported=result.imported,
        unresolved=result.unresolved,
        unresolved_sample=list(result.unresolved_films[:20]),
    )
