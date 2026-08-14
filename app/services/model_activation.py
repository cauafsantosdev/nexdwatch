"""Low-cost model-pointer watching and graceful API process recycling."""

import asyncio
import logging
import os
import signal
from collections.abc import Callable
from pathlib import Path

from app.ml.model_registry import (
    ServingModelLocation,
    read_current_version,
    recover_previous_model,
    resolve_serving_model,
)

logger = logging.getLogger(__name__)


def request_graceful_process_recycle() -> None:
    """Ask the current ASGI server process to run its normal SIGTERM shutdown."""
    os.kill(os.getpid(), signal.SIGTERM)


def resolve_startup_model(artifact_root: str | Path) -> ServingModelLocation:
    """Resolve current, rolling back one failed activation before API resources load."""
    try:
        return resolve_serving_model(artifact_root, validate=True)
    except (OSError, ValueError, TypeError) as activation_error:
        try:
            failed_version = read_current_version(artifact_root)
        except (OSError, ValueError, TypeError):
            raise activation_error
        if failed_version is None:
            raise
        logger.critical(
            "Promoted model failed startup validation version=%s; attempting one recovery",
            failed_version,
            exc_info=activation_error,
        )
        return recover_previous_model(artifact_root, failed_version)


class ModelPointerWatcher:
    """Validate changed pointers and request at most one process recycle."""

    def __init__(
        self,
        artifact_root: str | Path,
        loaded_version: str,
        *,
        interval_seconds: float,
        recycle: Callable[[], None] = request_graceful_process_recycle,
    ) -> None:
        self._artifact_root = Path(artifact_root)
        self._loaded_version = loaded_version
        self._interval_seconds = interval_seconds
        self._recycle = recycle
        self._recycle_requested = False

    @property
    def recycle_requested(self) -> bool:
        return self._recycle_requested

    def check_once(self) -> bool:
        """Read only current.json unless its version differs from loaded state."""
        if self._recycle_requested:
            return False
        try:
            promoted = read_current_version(self._artifact_root) or "legacy-flat"
        except (OSError, ValueError, TypeError):
            logger.exception(
                "Ignoring malformed model pointer while model_version=%s remains loaded",
                self._loaded_version,
            )
            return False
        if promoted == self._loaded_version:
            return False
        try:
            location = resolve_serving_model(self._artifact_root, validate=True)
        except (OSError, ValueError, TypeError):
            logger.exception(
                "Ignoring invalid promoted model version=%s; loaded version=%s remains active",
                promoted,
                self._loaded_version,
            )
            return False
        if location.model_version != promoted:
            logger.error("Model pointer changed during activation validation")
            return False
        self._recycle_requested = True
        logger.warning(
            "Model change detected old_loaded_version=%s new_promoted_version=%s; "
            "API graceful recycle requested",
            self._loaded_version,
            promoted,
        )
        self._recycle()
        return True

    async def run(self) -> None:
        """Poll conservatively until shutdown or the first valid transition."""
        while not self._recycle_requested:
            await asyncio.sleep(self._interval_seconds)
            self.check_once()
