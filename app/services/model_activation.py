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
    """Ask Uvicorn to perform normal SIGTERM shutdown and lifespan cleanup.

    The API signals only itself. Docker socket access and host-runtime control are
    intentionally unnecessary; Compose's restart policy owns process replacement.
    """
    os.kill(os.getpid(), signal.SIGTERM)


def resolve_startup_model(artifact_root: str | Path) -> ServingModelLocation:
    """Resolve current, restoring one previous valid selection on activation failure.

    Recovery occurs before either serving service loads resources, preventing mixed
    versions and bounding startup failure instead of entering a restart loop.

    Returns:
        ServingModelLocation: Fully validated selected or once-restored model root.

    Raises:
        Exception: Re-raises activation validation when no safe predecessor can be
            identified and restored.
    """
    # Validate the authoritative selection before service construction. Recovery is
    # attempted only for a readable versioned pointer, never an invalid flat layout.
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
    """Watch one loaded version and request at most one graceful process recycle.

    Steady-state checks read only ``current.json``. Full artifacts are validated only
    after a different version is observed, preserving immutable lifespan resources.
    """

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
        """Return whether this watcher has already requested its one transition."""
        return self._recycle_requested

    def check_once(self) -> bool:
        """Validate a changed selection and request one graceful recycle when ready.

        Steady state reads only the pointer. A changed version is fully validated and
        re-read for identity consistency before signalling; malformed or incomplete
        promotions are ignored while current in-memory resources continue serving.

        Returns:
            bool: ``True`` only when this call requested the watcher's sole recycle.
        """
        if self._recycle_requested:
            return False
        # Pointer parsing is deliberately cheap and failure-tolerant in steady state.
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
        # Pay full checksum/cross-artifact validation only after identity changes.
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
        """Poll conservatively until cancellation or the first valid transition.

        Returns:
            None: The coroutine exits after requesting one recycle; normal API
                shutdown cancels it from the lifespan context.
        """
        while not self._recycle_requested:
            await asyncio.sleep(self._interval_seconds)
            self.check_once()
