"""Worker-process-local bridge into the asynchronous service layer."""

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

from celery.signals import worker_process_init, worker_process_shutdown

from app.core.database import engine

T = TypeVar("T")


class WorkerAsyncBridge:
    """Own one asyncio runner for the lifetime of each Celery worker process.

    Celery task bodies remain synchronous while reusing async SQLAlchemy resources on
    one stable event loop. Fork lifecycle signals create the runner in the child and
    dispose database connections before the loop closes.
    """

    def __init__(self) -> None:
        self._runner: asyncio.Runner | None = None

    def start(self) -> None:
        """Initialize the worker-local event loop lazily."""
        if self._runner is None:
            self._runner = asyncio.Runner()

    def run(self, coroutine: Coroutine[Any, Any, T]) -> T:
        """Run one async service operation on the persistent worker loop.

        Returns:
            T: The coroutine result; exceptions propagate to the Celery task boundary.
        """
        self.start()
        if self._runner is None:  # pragma: no cover - defensive type narrowing
            raise RuntimeError("worker async runner is unavailable")
        return self._runner.run(coroutine)

    def close(self) -> None:
        """Dispose async database connections before closing the worker loop.

        Shutdown is idempotent so Celery signal ordering cannot close the runner twice.
        """
        if self._runner is None:
            return
        self._runner.run(engine.dispose())
        self._runner.close()
        self._runner = None


worker_async_bridge = WorkerAsyncBridge()


@worker_process_init.connect
def initialize_worker_loop(**_: object) -> None:
    """Initialize async resources after Celery forks a worker process."""
    worker_async_bridge.start()


@worker_process_shutdown.connect
def close_worker_loop(**_: object) -> None:
    """Close async resources before a worker process exits."""
    worker_async_bridge.close()
