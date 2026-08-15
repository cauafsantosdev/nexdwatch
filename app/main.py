"""Constructs the FastAPI application and owns immutable serving resources."""

import asyncio
import contextlib
import logging

from fastapi import FastAPI

from app.api.routes import (
    health_router,
    imports_router,
    recommendations_router,
    tasks_router,
)
from app.core.config import get_settings
from app.services.categorized_recommendation_service import (
    build_categorized_recommendation_service,
)
from app.services.model_activation import ModelPointerWatcher, resolve_startup_model
from app.services.recommendation_service import get_recommendation_service
from app.services.task_service import get_task_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def lifespan(application: FastAPI):
    """Load one immutable model version and own all serving resources.

    Both public recommendation services resolve the same artifact location before
    startup completes. Shutdown cancels the lightweight pointer watcher and releases
    database, Redis, and model resources through normal lifecycle hooks.

    Args:
        application: FastAPI process whose state owns categorized serving resources.

    Yields:
        None: Control is yielded only after both recommendation paths are configured,
            immutable categorized resources load, and pointer monitoring starts.

    Raises:
        RuntimeError: If the categorized artifact/catalog graph cannot load safely.
    """
    settings = get_settings()
    # Resolve one authoritative legacy or versioned location before either service
    # loads resources, preventing mixed model versions within a worker process.
    model_location = resolve_startup_model(settings.ARTIFACT_ROOT)
    recommendation_service = get_recommendation_service()
    configure_legacy = getattr(recommendation_service, "configure_artifact_root", None)
    if configure_legacy is not None:
        configure_legacy(model_location.root)
    recommendation_service.load_artifacts()
    categorized_service = build_categorized_recommendation_service()
    configure_categorized = getattr(categorized_service, "configure_artifacts", None)
    if configure_categorized is not None:
        configure_categorized(
            model_location.root,
            model_location.popularity_path,
        )
    # Publish lifespan-owned service state only after both service configurations
    # reference the exact same resolved bundle.
    application.state.model_version = model_location.model_version
    logger.info(
        "Recommendation startup model_version=%s versioned=%s",
        model_location.model_version,
        model_location.versioned,
    )
    application.state.categorized_recommendation_service = categorized_service
    # Keep resources immutable while requests are active. A validated pointer change
    # asks Uvicorn to shut down gracefully instead of hot-swapping NumPy/FAISS state.
    watcher = ModelPointerWatcher(
        settings.ARTIFACT_ROOT,
        model_location.model_version,
        interval_seconds=settings.MODEL_POINTER_CHECK_INTERVAL_SECONDS,
    )
    watcher_task: asyncio.Task[None] | None = None
    try:
        # Complete catalog/artifact loading before accepting traffic, then monitor the
        # durable pointer without hot-swapping arrays used by active requests.
        if not await categorized_service.load_resources():
            logger.critical("Categorized recommendation resource loading failed")
            raise RuntimeError("categorized recommendation resources unavailable")
        watcher_task = asyncio.create_task(watcher.run(), name="model-pointer-watcher")
        yield
    finally:
        # Stop pointer observation first, then close Redis and release immutable model
        # resources in dependency order during graceful shutdown.
        if watcher_task is not None:
            watcher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher_task
        try:
            await get_task_service().close()
        finally:
            categorized_service.unload_resources()
            recommendation_service.unload_artifacts()
            del application.state.categorized_recommendation_service
            del application.state.model_version


app = FastAPI(lifespan=lifespan)
app.include_router(health_router)
app.include_router(imports_router)
app.include_router(tasks_router)
app.include_router(recommendations_router)
