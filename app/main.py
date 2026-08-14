import contextlib
import logging

from fastapi import FastAPI

from app.api.routes import (
    health_router,
    imports_router,
    recommendations_router,
    tasks_router,
)
from app.services.categorized_recommendation_service import (
    build_categorized_recommendation_service,
)
from app.services.recommendation_service import get_recommendation_service
from app.services.task_service import get_task_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def lifespan(application: FastAPI):
    """Own both recommendation services for one application worker lifespan."""
    recommendation_service = get_recommendation_service()
    recommendation_service.load_artifacts()
    categorized_service = build_categorized_recommendation_service()
    application.state.categorized_recommendation_service = categorized_service
    try:
        if not await categorized_service.load_resources():
            logger.critical("Categorized recommendation resource loading failed")
            raise RuntimeError("categorized recommendation resources unavailable")
        yield
    finally:
        try:
            await get_task_service().close()
        finally:
            categorized_service.unload_resources()
            recommendation_service.unload_artifacts()
            del application.state.categorized_recommendation_service


app = FastAPI(lifespan=lifespan)
app.include_router(health_router)
app.include_router(imports_router)
app.include_router(tasks_router)
app.include_router(recommendations_router)
