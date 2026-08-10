import contextlib
import logging

from fastapi import FastAPI

from app.api.routes import (
    health_router,
    imports_router,
    recommendations_router,
    tasks_router,
)
from app.services.recommendation_service import get_recommendation_service
from app.services.task_service import get_task_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    """Load recommendation artifacts for the application lifespan."""
    recommendation_service = get_recommendation_service()
    recommendation_service.load_artifacts()
    try:
        yield
    finally:
        await get_task_service().close()
        recommendation_service.unload_artifacts()


app = FastAPI(lifespan=lifespan)
app.include_router(health_router)
app.include_router(imports_router)
app.include_router(tasks_router)
app.include_router(recommendations_router)
