"""Celery application for durable profile and maintenance workloads."""

from celery import Celery

from app.core.config import get_settings
from app.workers.schedules import MAINTENANCE_BEAT_SCHEDULE

PROFILE_SYNC_QUEUE = "profile_sync"
MAINTENANCE_QUEUE = "maintenance"

settings = get_settings()

celery_app = Celery(
    "nexdwatch",
    broker=settings.CELERY_BROKER_URL,
    include=["app.tasks.profile_sync", "app.tasks.maintenance"],
)
celery_app.conf.update(
    accept_content=["json"],
    broker_connection_retry_on_startup=True,
    broker_transport_options={
        "visibility_timeout": settings.CELERY_BROKER_VISIBILITY_TIMEOUT_SECONDS,
    },
    enable_utc=True,
    result_backend=None,
    task_acks_late=True,
    task_default_queue=PROFILE_SYNC_QUEUE,
    task_ignore_result=True,
    task_reject_on_worker_lost=True,
    task_routes={
        "app.tasks.profile_sync": {"queue": PROFILE_SYNC_QUEUE},
        "app.tasks.maintenance.*": {"queue": MAINTENANCE_QUEUE},
    },
    task_serializer="json",
    task_track_started=True,
    timezone="UTC",
    worker_prefetch_multiplier=1,
    beat_schedule=MAINTENANCE_BEAT_SCHEDULE,
)
