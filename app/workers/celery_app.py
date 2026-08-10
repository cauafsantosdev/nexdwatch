"""Celery application for durable profile synchronization."""

from celery import Celery

from app.core.config import get_settings

PROFILE_SYNC_QUEUE = "profile_sync"

settings = get_settings()

celery_app = Celery(
    "nexdwatch",
    broker=settings.CELERY_BROKER_URL,
    include=["app.tasks.profile_sync"],
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
    },
    task_serializer="json",
    task_track_started=True,
    timezone="UTC",
    worker_prefetch_multiplier=1,
)
