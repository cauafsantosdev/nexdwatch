"""Central UTC Celery Beat schedule for low-frequency maintenance."""

from celery.schedules import crontab

# All values are UTC. Production must run exactly one Beat scheduler instance.
MAINTENANCE_BEAT_SCHEDULE = {
    "weekly-film-queue": {
        "task": "app.tasks.maintenance.process_film_queue",
        "schedule": crontab(minute=0, hour=2, day_of_week="sun"),
    },
    "weekly-retraining-evaluation": {
        "task": "app.tasks.maintenance.evaluate_retraining",
        "schedule": crontab(minute=0, hour=4, day_of_week="sun"),
    },
    "january-catalog-refresh": {
        "task": "app.tasks.maintenance.refresh_recent_catalog",
        "schedule": crontab(minute=0, hour=3, day_of_month="15", month_of_year="1"),
    },
    "july-catalog-refresh": {
        "task": "app.tasks.maintenance.refresh_recent_catalog",
        "schedule": crontab(minute=0, hour=3, day_of_month="15", month_of_year="7"),
    },
}
