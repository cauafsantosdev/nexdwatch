"""Define the single-scheduler UTC cadence for backend maintenance.

Film queue processing and retraining evaluation run weekly. Aggregate catalog fields
refresh in January and July; distributed task locks make duplicate Beat delivery a
safe skip, but production must still operate exactly one scheduler instance.
"""

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
