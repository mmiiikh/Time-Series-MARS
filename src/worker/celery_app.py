from celery import Celery

celery_app = Celery(
    "mars_tasks",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/0",
)

celery_app.conf.update(
    task_track_started=True,
)

celery_app.autodiscover_tasks(["src.worker"])