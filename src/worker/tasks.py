from src.worker.celery_app import celery_app
from src.training.train_sarima import train_sarima
from src.utils.db import update_task_status


@celery_app.task(bind=True)
def train_sarima_task(self, task_id: int):
    update_task_status(task_id, "RUNNING")

    try:
        mape = train_sarima()
        update_task_status(task_id, "SUCCESS", mape=mape)
    except Exception as e:
        update_task_status(task_id, "FAILED")
        raise e