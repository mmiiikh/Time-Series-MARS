from fastapi import FastAPI
from src.worker.tasks import train_sarima_task
from src.utils.db import create_task, get_task

app = FastAPI()

@app.post("/train")
def start_training():
    task_id = create_task("SARIMA")
    train_sarima_task.delay(task_id)

    return {
        "task_id": task_id,
        "status": "PENDING"
    }


@app.get("/status/{task_id}")
def get_status(task_id: int):
    return get_task(task_id)