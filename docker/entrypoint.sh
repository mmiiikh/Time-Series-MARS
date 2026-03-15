#!/bin/bash
set -e

dvc pull

if [ "$1" = "worker" ]; then
    echo "Starting Celery worker"
    celery -A src.worker.celery_app worker --loglevel=info
else
    echo "Initializing database"
    python src/scripts/init_db.py

    echo "Starting FastAPI server"
    gunicorn src.api.main:app \
        -k uvicorn.workers.UvicornWorker \
        --bind 0.0.0.0:8000 \
        --workers 4
fi

#echo "Running SARIMA training..."
#python src/training/train_sarima.py