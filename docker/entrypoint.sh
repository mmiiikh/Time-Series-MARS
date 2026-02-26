#!/bin/bash
set -e

echo "Pulling data from DVC remote..."
dvc pull

if [ "$1" = "worker" ]; then
    echo "Starting Celery worker"
    celery -A src.worker.celery_app worker --loglevel=info
else
    echo "Starting FastAPI server..."
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
fi

#echo "Running SARIMA training..."
#python src/training/train_sarima.py