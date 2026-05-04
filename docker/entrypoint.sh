#!/bin/bash
set -e

case "$1" in
  api)
    echo "Starting FastAPI"
    echo "Pulling data via DVC..."
    dvc pull || echo "[WARN] dvc pull failed, continuing"
    exec uvicorn src.api.main:app \
      --host 0.0.0.0 \
      --port 8000 \
      --timeout-keep-alive 300
    ;;
  ui)
    echo "Starting Streamlit"
    exec streamlit run src/ui/app.py \
      --server.port 8501 \
      --server.address 0.0.0.0 \
      --server.headless true \
      --server.enableCORS false \
      --server.enableXsrfProtection false
    ;;
  *)
    echo "Usage: entrypoint.sh [api|ui]"
    exit 1
    ;;
esac