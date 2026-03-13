#!/bin/bash
set -e

case "$1" in
  api)
    exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
    ;;
  ui)
    exec streamlit run src/ui/app.py --server.port 8501 --server.address 0.0.0.0
    ;;
  *)
    echo "Укажи команду: api или ui"
    exit 1
    ;;
esac
