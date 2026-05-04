#!/bin/bash
set -e

case "$1" in
  api)
    echo "Starting FastAPI"
    echo "Downloading data from S3..."
    python3 -c "
import boto3, os
from botocore.config import Config

os.makedirs('data/raw', exist_ok=True)

if os.path.exists('data/raw/mars_data.xlsx'):
    print('Data already exists, skipping download')
else:
    s3 = boto3.client(
        's3',
        endpoint_url='https://storage.yandexcloud.net',
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        region_name=os.environ.get('AWS_DEFAULT_REGION', 'ru-central1'),
        config=Config(signature_version='s3v4'),
    )
    s3.download_file(
        'mlops-mikhaylova',
        'mars/files/md5/12/7b891d3a360dc00d943b3da9a9264d',
        'data/raw/mars_data.xlsx'
    )
    print('Data downloaded successfully')
" || echo "[WARN] S3 download failed"
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