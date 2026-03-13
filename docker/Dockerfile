FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install dvc[s3] gunicorn

COPY . .
ENV PYTHONPATH=/app
ENTRYPOINT ["bash", "docker/entrypoint.sh"]