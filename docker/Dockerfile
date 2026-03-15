FROM python:3.10-slim

WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install dvc[s3] gunicorn

COPY . .
ENV PYTHONPATH=/app
#CMD ["python", "docker/entrypoint.sh"]
ENTRYPOINT ["bash", "docker/entrypoint.sh"]