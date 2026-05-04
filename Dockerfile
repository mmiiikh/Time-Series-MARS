FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.docker.txt requirements.txt

RUN pip install --no-cache-dir \
    filelock fsspec networkx sympy "typing_extensions>=4.10.0" "Jinja2>=3.1" && \
    pip install --no-cache-dir \
    "torch==2.6.0+cpu" \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --no-deps

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir "dvc[s3]"

COPY . .

ENV PYTHONPATH=/app
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

ENTRYPOINT ["bash", "docker/entrypoint.sh"]
CMD ["api"]