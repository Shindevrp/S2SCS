FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock* ./
COPY models/ ./models/
COPY configs/ ./configs/
COPY app/ ./app/

RUN pip install uv && \
    uv pip install --system -e "."

ENV PYTHONUNBUFFERED=1
ENV S2SCS_CONFIG_PATH=/app/configs/config.yaml
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models

EXPOSE 8000

CMD ["python", "-m", "app.api.main"]