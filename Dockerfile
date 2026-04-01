# Stage 1: Builder — install Python dependencies
FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml .
RUN pip install --no-cache-dir --prefix=/install .

# Stage 2: Runtime (CPU-only)
FROM python:3.12-slim

# Install OpenCV system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ /app/src/
COPY config/ /app/config/

WORKDIR /app

# Models are mounted as a volume, not baked into the image
VOLUME /app/models

ENV WM_CONFIG_PATH=/app/config/default.yaml
EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "src.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
