# Stage 1: Builder — install Python dependencies
FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --prefix=/install .

# Stage 2: Runtime (CPU-only)
FROM python:3.12-slim

# Install OpenCV system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy config (not baked into a Python package)
COPY config/ /app/config/

WORKDIR /app

# Models are mounted as a volume, not baked into the image
VOLUME /app/models

ENV WM_CONFIG_PATH=/app/config/default.yaml
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"

USER appuser

CMD ["python3", "-m", "uvicorn", "src.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
