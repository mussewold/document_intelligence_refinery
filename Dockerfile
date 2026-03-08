# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsqlite3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install uv

# Copy uv dependency files
COPY pyproject.toml ./
# If uv.lock exists, uncomment:
# COPY uv.lock ./

# Install dependencies into system environment (or create virtualenv if preferred)
RUN uv pip install --system -r pyproject.toml

# Copy project files
COPY frontend ./frontend
COPY src ./src
COPY data ./data

# Expose FastAPI default port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
