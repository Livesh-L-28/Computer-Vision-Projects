# Multi-stage production Dockerfile for Computer Vision Intelligence Suite
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies for OpenCV and building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final Production Stage
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime dependencies for OpenCV, GUI and media processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy codebase
COPY . .

# Create outputs and models directories
RUN mkdir -p /app/outputs /app/models

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/status || exit 1

ENTRYPOINT ["python", "main.py", "--serve", "--host", "0.0.0.0", "--port", "8080"]
