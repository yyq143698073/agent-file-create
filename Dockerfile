# ── agent-file-create Docker image ────────────────────────────────────────
# Build:  docker build -t agent-file-create .
# Run:    docker run -p 8000:8000 --env-file .env agent-file-create
#
# For full stack with Ollama, use docker-compose.yml instead.

FROM python:3.11-slim

WORKDIR /app

# System dependencies for OCR and PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Create result directory
RUN mkdir -p result

EXPOSE 8000

# Health check — pings /api/health every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Default: start web server (override CMD for CLI mode)
CMD ["python", "-m", "agent_file_create.web"]
