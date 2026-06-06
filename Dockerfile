# ─────────────────────────────────────────────────────────────────────────────
# CareerMind — Dockerfile
# Usage:
#   docker build -t careermind .
#   docker run -p 8080:8080 --gpus all careermind          # with GPU
#   docker run -p 8080:8080 careermind                     # CPU-only
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY app.py .
COPY static/ static/

# Optional: copy pre-downloaded adapter (comment out if pulling from HF Hub)
# COPY notebooks/outputs/career-coach-qlora/final-adapter/ ./notebooks/outputs/career-coach-qlora/final-adapter/

# Environment
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
# Set this to your HuggingFace repo if adapter is hosted there:
# ENV ADAPTER_PATH=your-username/career-coach-llm

EXPOSE 8080

CMD ["python", "app.py"]
