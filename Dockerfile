# Container used by Railway to run the Streamlit dashboard.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU build (much smaller than the default CUDA build)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-railway.txt .
RUN pip install -r requirements-railway.txt

COPY . .

# The fine-tuned DistilBERT weights (268 MB) are stored with Git LFS. If the build only got the small
# LFS pointer file instead of the real weights, download the real file from GitHub's LFS media URL.
ARG DISTILBERT_URL=https://media.githubusercontent.com/media/Bhuan2124/ecommerce-sentiment-analysis/main/models/distilbert/model.safetensors
RUN f=models/distilbert/model.safetensors; \
    if [ ! -f "$f" ] || [ "$(stat -c %s "$f")" -lt 1000000 ]; then \
      curl -fL --retry 5 -o "$f" "$DISTILBERT_URL"; \
    fi && ls -lh models/distilbert

EXPOSE 8080
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
