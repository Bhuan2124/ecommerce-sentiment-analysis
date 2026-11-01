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

# The fine-tuned DistilBERT weights (268 MB) are too big for a normal GitHub file,
# so they are kept as a GitHub Release asset and downloaded while the image is built.
ARG DISTILBERT_URL=https://github.com/saud142/ecommerce-sentiment-analysis/releases/download/model-v1/model.safetensors
RUN if [ ! -f models/distilbert/model.safetensors ]; then \
      curl -fL --retry 5 -o models/distilbert/model.safetensors "$DISTILBERT_URL"; \
    fi && ls -lh models/distilbert

EXPOSE 8080
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
