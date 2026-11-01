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

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# The fine-tuned DistilBERT weights (268 MB) are bigger than GitHub's 100 MB file limit, so they are
# stored as 45 MB pieces in models/distilbert/parts/ and joined back together here.
RUN cd models/distilbert && \
    if [ ! -f model.safetensors ]; then cat parts/model.safetensors.part* > model.safetensors; fi && \
    echo "$(cat parts/SHA256)  model.safetensors" | sha256sum -c - && \
    rm -rf parts && ls -lh

EXPOSE 8080
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
