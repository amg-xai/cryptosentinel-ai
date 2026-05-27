FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build gcc g++ libssl-dev git curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --no-create-home --shell /bin/false --uid 1001 sentinel

WORKDIR /app

# Install PyTorch CPU first
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.12.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install core dependencies
RUN pip install --no-cache-dir \
    fastapi==0.115.12 \
    "uvicorn[standard]" \
    pydantic-settings \
    loguru \
    prometheus-client \
    prometheus-fastapi-instrumentator \
    opentelemetry-sdk \
    opentelemetry-api \
    opentelemetry-exporter-otlp-proto-grpc \
    requests \
    networkx \
    python-louvain \
    numpy \
    scikit-learn \
    joblib \
    "python-jose[cryptography]" \
    "passlib[bcrypt]" \
    python-multipart \
    web3 \
    confluent-kafka \
    redis \
    liboqs-python \
    pyevmasm \
    xgboost \
    plotly \
    polars \
    python-dotenv

COPY --chown=sentinel:sentinel src/ ./src/
COPY --chown=sentinel:sentinel config/ ./config/
COPY --chown=sentinel:sentinel .streamlit/ ./.streamlit/

RUN mkdir -p data/models data/processed data/visualizations && \
    chown -R sentinel:sentinel data/ && \
    chown -R sentinel:sentinel /app

USER sentinel
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# NOTE: liboqs-python compiles the PQC library on first run (~2 min).
# For production: pre-build liboqs and copy the .so file directly.
# See docs/DEPLOYMENT.md for production PQC container setup.
