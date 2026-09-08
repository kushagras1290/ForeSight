# syntax=docker/dockerfile:1
#
# Project FORESIGHT — single deployable image.
#
# Builds the dashboard, installs the Python service, and (by default) runs the
# analysis so the image ships with a scored plan inside it. One container serves
# both the API and the dashboard from the same origin, which is why the frontend
# needs no environment-specific API URL.
#
#   docker build -t foresight .
#   docker run --rm -p 8000:8000 foresight
#
# The default build uses the LightGBM model only, which keeps build time to a
# few minutes. To bake in the adaptive ensemble instead (~10 minutes):
#
#   docker build --build-arg FULL_TRAIN=true -t foresight .
#
# To supply your own extracts rather than generating them, mount them over
# /app/data/raw and run scripts/06_refresh.py inside the container.

# --------------------------------------------------------------------------- #
# Stage 1 — dashboard
# --------------------------------------------------------------------------- #
FROM node:22-alpine AS dashboard

WORKDIR /build

# Dependencies first: this layer is cached unless the lockfile changes.
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY dashboard/ ./
RUN npm run build


# --------------------------------------------------------------------------- #
# Stage 2 — service
# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS runtime

# PYTHONDONTWRITEBYTECODE: no .pyc in a read-only-ish container.
# PYTHONUNBUFFERED: logs reach the collector immediately rather than on flush.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FORESIGHT_LOG_FORMAT=json

# libgomp1 is LightGBM's OpenMP runtime — it will not import without it.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY service/ ./service/
COPY scripts/ ./scripts/
RUN pip install -e . --no-deps

COPY --from=dashboard /build/dist ./dashboard/dist

# --- Bake in a scored plan ------------------------------------------------- #
ARG FULL_TRAIN=false
RUN mkdir -p data/raw data/interim data/processed artifacts reports/figures \
    && python scripts/00_generate_data.py \
    && python scripts/01_run_pipeline.py \
    && if [ "$FULL_TRAIN" = "true" ]; then \
         python scripts/03_train_backtest.py && python scripts/04_score_risk.py ; \
       else \
         python scripts/03_train_backtest.py --skip-ensemble \
         && python scripts/04_score_risk.py --model gbm ; \
       fi

# --- Run unprivileged ------------------------------------------------------- #
# A web-facing process has no reason to be root. Ownership is set after the
# build steps so the earlier layers stay cacheable.
RUN useradd --create-home --uid 10001 foresight \
    && chown -R foresight:foresight /app
USER foresight

EXPOSE 8000

# Readiness, not liveness: the process can be up while having nothing to serve,
# and an orchestrator should not route traffic to it in that state.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/ready || exit 1

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
