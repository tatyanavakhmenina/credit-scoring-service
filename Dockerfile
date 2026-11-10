FROM python:3.12-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_COMPILE_BYTECODE=1
WORKDIR /app
# Сначала только зависимости: при правке кода этот тяжёлый слой берётся из кэша.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app
ARG GIT_SHA=local
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 GIT_SHA=$GIT_SHA \
    MLFLOW_DISABLE_AGENT_HINT=1
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv ./.venv
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=5 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://localhost:8000/healthz', timeout=2)"]
CMD ["uvicorn", "credit_scoring.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
