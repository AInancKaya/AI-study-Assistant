# Multi-stage build: the builder resolves dependencies with uv, the runtime image
# only receives the finished /app directory (venv + source) and runs as non-root.
# One image serves two compose services: the API and the Streamlit UI.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Layer 1: dependencies only. Cached until pyproject.toml or uv.lock changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# Layer 2: the project itself. README.md and LICENSE are referenced by pyproject.toml.
COPY README.md LICENSE ./
COPY src ./src
COPY ui ./ui
RUN uv sync --locked --no-dev


FROM python:3.12-slim-bookworm

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STUDY_DATA_DIR=/app/data \
    HF_HOME=/app/.cache/huggingface

RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --create-home app

WORKDIR /app

COPY --from=builder --chown=app:app /app /app

# Writable directories for the uploaded notes and the Hugging Face model cache.
# Named volumes copy ownership from these paths when they are first created.
RUN mkdir -p /app/data /app/.cache/huggingface \
    && chown -R app:app /app/data /app/.cache

USER app

EXPOSE 8000 8501

CMD ["uvicorn", "study_assistant.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
