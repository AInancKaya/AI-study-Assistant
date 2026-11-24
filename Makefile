MODEL ?= gemma2:9b

.PHONY: setup api ui run test lint eval live-eval

setup:  ## Install dependencies and pull the local LLM
	uv sync
	ollama pull $(MODEL)

api:  ## Start the FastAPI backend on :8000
	uv run uvicorn study_assistant.api.main:create_app --factory --reload --port 8000

ui:  ## Start the Streamlit UI
	uv run streamlit run ui/app.py

run:  ## Start API and UI together (Ctrl+C stops both)
	$(MAKE) -j2 api ui

test:  ## Unit tests (no network, no Ollama, no real embedder)
	uv run pytest

lint:  ## Lint and format check
	uv run ruff check .
	uv run ruff format --check .

eval:  ## Offline retrieval eval (real embedder, no LLM)
	uv run pytest -m slow --no-cov

live-eval:  ## End-to-end eval against a running Ollama
	uv run pytest -m live --no-cov
