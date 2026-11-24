MODEL ?= gemma2:9b

.PHONY: setup api ui run start test lint eval live-eval

setup:  ## Install dependencies and pull the local LLM
	uv sync
	ollama pull $(MODEL)

api:  ## Start the FastAPI backend on :8000
	uv run uvicorn study_assistant.api.main:create_app --factory --reload --port 8000

ui:  ## Start the Streamlit UI
	uv run streamlit run ui/app.py

run:  ## Start API and UI together (Ctrl+C stops both)
	$(MAKE) -j2 api ui

start:  ## Check Ollama, then run API + UI
	bash scripts/run_local.sh

test:  ## Unit tests (no network, no Ollama, no real embedder)
	uv run pytest

lint:  ## Lint and format check
	uv run ruff check .
	uv run ruff format --check .

eval:  ## Offline retrieval eval (real embedder, no LLM) -> evals/results/retrieval.md
	uv run pytest -m slow --no-cov
	uv run python evals/retrieval_eval.py

live-eval:  ## End-to-end eval against a running Ollama -> evals/results/live_eval.md
	uv run pytest -m live --no-cov
	uv run python evals/live_eval.py
