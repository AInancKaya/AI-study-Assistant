"""Offline retrieval evaluation: hit@k and MRR of the real embedder on the sample notes.

Usage: ``uv run python evals/retrieval_eval.py`` (no LLM needed). Builds a temporary
vector store from ``evals/data/sample_notes.pdf`` with the configured embedding model,
runs every question of ``evals/data/gold.jsonl`` through ``VectorStore.search`` and writes
``evals/results/retrieval.md``. Exits 1 if hit@k is below ``MIN_HIT_AT_K``.
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

from study_assistant.config import Settings, get_settings
from study_assistant.embeddings import E5Embedder
from study_assistant.ingest.chunking import chunk_pages
from study_assistant.ingest.pdf import extract_pages
from study_assistant.models import RetrievedChunk
from study_assistant.store import VectorStore

EVALS_DIR = Path(__file__).resolve().parent
NOTES_PDF = EVALS_DIR / "data" / "sample_notes.pdf"
GOLD_PATH = EVALS_DIR / "data" / "gold.jsonl"
RESULT_PATH = EVALS_DIR / "results" / "retrieval.md"
MIN_HIT_AT_K = 0.8

GoldItem = dict[str, Any]


def load_gold(path: Path = GOLD_PATH) -> list[GoldItem]:
    """Read one JSON object per non-empty line."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def build_store(path: Path, settings: Settings) -> VectorStore:
    """Index the sample notes with the real embedder, chunked with the configured settings."""
    pages = extract_pages(NOTES_PDF.read_bytes(), NOTES_PDF.name)
    chunks = chunk_pages(pages, size=settings.chunk_size, overlap=settings.chunk_overlap)
    store = VectorStore(path, E5Embedder(settings.embedding_model))
    store.add_document(chunks, pages=len(pages))
    return store


def first_relevant_rank(hits: list[RetrievedChunk], expected_pages: list[int]) -> int | None:
    """1-based position of the first hit on an expected page, or None if there is none."""
    for position, hit in enumerate(hits, start=1):
        if hit.chunk.page in expected_pages:
            return position
    return None


def evaluate(store: VectorStore, gold: list[GoldItem], k: int) -> dict[str, Any]:
    """Score retrieval on ``gold``.

    Answerable items (non-empty ``expected_pages``) give ``hit_at_k`` and ``mrr``.
    ``relevant_min_score`` is the lowest top-1 score among answerable items whose top-1
    hit is on an expected page; ``offtopic_max_score`` is the highest top-1 score among
    off-topic items. ``suggested_min_score`` is their midpoint, or None when they overlap.
    Raises ValueError if ``gold`` has no answerable item.
    """
    items = [_score_item(store, item, k) for item in gold]
    answerable = [i for i in items if i["expected_pages"]]
    if not answerable:
        raise ValueError("gold set has no answerable items")
    ranks = [i["rank"] for i in answerable]
    relevant = [i["top1_score"] for i in answerable if i["rank"] == 1]
    offtopic = [i["top1_score"] for i in items if not i["expected_pages"]]
    relevant_min = min(relevant) if relevant else None
    offtopic_max = max(offtopic) if offtopic else None
    return {
        "k": k,
        "n": len(answerable),
        "hit_at_k": sum(r is not None for r in ranks) / len(ranks),
        "mrr": sum(1 / r for r in ranks if r is not None) / len(ranks),
        "relevant_min_score": relevant_min,
        "offtopic_max_score": offtopic_max,
        "suggested_min_score": _midpoint(relevant_min, offtopic_max),
        "items": items,
    }


def _score_item(store: VectorStore, item: GoldItem, k: int) -> dict[str, Any]:
    hits = store.search(item["question"], k=k)
    expected = list(item["expected_pages"])
    return {
        "question": item["question"],
        "lang": item["lang"],
        "expected_pages": expected,
        "rank": first_relevant_rank(hits, expected) if expected else None,
        "top1_page": hits[0].chunk.page if hits else None,
        "top1_score": hits[0].score if hits else 0.0,
    }


def _midpoint(relevant_min: float | None, offtopic_max: float | None) -> float | None:
    if relevant_min is None or offtopic_max is None or offtopic_max >= relevant_min:
        return None
    return (relevant_min + offtopic_max) / 2


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_markdown(result: dict[str, Any], settings: Settings) -> str:
    """Summary table plus one row per gold question."""
    k = result["k"]
    suggested = result["suggested_min_score"]
    threshold = "no clean threshold" if suggested is None else f"{suggested:.3f}"
    lines = [
        "# Retrieval eval",
        "",
        f"- Date: {date.today().isoformat()}",
        f"- Embedding model: `{settings.embedding_model}`",
        f"- Notes: `evals/data/sample_notes.pdf`, chunk_size={settings.chunk_size}, "
        f"chunk_overlap={settings.chunk_overlap}, k={k}",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| answerable questions (n) | {result['n']} |",
        f"| hit@{k} | {result['hit_at_k']:.3f} |",
        f"| MRR | {result['mrr']:.3f} |",
        f"| lowest correct top-1 score | {_fmt(result['relevant_min_score'])} |",
        f"| highest off-topic top-1 score | {_fmt(result['offtopic_max_score'])} |",
        f"| suggested min_score | {threshold} |",
        f"| configured min_score | {settings.min_score:.2f} |",
        "",
        "| # | lang | question | expected | rank | top-1 page | top-1 score |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for n, item in enumerate(result["items"], start=1):
        expected = ", ".join(map(str, item["expected_pages"])) or "off-topic"
        rank = item["rank"] if item["rank"] is not None else "-"
        lines.append(
            f"| {n} | {item['lang']} | {item['question']} | {expected} | {rank} "
            f"| {item['top1_page']} | {item['top1_score']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    settings = get_settings()
    gold = load_gold()
    with tempfile.TemporaryDirectory() as tmp:
        store = build_store(Path(tmp) / "chroma", settings)
        result = evaluate(store, gold, k=settings.top_k)
    report = render_markdown(result, settings)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(report, encoding="utf-8")
    print(report)
    if result["hit_at_k"] < MIN_HIT_AT_K:
        print(f"FAIL: hit@{result['k']} {result['hit_at_k']:.3f} < {MIN_HIT_AT_K}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
