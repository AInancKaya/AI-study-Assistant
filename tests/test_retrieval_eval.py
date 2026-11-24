"""Tests for the retrieval metrics (fake store with known rankings; no embedder)."""

from __future__ import annotations

import pytest

from evals.retrieval_eval import evaluate, first_relevant_rank
from study_assistant.models import Chunk, RetrievedChunk


def _hit(page: int, score: float) -> RetrievedChunk:
    chunk = Chunk(
        id=f"d:{page}:{page}", doc_id="d", filename="n.pdf", page=page, chunk_idx=page, text="t"
    )
    return RetrievedChunk(chunk=chunk, score=score)


class FakeStore:
    """Returns a fixed ranking per question and records the requested ``k``."""

    def __init__(self, rankings: dict[str, list[RetrievedChunk]]) -> None:
        self.rankings = rankings
        self.ks: list[int] = []

    def search(self, query: str, k: int, doc_id: str | None = None) -> list[RetrievedChunk]:
        self.ks.append(k)
        return self.rankings[query][:k]


GOLD = [
    {"question": "q1", "expected_pages": [1], "lang": "tr"},
    {"question": "q2", "expected_pages": [2], "lang": "tr"},
    {"question": "q3", "expected_pages": [3], "lang": "en"},
    {"question": "off", "expected_pages": [], "lang": "tr"},
]
RANKINGS = {
    "q1": [_hit(1, 0.90), _hit(2, 0.80)],  # rank 1
    "q2": [_hit(5, 0.88), _hit(2, 0.85)],  # rank 2
    "q3": [_hit(4, 0.86), _hit(5, 0.84)],  # not found
    "off": [_hit(4, 0.79), _hit(1, 0.70)],
}


def test_first_relevant_rank() -> None:
    hits = [_hit(5, 0.9), _hit(2, 0.8), _hit(2, 0.7)]

    assert first_relevant_rank(hits, [2, 3]) == 2
    assert first_relevant_rank(hits, [7]) is None
    assert first_relevant_rank([], [1]) is None


def test_evaluate_metrics_math() -> None:
    store = FakeStore(RANKINGS)

    result = evaluate(store, GOLD, k=6)

    assert result["n"] == 3
    assert result["k"] == 6
    assert result["hit_at_k"] == pytest.approx(2 / 3)
    assert result["mrr"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert [item["rank"] for item in result["items"]] == [1, 2, None, None]
    assert set(store.ks) == {6}


def test_suggested_min_score_is_midpoint() -> None:
    result = evaluate(FakeStore(RANKINGS), GOLD, k=6)

    # Only q1 has a correct top-1 hit (0.90); the off-topic top-1 is 0.79.
    assert result["relevant_min_score"] == pytest.approx(0.90)
    assert result["offtopic_max_score"] == pytest.approx(0.79)
    assert result["suggested_min_score"] == pytest.approx(0.845)


def test_no_clean_threshold_when_scores_overlap() -> None:
    rankings = {**RANKINGS, "off": [_hit(4, 0.95)]}

    result = evaluate(FakeStore(rankings), GOLD, k=6)

    assert result["offtopic_max_score"] == pytest.approx(0.95)
    assert result["suggested_min_score"] is None


def test_evaluate_rejects_gold_without_answerable_items() -> None:
    with pytest.raises(ValueError):
        evaluate(FakeStore(RANKINGS), [GOLD[3]], k=6)
