"""Tests for the live-eval aggregation (no Ollama: records are built by hand)."""

import pytest

from evals.live_eval import JudgeVerdict, Record, percentile, summarize


def _verdict(grounded: bool, pages: bool = True) -> JudgeVerdict:
    return JudgeVerdict(grounded=grounded, correct_pages=pages)


def test_percentile_nearest_rank() -> None:
    values = [float(v) for v in range(1, 21)]

    assert percentile(values, 95) == 19.0
    assert percentile(values, 100) == 20.0
    assert percentile([3.0], 95) == 3.0


def test_summarize_metrics() -> None:
    records = [
        Record("a", [1], 2.0, found=True, cited_pages=[1], verdict=_verdict(True)),
        Record("b", [2], 4.0, found=True, cited_pages=[3], verdict=_verdict(False, False)),
        Record("c", [3], 1.0, found=False, cited_pages=[]),
        Record("d", [4], 9.0, error="timeout"),
        Record("off1", [], 0.5, found=False, cited_pages=[]),
        Record("off2", [], 3.0, found=True, cited_pages=[5], verdict=_verdict(True, False)),
    ]

    s = summarize(records)

    assert s["answered_rate"] == pytest.approx(2 / 4)
    assert s["grounded"] == pytest.approx(2 / 3)
    assert s["citation_page_accuracy"] == pytest.approx(1 / 2)
    assert s["judge_correct_pages"] == pytest.approx(1 / 2)
    assert s["not_found_accuracy"] == pytest.approx(1 / 2)
    assert s["n_errors"] == 1
    assert s["latency_avg_s"] == pytest.approx((2.0 + 4.0 + 1.0 + 0.5 + 3.0) / 5)
    assert s["latency_p95_s"] == 4.0
