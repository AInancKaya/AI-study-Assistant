"""End-to-end eval against a running Ollama: answer quality of ``ask`` judged by the LLM.

Usage: ``uv run python evals/live_eval.py`` (needs ``ollama serve`` and the configured
model). Every gold question goes through ``rag.ask`` on a temporary store of the sample
notes. Each answered question is then judged by the same local model (temperature 0,
structured output ``JudgeVerdict``). Writes ``evals/results/live_eval.md``.
"""

from __future__ import annotations

import math
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from pydantic import BaseModel

if __package__ in (None, ""):  # run as a script: make ``evals`` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.retrieval_eval import GoldItem, build_store, load_gold
from study_assistant.config import Settings, get_settings
from study_assistant.llm import LLMClient, LLMError, OllamaClient
from study_assistant.models import Answer
from study_assistant.prompts import format_sources
from study_assistant.rag import ask
from study_assistant.store import VectorStore

RESULT_PATH = Path(__file__).resolve().parent / "results" / "live_eval.md"

JUDGE_SYSTEM = """You are a strict examiner checking a study assistant's answer. The user \
message contains the student's question, the numbered sources from the lecture notes that \
the assistant was given, the assistant's answer, the pages it cited, and the pages where \
the answer is expected to be.

Decide two things:
- "grounded": true if every factual claim in the answer is supported by the sources. \
false if the answer adds facts that are not in the sources or contradicts them.
- "correct_pages": true if at least one cited page is one of the expected pages and \
the cited pages contain the information used in the answer; otherwise false.

Reply with JSON only: {"grounded": boolean, "correct_pages": boolean}"""


class JudgeVerdict(BaseModel):
    grounded: bool
    correct_pages: bool


@dataclass
class Record:
    """Outcome of one gold question."""

    question: str
    expected_pages: list[int]
    latency_s: float
    found: bool = False
    cited_pages: list[int] | None = None
    verdict: JudgeVerdict | None = None
    error: str | None = None

    @property
    def answerable(self) -> bool:
        return bool(self.expected_pages)

    @property
    def pages_hit(self) -> bool:
        return bool(set(self.cited_pages or []) & set(self.expected_pages))


def judge_prompt(question: str, sources: str, answer: Answer, expected_pages: list[int]) -> str:
    """User message for the judge."""
    cited = sorted({c.page for c in answer.citations})
    return (
        f"Question: {question}\n\nSources:\n{sources}\n\nAnswer: {answer.answer}\n\n"
        f"Cited pages: {cited}\nExpected pages: {expected_pages}"
    )


def run_item(item: GoldItem, store: VectorStore, llm: LLMClient, settings: Settings) -> Record:
    """Ask one gold question, time it, and judge the answer if one was given."""
    question, expected = item["question"], list(item["expected_pages"])
    start = time.perf_counter()
    try:
        answer = ask(question, store, llm, k=settings.top_k, min_score=settings.min_score)
    except LLMError as exc:
        return Record(question, expected, time.perf_counter() - start, error=str(exc))
    record = Record(
        question,
        expected,
        time.perf_counter() - start,
        found=answer.found,
        cited_pages=sorted({c.page for c in answer.citations}),
    )
    if answer.found:
        hits = store.search(question, k=settings.top_k)
        sources = format_sources([h.chunk for h in hits if h.score >= settings.min_score])
        try:
            record.verdict = llm.complete_json(
                JUDGE_SYSTEM, judge_prompt(question, sources, answer, expected), JudgeVerdict
            )
        except LLMError as exc:
            record.error = f"judge: {exc}"
    return record


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (``pct`` in 0..100) of a non-empty list."""
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def _ratio(hits: int, total: int) -> float | None:
    return hits / total if total else None


def summarize(records: list[Record]) -> dict[str, float | int | None]:
    """Aggregate metrics; ratios are None when their denominator is zero."""
    answerable = [r for r in records if r.answerable]
    offtopic = [r for r in records if not r.answerable]
    answered = [r for r in records if r.found]
    judged = [r for r in answered if r.verdict is not None]
    answered_in_scope = [r for r in answerable if r.found]
    latencies = [r.latency_s for r in records if r.error is None or r.found]
    return {
        "n_answerable": len(answerable),
        "n_offtopic": len(offtopic),
        "n_errors": sum(r.error is not None for r in records),
        "answered_rate": _ratio(len(answered_in_scope), len(answerable)),
        "grounded": _ratio(sum(r.verdict.grounded for r in judged if r.verdict), len(judged)),
        "citation_page_accuracy": _ratio(
            sum(r.pages_hit for r in answered_in_scope), len(answered_in_scope)
        ),
        "judge_correct_pages": _ratio(
            sum(r.verdict.correct_pages for r in judged if r.verdict and r.answerable),
            sum(r.answerable for r in judged),
        ),
        "not_found_accuracy": _ratio(sum(not r.found for r in offtopic), len(offtopic)),
        "latency_avg_s": sum(latencies) / len(latencies) if latencies else None,
        "latency_p95_s": percentile(latencies, 95) if latencies else None,
    }


def _pct(value: float | int | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _secs(value: float | int | None) -> str:
    return "n/a" if value is None else f"{value:.1f} s"


def render_markdown(records: list[Record], settings: Settings) -> str:
    """Summary table plus one row per gold question."""
    s = summarize(records)
    lines = [
        "# Live eval (RAG answers judged by the local LLM)",
        "",
        f"- Date: {date.today().isoformat()}",
        f"- LLM (answers and judge): `{settings.llm_model}` via Ollama, temperature 0",
        f"- Embedding model: `{settings.embedding_model}`",
        f"- top_k={settings.top_k}, min_score={settings.min_score:.2f}",
        f"- Questions: {s['n_answerable']} answerable, {s['n_offtopic']} off-topic; "
        f"errors: {s['n_errors']}",
        "",
        "| metric | value | over |",
        "| --- | --- | --- |",
        f"| answered rate | {_pct(s['answered_rate'])} | answerable questions |",
        f"| grounded (judge) | {_pct(s['grounded'])} | answered questions |",
        f"| citation-page accuracy | {_pct(s['citation_page_accuracy'])} "
        "| answered answerable questions (cited ∩ expected ≠ ∅) |",
        f"| correct pages (judge) | {_pct(s['judge_correct_pages'])} "
        "| answered answerable questions |",
        f"| not-found accuracy | {_pct(s['not_found_accuracy'])} | off-topic questions |",
        f"| avg latency of `ask` | {_secs(s['latency_avg_s'])} | all questions |",
        f"| p95 latency of `ask` | {_secs(s['latency_p95_s'])} | all questions |",
        "",
        "| # | question | expected | found | cited | grounded | pages ok | latency | error |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for n, r in enumerate(records, start=1):
        expected = ", ".join(map(str, r.expected_pages)) or "off-topic"
        cited = ", ".join(map(str, r.cited_pages or [])) or "-"
        grounded = "-" if r.verdict is None else ("yes" if r.verdict.grounded else "no")
        pages_ok = "-" if not (r.found and r.answerable) else ("yes" if r.pages_hit else "no")
        lines.append(
            f"| {n} | {r.question} | {expected} | {'yes' if r.found else 'no'} | {cited} "
            f"| {grounded} | {pages_ok} | {r.latency_s:.1f} s | {r.error or ''} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    settings = get_settings()
    llm = OllamaClient(settings.ollama_host, settings.llm_model, settings.llm_timeout_s)
    try:
        llm.check()
    except LLMError as exc:
        print(exc, file=sys.stderr)
        return 1
    gold = load_gold()
    with tempfile.TemporaryDirectory() as tmp:
        store = build_store(Path(tmp) / "chroma", settings)
        records = []
        for n, item in enumerate(gold, start=1):
            records.append(run_item(item, store, llm, settings))
            print(f"[{n}/{len(gold)}] {records[-1].latency_s:.1f}s {item['question']}", flush=True)
    report = render_markdown(records, settings)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
