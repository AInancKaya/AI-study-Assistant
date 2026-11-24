"""Tests for embedders and the shared test fakes.

Only ``test_e5_prefixes_and_dim`` touches the real model (marked ``slow``).
"""

import subprocess
import sys

import pytest

from study_assistant.embeddings import E5Embedder
from study_assistant.models import LLMAnswer, LLMGrade
from tests.fakes import FakeLLM, HashEmbedder, cos, norm

# --- HashEmbedder -----------------------------------------------------------


def test_hash_embedder_similarity():
    e = HashEmbedder()
    a, b, c = e.embed_passages(
        ["fotosentez bitkiler", "bitkiler fotosentez yapar", "borsa endeksi"]
    )
    assert cos(a, b) > cos(a, c)


def test_hash_embedder_normalized():
    assert abs(norm(HashEmbedder().embed_query("x y")) - 1) < 1e-6


def test_hash_embedder_deterministic_and_dim():
    a = HashEmbedder(dim=64).embed_query("Fotosentez nedir?")
    b = HashEmbedder(dim=64).embed_query("Fotosentez nedir?")
    assert a == b
    assert len(a) == 64


def test_hash_embedder_turkish_case_folding():
    e = HashEmbedder()
    # Turkish-aware: "İ" -> "i" and "I" -> "ı".
    assert e.embed_query("İSTANBUL") == e.embed_query("istanbul")
    assert e.embed_query("ISIK") == e.embed_query("ısık")
    assert e.embed_query("ISIK") != e.embed_query("isik")


def test_hash_embedder_empty_text_is_zero_vector():
    v = HashEmbedder(dim=8).embed_query("   ")
    assert v == [0.0] * 8


# --- FakeLLM -----------------------------------------------------------------


def test_fake_llm_returns_in_order_and_records_calls():
    first = LLMAnswer(answer="a", found=True, cited_sources=[1])
    second = LLMAnswer(answer="b", found=False, cited_sources=[])
    llm = FakeLLM([first, second])
    assert llm.complete_json("sys", "u1", LLMAnswer) is first
    assert llm.complete_json("sys", "u2", LLMAnswer) is second
    assert llm.calls == [("sys", "u1", LLMAnswer), ("sys", "u2", LLMAnswer)]


def test_fake_llm_raises_queued_exception_and_records_call():
    llm = FakeLLM([ValueError("bad json")])
    with pytest.raises(ValueError, match="bad json"):
        llm.complete_json("s", "u", LLMAnswer)
    assert llm.calls == [("s", "u", LLMAnswer)]


def test_fake_llm_empty_queue():
    with pytest.raises(AssertionError, match="FakeLLM: no more responses"):
        FakeLLM([]).complete_json("s", "u", LLMAnswer)


def test_fake_llm_schema_mismatch():
    llm = FakeLLM([LLMGrade(score=1, feedback="ok")])
    with pytest.raises(AssertionError):
        llm.complete_json("s", "u", LLMAnswer)


def test_fake_llm_check():
    assert FakeLLM([]).check() is None
    with pytest.raises(ConnectionError):
        FakeLLM([], check_error=ConnectionError("down")).check()


# --- E5Embedder without the real model --------------------------------------


class _StubModel:
    """Records encode() calls and returns a fixed-length vector per input."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def encode(self, sentences: list[str], **kwargs: object) -> list[list[float]]:
        self.calls.append((list(sentences), kwargs))
        return [[float(i), 1.0] for i in range(len(sentences))]


def _stubbed(embedder: E5Embedder) -> _StubModel:
    stub = _StubModel()
    embedder._model = stub
    return stub


def test_e5_empty_passages_do_not_load_model():
    e = E5Embedder("does/not-exist")
    assert e.embed_passages([]) == []
    assert e._model is None


def test_e5_passage_prefix_and_encode_options():
    e = E5Embedder("any")
    stub = _stubbed(e)
    out = e.embed_passages(["bir", "iki"])
    texts, kwargs = stub.calls[0]
    assert texts == ["passage: bir", "passage: iki"]
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["batch_size"] == 32
    assert kwargs["show_progress_bar"] is False
    assert out == [[0.0, 1.0], [1.0, 1.0]]
    assert all(type(x) is float for row in out for x in row)


def test_e5_query_prefix():
    e = E5Embedder("any")
    stub = _stubbed(e)
    assert e.embed_query("Fotosentez nedir?") == [0.0, 1.0]
    assert stub.calls[0][0] == ["query: Fotosentez nedir?"]


def test_importing_embeddings_does_not_import_torch():
    code = (
        "import sys, study_assistant.embeddings; "
        "assert 'sentence_transformers' not in sys.modules; "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# --- Real model (slow) -------------------------------------------------------


@pytest.mark.slow
def test_e5_prefixes_and_dim():
    e = E5Embedder("intfloat/multilingual-e5-small")
    q = e.embed_query("Fotosentez nedir?")
    p = e.embed_passages(
        ["Fotosentez, bitkilerin ışıkla besin üretmesidir.", "Enflasyon fiyatların artmasıdır."]
    )
    assert len(q) == 384 and cos(q, p[0]) > cos(q, p[1])
    assert abs(norm(q) - 1) < 1e-4
