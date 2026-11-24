"""Deterministic test doubles shared across the test suite.

- ``HashEmbedder``: a bag-of-words embedder (no model, no network).
- ``FakeLLM``: replays queued responses; matches the shape of ``LLMClient`` (Task 6).
- ``cos`` / ``norm``: tiny vector helpers for assertions.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_TOKEN_RE = re.compile(r"\w+")


def norm(v: list[float]) -> float:
    """Euclidean (L2) length of ``v``."""
    return math.sqrt(sum(x * x for x in v))


def cos(a: list[float], b: list[float]) -> float:
    """Cosine similarity of ``a`` and ``b`` (0.0 if either is a zero vector)."""
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)


def turkish_lower(text: str) -> str:
    """Lowercase with Turkish dotted/dotless I rules (``İ→i``, ``I→ı``)."""
    return text.replace("İ", "i").replace("I", "ı").lower()


class HashEmbedder:
    """Hashes each word to one of ``dim`` buckets and L2-normalizes the counts.

    Deterministic across runs (md5, not Python's salted ``hash``); texts that share
    words get a higher cosine similarity.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN_RE.findall(turkish_lower(text)):
            vec[self._bucket(token)] += 1.0
        length = norm(vec)
        return [x / length for x in vec] if length else vec

    def _bucket(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % self.dim


class FakeLLM:
    """Replays queued responses in order; records every ``complete_json`` call.

    A queued ``Exception`` is raised instead of returned. A queued model of a different
    type than the requested ``schema`` fails loudly, to catch wiring bugs.
    """

    def __init__(
        self,
        responses: list[BaseModel | Exception],
        check_error: Exception | None = None,
    ) -> None:
        self.responses = list(responses)
        self.check_error = check_error
        self.calls: list[tuple[str, str, type[BaseModel]]] = []

    def complete_json(self, system: str, user: str, schema: type[T]) -> T:
        self.calls.append((system, user, schema))
        if not self.responses:
            raise AssertionError("FakeLLM: no more responses")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if not isinstance(response, schema):
            raise AssertionError(
                f"FakeLLM: queued {type(response).__name__}, but {schema.__name__} was requested"
            )
        return response

    def check(self) -> None:
        if self.check_error is not None:
            raise self.check_error
