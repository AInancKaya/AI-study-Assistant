"""Text embedders: turn passages and queries into L2-normalized vectors.

The real implementation uses a local multilingual e5 model via sentence-transformers.
``sentence_transformers`` (and therefore torch) is imported lazily so that importing this
module stays cheap and the unit-test suite never loads torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "
BATCH_SIZE = 32


class Embedder(Protocol):
    """Anything that can embed passages (for indexing) and queries (for search)."""

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class E5Embedder:
    """Embedder backed by an e5 sentence-transformers model, loaded on first use.

    e5 models are trained with role prefixes, so passages get ``"passage: "`` and queries
    get ``"query: "``. Vectors are L2-normalized, so a dot product equals cosine similarity.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | Any | None = None

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode([PASSAGE_PREFIX + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([QUERY_PREFIX + text])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._get_model().encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in row] for row in vectors]

    def _get_model(self) -> SentenceTransformer | Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model
