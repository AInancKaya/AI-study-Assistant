"""Split page text into overlapping chunks that never cross a page boundary.

Splitting cascades from coarse to fine: paragraphs (blank lines), then sentences, then a
fixed-width window for one sentence longer than ``size``. Every chunk is a run of
consecutive pieces of a single page, so its text is a substring of that page's normalized
text and its ``page`` is the page it came from.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from study_assistant.models import Chunk, Page

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


@dataclass(frozen=True)
class _Piece:
    text: str
    sep: str  # text that precedes this piece in the page: "", " " or "\n\n"
    whole: bool  # True for a complete sentence, False for a hard-split window


def chunk_pages(pages: list[Page], size: int = 1500, overlap: int = 200) -> list[Chunk]:
    """Return chunks of at most ``size`` characters, in page order.

    Consecutive chunks on the same page share about ``overlap`` characters. ``chunk_idx``
    counts across the whole list and ``id`` is ``"{doc_id}:{page}:{chunk_idx}"``.

    Raises ValueError unless ``size > 0`` and ``0 <= overlap < size``.
    """
    _validate(size, overlap)
    chunks: list[Chunk] = []
    for page in pages:
        for text in _pack(_pieces(page.text, size, overlap), size, overlap):
            idx = len(chunks)
            chunks.append(
                Chunk(
                    id=f"{page.doc_id}:{page.page}:{idx}",
                    doc_id=page.doc_id,
                    filename=page.filename,
                    page=page.page,
                    chunk_idx=idx,
                    text=text,
                )
            )
    return chunks


def _validate(size: int, overlap: int) -> None:
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")


def _pieces(text: str, size: int, overlap: int) -> list[_Piece]:
    """Cut one page into pieces of at most ``size`` characters, in page order."""
    pieces: list[_Piece] = []
    for paragraph in _PARAGRAPH_BREAK.split(text):
        collapsed = " ".join(paragraph.split())
        if not collapsed:
            continue
        new_paragraph = True
        for sentence in _SENTENCE_END.split(collapsed):
            parts = _hard_split(sentence, size, overlap)
            for i, part in enumerate(parts):
                if not pieces or i > 0:
                    sep = ""  # first piece of the page, or contiguous window of one sentence
                elif new_paragraph:
                    sep = "\n\n"
                else:
                    sep = " "
                pieces.append(_Piece(part, sep, whole=len(parts) == 1))
                new_paragraph = False
    return pieces


def _hard_split(sentence: str, size: int, overlap: int) -> list[str]:
    """Return the sentence unchanged, or windows of ``size - overlap`` characters.

    Each window is at most ``size - overlap`` long, so a window plus the ``overlap``
    characters carried over from the previous chunk still fits in ``size``.
    """
    if len(sentence) <= size:
        return [sentence]
    step = size - overlap
    return [sentence[start : start + step] for start in range(0, len(sentence), step)]


def _pack(pieces: Sequence[_Piece], size: int, overlap: int) -> list[str]:
    """Greedily fill chunks with pieces, carrying an overlap seed into each new chunk."""
    texts: list[str] = []
    current: list[_Piece] = []
    length = 0  # running len(_render(current)), so each piece is O(1) to place
    for piece in pieces:
        if current and length + len(piece.sep) + len(piece.text) > size:
            texts.append(_render(current))
            current = _seed(current, piece, size, overlap)
            length = _length(current)
        length = length + len(piece.sep) + len(piece.text) if current else len(piece.text)
        current.append(piece)
    if current:
        texts.append(_render(current))
    return texts


def _seed(current: Sequence[_Piece], nxt: _Piece, size: int, overlap: int) -> list[_Piece]:
    """Pieces to carry from the chunk being closed into the chunk that starts with ``nxt``.

    Prefers whole trailing sentences reaching ``overlap`` characters. Otherwise falls back
    to the last characters of the closed chunk. Either way the seed leaves room for
    ``nxt``, so no chunk exceeds ``size``.
    """
    if overlap == 0:
        return []
    room = size - len(nxt.sep) - len(nxt.text)
    sentences = _trailing_sentences(current, overlap, room)
    if sentences:
        return sentences
    text = _render(current)
    keep = min(overlap, room, len(text))
    if keep <= 0:
        return []
    return [_Piece(text[len(text) - keep :], "", whole=False)]


def _trailing_sentences(current: Sequence[_Piece], overlap: int, room: int) -> list[_Piece]:
    """Shortest run of whole sentences ending ``current`` that reaches ``overlap`` and fits."""
    run: list[_Piece] = []
    for piece in reversed(current):
        if not piece.whole:
            break
        run.insert(0, piece)
        length = _length(run)
        if length > room:
            return []
        if length >= overlap:
            return run
    return []


def _render(pieces: Sequence[_Piece]) -> str:
    """Join pieces into chunk text. The first piece's separator is not part of the chunk."""
    if not pieces:
        return ""
    first, *rest = pieces
    return first.text + "".join(p.sep + p.text for p in rest)


def _length(pieces: Sequence[_Piece]) -> int:
    return len(_render(pieces))
