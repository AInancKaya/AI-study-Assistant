"""Generate text-layer PDFs for tests from a markdown-like source file.

Usage: uv run python scripts/make_pdf.py in.md out.pdf

Pages in the input are separated by a line that reads exactly ``---page---``.
Each source line is drawn on its own line of the PDF, so line breaks in the
source (for example a hyphenated word split across two lines) are preserved.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

PAGE_SEPARATOR = "---page---"
FONT_NAME = "UnicodeSans"
FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
)
FONT_SIZE = 12
LEADING = 16
MARGIN = 72
WRAP_WIDTH = 90


def _register_font() -> None:
    """Register the first available Unicode TTF font under ``FONT_NAME``."""
    for path in FONT_CANDIDATES:
        if path.exists():
            pdfmetrics.registerFont(TTFont(FONT_NAME, str(path)))
            return
    raise FileNotFoundError(f"No Unicode TTF font found among: {FONT_CANDIDATES}")


def _wrap_lines(page: str) -> list[str]:
    """Split a page into drawable lines, wrapping long source lines at WRAP_WIDTH."""
    lines: list[str] = []
    for raw in page.splitlines():
        lines.extend(textwrap.wrap(raw, WRAP_WIDTH, break_long_words=False, break_on_hyphens=False))
    return lines


def make_pdf(pages: list[str], out: Path) -> None:
    """Write one PDF page per entry in ``pages``; an empty string gives a blank page."""
    _register_font()
    width, height = A4
    top = height - MARGIN
    c = canvas.Canvas(str(out), pagesize=A4)
    for page in pages:
        c.setFont(FONT_NAME, FONT_SIZE)
        y = top
        for line in _wrap_lines(page):
            if y < MARGIN:
                c.showPage()
                c.setFont(FONT_NAME, FONT_SIZE)
                y = top
            c.drawString(MARGIN, y, line)
            y -= LEADING
        c.showPage()
    c.save()


def split_pages(source: str) -> list[str]:
    """Split markdown-like source text into pages on ``---page---`` lines."""
    pages: list[str] = []
    current: list[str] = []
    for line in source.splitlines():
        if line.strip() == PAGE_SEPARATOR:
            pages.append("\n".join(current))
            current = []
        else:
            current.append(line)
    pages.append("\n".join(current))
    return pages


def main(argv: list[str]) -> int:
    """Command line entry point: ``in.md out.pdf``."""
    if len(argv) != 2:
        print("usage: make_pdf.py in.md out.pdf", file=sys.stderr)
        return 2
    source = Path(argv[0]).read_text(encoding="utf-8")
    make_pdf(split_pages(source), Path(argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
