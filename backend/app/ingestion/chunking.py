"""Overlapping, boundary-aware chunking that never crosses page boundaries.

Keeping each chunk within a single page means every citation maps to exactly one
page number.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from app.ingestion.extraction import PageText


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    page_number: int | None
    char_start: int  # offset within the page text
    char_end: int
    content: str
    content_hash: str
    token_estimate: int


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n\n")


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 characters per token for English text)."""
    return max(1, round(len(text) / 4))


def _find_break(text: str, start: int, hard_end: int, min_end: int) -> int:
    """Pick the best split position in ``text[min_end:hard_end]``."""
    if hard_end >= len(text):
        return len(text)
    window = text[min_end:hard_end]
    for pattern in ("\n\n", ". ", "? ", "! ", "; ", ", ", " "):
        pos = window.rfind(pattern)
        if pos != -1:
            return min_end + pos + len(pattern)
    return hard_end


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    """Return ``(start, end)`` spans covering ``text`` with the requested overlap."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
    spans: list[tuple[int, int]] = []
    start, n = 0, len(text)
    while start < n:
        while start < n and text[start].isspace():
            start += 1
        if start >= n:
            break
        hard_end = min(start + chunk_size, n)
        end = _find_break(text, start, hard_end, start + chunk_size // 2)
        spans.append((start, end))
        if end >= n:
            break
        next_start = max(end - overlap, start + 1)
        # Start the overlap on a word boundary to avoid cut-off words.
        space = text.find(" ", next_start, end)
        start = space + 1 if space != -1 and overlap > 0 else next_start
    return spans


def chunk_pages(pages: list[PageText], *, chunk_size: int, overlap: int, min_chars: int) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    seen_hashes: set[str] = set()
    for page in pages:
        for start, end in chunk_text(page.text, chunk_size, overlap):
            content = page.text[start:end].strip()
            if len(content) < min_chars and not (len(page.text.strip()) < min_chars and content):
                continue
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if digest in seen_hashes:  # exact duplicate text (e.g., repeated boilerplate)
                continue
            seen_hashes.add(digest)
            chunks.append(
                TextChunk(
                    chunk_index=len(chunks),
                    page_number=page.page_number,
                    char_start=start,
                    char_end=max(end, start + 1),
                    content=content,
                    content_hash=digest,
                    token_estimate=estimate_tokens(content),
                )
            )
    return chunks
