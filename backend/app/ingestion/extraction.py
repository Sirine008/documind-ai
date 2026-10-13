"""Text extraction that preserves page boundaries, plus normalization."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import pymupdf as fitz


@dataclass(frozen=True)
class PageText:
    page_number: int | None  # 1-based for PDFs; None for plain text
    text: str


class ExtractionError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}


_HEADING_MAX_CHARS = 60


def _separate_headings(text: str) -> str:
    """Turn a short line without closing punctuation that is followed by a capitalized
    line (a heading, title or footer) into its own paragraph, so it is not glued onto
    the next sentence when soft line breaks are joined."""
    lines = text.split("\n")
    out: list[str] = []
    for i, line in enumerate(lines):
        out.append(line)
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if line and nxt and len(line) <= _HEADING_MAX_CHARS and line[-1] not in ".,;:!?-\u2014" and nxt[:1].isupper():
            out.append("")
    return "\n".join(out)


def normalize_text(text: str) -> str:
    """Clean common extraction artifacts while keeping paragraph structure."""
    text = unicodedata.normalize("NFKC", text)
    for lig, rep in _LIGATURES.items():
        text = text.replace(lig, rep)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"[​-‍﻿]", "", text)
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch) != "Cc")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate words split across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = _separate_headings(text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # join soft line breaks inside paragraphs
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(data: bytes) -> list[PageText]:
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ExtractionError("corrupt_pdf", "The PDF is corrupt or cannot be opened.") from exc
    try:
        if doc.needs_pass:
            raise ExtractionError("encrypted_pdf", "The PDF is password-protected.")
        pages = []
        for index, page in enumerate(doc):
            pages.append(PageText(page_number=index + 1, text=normalize_text(page.get_text("text"))))
        return pages
    finally:
        doc.close()


def extract_text_file(data: bytes) -> list[PageText]:
    return [PageText(page_number=None, text=normalize_text(data.decode("utf-8", errors="replace")))]


def extract(kind: str, data: bytes) -> list[PageText]:
    if kind == "pdf":
        return extract_pdf(data)
    if kind == "text":
        return extract_text_file(data)
    raise ExtractionError("unsupported_type", f"Unsupported document kind: {kind}")
