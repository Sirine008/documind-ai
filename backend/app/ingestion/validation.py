"""Upload validation.

Files are validated by inspecting their bytes (magic numbers, decodability, PDF
structure), never by trusting the client-supplied filename or MIME type alone.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import PurePath

import pymupdf as fitz

from app.core.errors import PayloadTooLarge, UnsupportedMediaType, ValidationFailed

ALLOWED_EXTENSIONS = {".pdf": "pdf", ".txt": "text", ".md": "text", ".text": "text"}
PDF_MAGIC = b"%PDF-"


@dataclass(frozen=True)
class ValidatedFile:
    kind: str  # "pdf" | "text"
    sha256: str
    size_bytes: int
    safe_filename: str
    title: str
    page_count: int | None


def sanitize_filename(filename: str | None, max_length: int = 200) -> str:
    """Return a filesystem- and display-safe version of a client filename."""
    name = PurePath((filename or "").replace("\\", "/")).name
    name = unicodedata.normalize("NFKC", name)
    name = "".join(ch for ch in name if ch.isprintable())
    name = re.sub(r"[^\w.\- ()]+", "_", name, flags=re.UNICODE).strip(" ._")
    if not name:
        name = "document"
    if len(name) > max_length:
        stem, dot, ext = name.rpartition(".")
        name = (stem[: max_length - len(ext) - 1] + "." + ext) if dot and len(ext) <= 8 else name[:max_length]
    return name


def title_from_filename(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    title = re.sub(r"[_\-]+", " ", stem).strip()
    return (title or "Untitled document")[:255]


def _looks_like_text(data: bytes) -> bool:
    if b"\x00" in data[:8192]:
        return False
    try:
        sample = data[:65536].decode("utf-8")
    except UnicodeDecodeError as exc:
        # A multi-byte character may be cut at the sample boundary.
        if exc.start < len(data[:65536]) - 4:
            return False
        sample = data[: exc.start].decode("utf-8")
    if not sample:
        return False
    control = sum(1 for ch in sample if unicodedata.category(ch) == "Cc" and ch not in "\n\r\t\f")
    return control / len(sample) < 0.01


def validate_upload(data: bytes, filename: str | None, *, max_bytes: int, max_pages: int) -> ValidatedFile:
    """Validate raw upload bytes and return metadata, or raise a descriptive error."""
    if len(data) == 0:
        raise ValidationFailed("The uploaded file is empty.", code="empty_file")
    if len(data) > max_bytes:
        raise PayloadTooLarge(
            f"The file exceeds the {max_bytes // (1024 * 1024)} MB upload limit.", code="file_too_large"
        )

    safe_name = sanitize_filename(filename)
    ext = PurePath(safe_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedMediaType(
            "Unsupported file type. Upload a PDF (.pdf) or plain-text (.txt, .md) file.",
            code="unsupported_type",
        )
    declared = ALLOWED_EXTENSIONS[ext]
    is_pdf_bytes = data.lstrip()[:5] == PDF_MAGIC

    page_count: int | None = None
    if declared == "pdf":
        if not is_pdf_bytes:
            raise UnsupportedMediaType(
                "The file has a .pdf extension but its contents are not a PDF.", code="content_mismatch"
            )
        page_count = _inspect_pdf(data, max_pages)
    else:
        if is_pdf_bytes:
            raise UnsupportedMediaType(
                "The file contents are a PDF but the extension is not .pdf.", code="content_mismatch"
            )
        if not _looks_like_text(data):
            raise UnsupportedMediaType("The file is not valid UTF-8 plain text.", code="invalid_text_encoding")
        if not data.decode("utf-8", errors="ignore").strip():
            raise ValidationFailed("The text file contains no readable text.", code="empty_file")

    return ValidatedFile(
        kind=declared,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        safe_filename=safe_name,
        title=title_from_filename(safe_name),
        page_count=page_count,
    )


def _inspect_pdf(data: bytes, max_pages: int) -> int:
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ValidationFailed("The PDF is corrupt or cannot be opened.", code="corrupt_pdf") from exc
    try:
        if doc.needs_pass or doc.is_encrypted:
            raise ValidationFailed(
                "The PDF is password-protected. Remove the password and upload it again.",
                code="encrypted_pdf",
            )
        pages = doc.page_count
        if pages == 0:
            raise ValidationFailed("The PDF contains no pages.", code="empty_file")
        if pages > max_pages:
            raise PayloadTooLarge(f"The PDF has {pages} pages; the limit is {max_pages}.", code="too_many_pages")
        return pages
    finally:
        doc.close()
