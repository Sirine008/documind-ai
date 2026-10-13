"""Regenerate the sample documents in data/sample_documents/ from scripts/sample_content.py.

Usage:  python scripts/generate_sample_docs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_content import ALL_DOCUMENTS

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_documents"


def write_pdf(path: Path, title: str, pages: list[str]) -> None:
    doc = pymupdf.open()
    for number, text in enumerate(pages, start=1):
        page = doc.new_page(width=595, height=842)  # A4 in points
        rect = pymupdf.Rect(60, 60, 535, 790)
        leftover = page.insert_textbox(rect, text, fontsize=11, fontname="helv", lineheight=1.35)
        if leftover < 0:
            raise RuntimeError(f"Text overflowed page {number} of {path.name}")
        page.insert_text((60, 815), f"{title} - page {number}", fontsize=8, fontname="helv")
    doc.set_metadata({"title": title, "author": "DocuMind AI sample data (CC0)"})
    # Fixed IDs/dates are not settable in all versions; garbage/deflate keep output compact.
    doc.save(path, garbage=4, deflate=True)
    doc.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for item in ALL_DOCUMENTS:
        path = OUT_DIR / item["filename"]
        if "pages" in item:
            write_pdf(path, item["title"], item["pages"])
        else:
            path.write_text(item["text"], encoding="utf-8")
        print(f"wrote {path.relative_to(OUT_DIR.parents[1])}")


if __name__ == "__main__":
    main()
