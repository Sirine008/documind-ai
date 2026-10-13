import itertools

import pytest

from app.ingestion.chunking import chunk_pages, chunk_text, estimate_tokens
from app.ingestion.extraction import ExtractionError, PageText, extract, extract_pdf, normalize_text
from tests.conftest import make_pdf


def test_pdf_extraction_preserves_page_boundaries():
    pages = extract_pdf(make_pdf(["Alpha content on page one.", "", "Gamma content on page three."]))
    assert [p.page_number for p in pages] == [1, 2, 3]
    assert "Alpha" in pages[0].text and pages[1].text == "" and "Gamma" in pages[2].text


def test_text_file_has_no_page_numbers():
    pages = extract("text", b"Line one.\nLine two.")
    assert len(pages) == 1 and pages[0].page_number is None


def test_corrupt_pdf_raises_extraction_error():
    with pytest.raises(ExtractionError):
        extract_pdf(b"%PDF-1.4 nope")


def test_normalize_text_cleans_artifacts():
    raw = "The ﬁrst line con-\ntinues here.\r\nSame   para.\n\n\n\nNew para.­​"
    out = normalize_text(raw)
    assert out == "The first line continues here. Same para.\n\nNew para."


def test_chunk_text_covers_text_with_overlap():
    text = " ".join(f"word{i}" for i in range(400))
    spans = chunk_text(text, chunk_size=200, overlap=50)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    for (s1, e1), (s2, _e2) in itertools.pairwise(spans):
        assert s2 < e1, "consecutive chunks must overlap"
        assert s2 > s1, "chunking must make progress"
        assert text[s2 - 1] == " ", "overlap starts on a word boundary"
    assert all(e - s <= 200 for s, e in spans)


def test_chunk_text_prefers_sentence_boundaries():
    text = "First sentence is here. " * 30
    for start, end in chunk_text(text, chunk_size=120, overlap=20)[:-1]:
        assert text[start:end].rstrip().endswith(".")


def test_chunk_text_rejects_bad_parameters():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=100, overlap=100)


def test_chunk_pages_preserves_metadata_and_never_crosses_pages():
    pages = [
        PageText(1, "Page one text. " * 40),
        PageText(2, "Page two text. " * 40),
    ]
    chunks = chunk_pages(pages, chunk_size=200, overlap=40, min_chars=20)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert {c.page_number for c in chunks} == {1, 2}
    for c in chunks:
        assert ("one" in c.content) != ("two" in c.content)
        assert c.char_end > c.char_start
        assert len(c.content_hash) == 64
        assert c.token_estimate == estimate_tokens(c.content)


def test_chunk_pages_skips_exact_duplicates():
    pages = [
        PageText(1, "Repeated footer text that appears everywhere."),
        PageText(2, "Repeated footer text that appears everywhere."),
    ]
    assert len(chunk_pages(pages, chunk_size=500, overlap=50, min_chars=10)) == 1


def test_short_page_is_kept_when_it_is_all_there_is():
    chunks = chunk_pages([PageText(1, "Tiny note.")], chunk_size=500, overlap=50, min_chars=40)
    assert len(chunks) == 1
