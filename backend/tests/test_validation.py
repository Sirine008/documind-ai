import pytest

from app.core.errors import PayloadTooLarge, UnsupportedMediaType, ValidationFailed
from app.ingestion.validation import sanitize_filename, title_from_filename, validate_upload
from tests.conftest import make_encrypted_pdf, make_pdf

MB = 1024 * 1024


def validate(data, name, max_bytes=MB, max_pages=50):
    return validate_upload(data, name, max_bytes=max_bytes, max_pages=max_pages)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd.txt", "passwd.txt"),
        ("C:\\Users\\me\\report.pdf", "report.pdf"),
        ("my report (final).pdf", "my report (final).pdf"),
        ("evil<script>.txt", "evil_script_.txt"),
        ("", "document"),
        ("....", "document"),
        ("na\x00me\x07.txt", "name.txt"),
    ],
)
def test_sanitize_filename(raw, expected):
    assert sanitize_filename(raw) == expected


def test_sanitize_filename_truncates_but_keeps_extension():
    name = sanitize_filename("a" * 400 + ".pdf")
    assert len(name) <= 200 and name.endswith(".pdf")


def test_title_from_filename():
    assert title_from_filename("quarterly_report-2024.pdf") == "quarterly report 2024"


def test_valid_pdf_and_text():
    pdf = validate(make_pdf(["Hello world page one", "Page two"]), "doc.pdf")
    assert pdf.kind == "pdf" and pdf.page_count == 2 and len(pdf.sha256) == 64
    txt = validate(b"Plain text content.\n", "notes.md")
    assert txt.kind == "text" and txt.page_count is None


def test_empty_file_rejected():
    with pytest.raises(ValidationFailed) as exc:
        validate(b"", "a.txt")
    assert exc.value.code == "empty_file"


def test_whitespace_only_text_rejected():
    with pytest.raises(ValidationFailed):
        validate(b"   \n\t  ", "a.txt")


def test_oversized_file_rejected():
    with pytest.raises(PayloadTooLarge):
        validate(b"x" * (MB + 1), "big.txt")


def test_unsupported_extension_rejected():
    with pytest.raises(UnsupportedMediaType) as exc:
        validate(b"data", "image.png")
    assert exc.value.code == "unsupported_type"


def test_extension_is_not_trusted_for_pdf():
    with pytest.raises(UnsupportedMediaType) as exc:
        validate(b"just some text pretending", "fake.pdf")
    assert exc.value.code == "content_mismatch"


def test_pdf_renamed_to_txt_rejected():
    with pytest.raises(UnsupportedMediaType):
        validate(make_pdf(["hello"]), "renamed.txt")


def test_binary_data_with_txt_extension_rejected():
    with pytest.raises(UnsupportedMediaType) as exc:
        validate(bytes(range(256)) * 10, "binary.txt")
    assert exc.value.code == "invalid_text_encoding"


def test_corrupt_pdf_rejected():
    with pytest.raises(ValidationFailed) as exc:
        validate(b"%PDF-1.7\n garbage garbage", "broken.pdf")
    assert exc.value.code == "corrupt_pdf"


def test_encrypted_pdf_rejected():
    with pytest.raises(ValidationFailed) as exc:
        validate(make_encrypted_pdf(), "locked.pdf")
    assert exc.value.code == "encrypted_pdf"


def test_page_limit_enforced():
    with pytest.raises(PayloadTooLarge) as exc:
        validate(make_pdf(["p"] * 3), "long.pdf", max_pages=2)
    assert exc.value.code == "too_many_pages"


def test_utf8_text_with_multibyte_characters_accepted():
    data = ("Café résumé naïve — 東京 " * 3000).encode()
    assert validate(data, "unicode.txt").kind == "text"
