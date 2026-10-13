import pytest

from app.core.config import Settings


def test_cors_origins_accepts_comma_separated_env(monkeypatch):
    monkeypatch.setenv("DOCUMIND_CORS_ORIGINS", "http://a.test, http://b.test")
    assert Settings().cors_origins == ["http://a.test", "http://b.test"]


def test_cors_origins_accepts_json_list(monkeypatch):
    monkeypatch.setenv("DOCUMIND_CORS_ORIGINS", '["http://a.test"]')
    assert Settings().cors_origins == ["http://a.test"]


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        Settings(chunk_size=100, chunk_overlap=100)


def test_api_key_is_secret():
    settings = Settings(llm_api_key="sk-very-secret-value")
    assert "sk-very-secret-value" not in repr(settings)
