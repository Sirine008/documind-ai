"""LLM provider factory."""

from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from app.llm.base import LLMProvider
from app.llm.mock import MockProvider
from app.llm.providers import AnthropicProvider, OpenAICompatibleProvider


@lru_cache
def get_llm_provider() -> LLMProvider:
    settings = get_settings()
    if settings.llm_provider == "openai":
        return OpenAICompatibleProvider(settings)
    if settings.llm_provider == "anthropic":
        return AnthropicProvider(settings)
    return MockProvider()
