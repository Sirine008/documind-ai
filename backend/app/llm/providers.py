"""Remote LLM providers (OpenAI-compatible and Anthropic) implemented with httpx.

The OpenAI-compatible provider also works with local servers that expose the
``/v1/chat/completions`` API (for example Ollama or llama.cpp's server) by setting
``DOCUMIND_LLM_BASE_URL``.
"""

from __future__ import annotations

import logging

import httpx

from app.core.config import Settings
from app.core.errors import ProviderError
from app.llm.base import LLMRequest, LLMResponse, LLMUsage

logger = logging.getLogger(__name__)


def _estimate(text: str) -> int:
    return max(1, len(text) // 4)


class OpenAICompatibleProvider:
    name = "openai"

    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.model = settings.llm_model
        self.base_url = settings.llm_base_url.rstrip("/")
        self.api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
        self.client = client or httpx.Client(timeout=settings.llm_timeout_s)

    def generate(self, request: LLMRequest) -> LLMResponse:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": request.system}, *request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if request.json_output:
            payload["response_format"] = {"type": "json_object"}
        try:
            resp = self.client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise ProviderError("The language model provider could not be reached.") from exc
        if resp.status_code >= 400:
            logger.warning("llm_http_error", extra={"status": resp.status_code, "provider": self.name})
            raise ProviderError(f"The language model provider returned HTTP {resp.status_code}.")
        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError("The language model provider returned an unexpected response.") from exc
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            provider=self.name,
            model=data.get("model", self.model),
            usage=LLMUsage(
                input_tokens=int(usage.get("prompt_tokens") or _estimate(str(payload["messages"]))),
                output_tokens=int(usage.get("completion_tokens") or _estimate(text)),
                estimated=not bool(usage),
            ),
        )


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.model = settings.llm_model
        base = settings.llm_base_url.rstrip("/")
        self.base_url = base if "anthropic" in base else "https://api.anthropic.com/v1"
        self.api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
        self.client = client or httpx.Client(timeout=settings.llm_timeout_s)

    def generate(self, request: LLMRequest) -> LLMResponse:
        if not self.api_key:
            raise ProviderError("DOCUMIND_LLM_API_KEY is required for the anthropic provider.")
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "system": request.system,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        try:
            resp = self.client.post(f"{self.base_url}/messages", json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise ProviderError("The language model provider could not be reached.") from exc
        if resp.status_code >= 400:
            logger.warning("llm_http_error", extra={"status": resp.status_code, "provider": self.name})
            raise ProviderError(f"The language model provider returned HTTP {resp.status_code}.")
        try:
            data = resp.json()
            text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderError("The language model provider returned an unexpected response.") from exc
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            provider=self.name,
            model=data.get("model", self.model),
            usage=LLMUsage(
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or _estimate(text)),
                estimated=not bool(usage),
            ),
        )
