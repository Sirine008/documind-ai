"""Remote LLM providers (OpenAI-compatible and Anthropic) implemented with httpx.

The OpenAI-compatible provider also works with local servers that expose the
``/v1/chat/completions`` API (for example Ollama or llama.cpp's server) by setting
``DOCUMIND_LLM_BASE_URL``.

Both providers can stream: they request server-sent events and yield text deltas,
followed by one ``LLMResponse`` carrying the full text and token usage.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

import httpx

from app.core.config import Settings
from app.core.errors import ProviderError
from app.llm.base import LLMRequest, LLMResponse, LLMUsage

logger = logging.getLogger(__name__)

UNREACHABLE = "The language model provider could not be reached."
UNEXPECTED = "The language model provider returned an unexpected response."


def _estimate(text: str) -> int:
    return max(1, len(text) // 4)


def _raise_for_status(resp: httpx.Response, provider: str) -> None:
    if resp.status_code >= 400:
        logger.warning("llm_http_error", extra={"status": resp.status_code, "provider": provider})
        raise ProviderError(f"The language model provider returned HTTP {resp.status_code}.")


def _sse_data(resp: httpx.Response) -> Iterator[str]:
    """The ``data:`` payloads of a server-sent event stream."""
    for line in resp.iter_lines():
        if line.startswith("data:"):
            yield line[5:].strip()


class OpenAICompatibleProvider:
    name = "openai"

    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.model = settings.llm_model
        self.base_url = settings.llm_base_url.rstrip("/")
        self.api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
        self.client = client or httpx.Client(timeout=settings.llm_timeout_s)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, request: LLMRequest) -> dict:
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": request.system}, *request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if request.json_output:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _usage(self, usage: dict, payload: dict, text: str) -> LLMUsage:
        return LLMUsage(
            input_tokens=int(usage.get("prompt_tokens") or _estimate(str(payload["messages"]))),
            output_tokens=int(usage.get("completion_tokens") or _estimate(text)),
            estimated=not bool(usage),
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        payload = self._payload(request)
        try:
            resp = self.client.post(f"{self.base_url}/chat/completions", json=payload, headers=self._headers())
        except httpx.HTTPError as exc:
            raise ProviderError(UNREACHABLE) from exc
        _raise_for_status(resp, self.name)
        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(UNEXPECTED) from exc
        return LLMResponse(
            text=text,
            provider=self.name,
            model=data.get("model", self.model),
            usage=self._usage(data.get("usage") or {}, payload, text),
        )

    def stream(self, request: LLMRequest) -> Iterator[str | LLMResponse]:
        payload = {**self._payload(request), "stream": True, "stream_options": {"include_usage": True}}
        parts: list[str] = []
        usage: dict = {}
        model = self.model
        try:
            with self.client.stream(
                "POST", f"{self.base_url}/chat/completions", json=payload, headers=self._headers()
            ) as resp:
                _raise_for_status(resp, self.name)
                for data in _sse_data(resp):
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        model = chunk.get("model") or model
                        usage = chunk.get("usage") or usage
                        delta = chunk["choices"][0]["delta"].get("content") if chunk.get("choices") else None
                    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
                        raise ProviderError(UNEXPECTED) from exc
                    if delta:
                        parts.append(delta)
                        yield delta
        except httpx.HTTPError as exc:
            raise ProviderError(UNREACHABLE) from exc
        text = "".join(parts)
        yield LLMResponse(text=text, provider=self.name, model=model, usage=self._usage(usage, payload, text))


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.model = settings.llm_model
        base = settings.llm_base_url.rstrip("/")
        self.base_url = base if "anthropic" in base else "https://api.anthropic.com/v1"
        self.api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
        self.client = client or httpx.Client(timeout=settings.llm_timeout_s)

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ProviderError("DOCUMIND_LLM_API_KEY is required for the anthropic provider.")
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _payload(self, request: LLMRequest) -> dict:
        return {
            "model": self.model,
            "system": request.system,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

    def generate(self, request: LLMRequest) -> LLMResponse:
        headers = self._headers()
        try:
            resp = self.client.post(f"{self.base_url}/messages", json=self._payload(request), headers=headers)
        except httpx.HTTPError as exc:
            raise ProviderError(UNREACHABLE) from exc
        _raise_for_status(resp, self.name)
        try:
            data = resp.json()
            text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderError(UNEXPECTED) from exc
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

    def stream(self, request: LLMRequest) -> Iterator[str | LLMResponse]:
        headers = self._headers()
        payload = {**self._payload(request), "stream": True}
        parts: list[str] = []
        usage: dict = {}
        model = self.model
        try:
            with self.client.stream("POST", f"{self.base_url}/messages", json=payload, headers=headers) as resp:
                _raise_for_status(resp, self.name)
                for data in _sse_data(resp):
                    try:
                        event = json.loads(data)
                        kind = event["type"]
                        if kind == "message_start":
                            model = event["message"].get("model") or model
                            usage.update(event["message"].get("usage") or {})
                        elif kind == "message_delta":
                            usage.update(event.get("usage") or {})
                        delta = event["delta"].get("text") if kind == "content_block_delta" else None
                    except (ValueError, KeyError, TypeError, AttributeError) as exc:
                        raise ProviderError(UNEXPECTED) from exc
                    if kind == "error":
                        logger.warning("llm_stream_error", extra={"provider": self.name})
                        raise ProviderError("The language model provider reported an error while streaming.")
                    if kind == "message_stop":
                        break
                    if delta:
                        parts.append(delta)
                        yield delta
        except httpx.HTTPError as exc:
            raise ProviderError(UNREACHABLE) from exc
        text = "".join(parts)
        yield LLMResponse(
            text=text,
            provider=self.name,
            model=model,
            usage=LLMUsage(
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or _estimate(text)),
                estimated=not bool(usage),
            ),
        )
