"""LLM provider interface shared by all providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol


@dataclass(frozen=True)
class SourceRef:
    """A retrieved passage as presented to the model (label like ``S1``)."""

    label: str
    title: str
    page_number: int | None
    content: str


@dataclass
class LLMRequest:
    system: str
    messages: list[dict[str, str]]
    task: Literal["answer", "summary", "key_points", "revision_notes", "quiz"] = "answer"
    # Structured copies of the sources and question. Remote providers only use the
    # rendered prompt text; the deterministic mock provider uses these directly.
    sources: list[SourceRef] = field(default_factory=list)
    question: str = ""
    max_tokens: int = 800
    temperature: float = 0.1
    json_output: bool = False


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    estimated: bool = True

    def as_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated": self.estimated,
        }


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    usage: LLMUsage = field(default_factory=LLMUsage)


class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, request: LLMRequest) -> LLMResponse: ...
