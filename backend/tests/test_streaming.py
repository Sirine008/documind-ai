"""Streaming answers: the preview filter, provider streams (mocked HTTP) and the SSE endpoint."""

import json
import uuid

import httpx
import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.core.errors import ProviderError
from app.llm.base import LLMRequest, LLMResponse, SourceRef
from app.llm.mock import MockProvider
from app.llm.prompts import AnswerStreamFilter
from app.llm.providers import AnthropicProvider, OpenAICompatibleProvider
from tests.conftest import make_pdf

API = "/api/v1"
SECRET = "sk-test-secret-key-123456"


def run_filter(deltas: list[str], secrets: list[str] | None = None) -> str:
    f = AnswerStreamFilter(secrets or [])
    return "".join(f.feed(d) for d in deltas) + f.finish()


def chars(text: str) -> list[str]:
    return list(text)


# --- preview filter --------------------------------------------------------------------------
def test_filter_strips_answer_prefix_and_stops_at_inference():
    raw = "ANSWER: The stipend is 600 euros [S1].\nINFERENCE: It seems generous."
    assert run_filter(chars(raw)) == "The stipend is 600 euros [S1]."


def test_filter_passes_unformatted_answers_through():
    assert run_filter(["Plain ", "answer ", "[S2]."]) == "Plain answer [S2]."


@pytest.mark.parametrize("raw", ["INSUFFICIENT_EVIDENCE", "ANSWER: INSUFFICIENT_EVIDENCE\nINFERENCE: None"])
def test_filter_never_shows_abstentions(raw):
    assert run_filter(chars(raw)) == ""


def test_filter_redacts_secret_split_across_deltas():
    raw = f"ANSWER: The key is {SECRET} and more text follows here."
    out = run_filter(chars(raw), [SECRET])
    assert SECRET not in out and "[redacted]" in out
    # No prefix of the secret leaks before the full secret has been seen.
    f, sent = AnswerStreamFilter([SECRET]), ""
    for c in raw[: raw.index(SECRET) + 10]:
        sent += f.feed(c)
    assert "sk-test" not in sent


def test_filter_output_is_append_only():
    f = AnswerStreamFilter([])
    pieces = [f.feed(c) for c in "ANSWER: abc def ghi jkl mno pqr stu vwx yz"] + [f.finish()]
    assert "".join(pieces) == "abc def ghi jkl mno pqr stu vwx yz"


# --- mock provider --------------------------------------------------------------------------
def test_mock_provider_streams_its_deterministic_answer():
    sources = [SourceRef("S1", "Policy", 2, "Each remote employee receives a home office stipend of 600 euros.")]
    req = LLMRequest(system="s", messages=[], sources=sources, question="How much is the home office stipend?")
    items = list(MockProvider().stream(req))
    *deltas, final = items
    assert isinstance(final, LLMResponse) and len(deltas) > 3
    assert "".join(deltas) == final.text == MockProvider().generate(req).text


# --- remote providers via mocked HTTP ----------------------------------------------------------
def _settings(**kw):
    return Settings(llm_api_key=SecretStr(SECRET), llm_model="test-model", **kw)


def _request():
    return LLMRequest(system="sys", messages=[{"role": "user", "content": "hello"}], max_tokens=50)


def _sse_body(events: list[str]) -> bytes:
    return "".join(f"{e}\n\n" for e in events).encode()


def _openai_chunks(texts: list[str], usage: dict | None = None) -> bytes:
    events = [
        "data: " + json.dumps({"model": "llama3.2:3b", "choices": [{"index": 0, "delta": {"content": t}}]})
        for t in texts
    ]
    if usage:
        events.append("data: " + json.dumps({"model": "llama3.2:3b", "choices": [], "usage": usage}))
    return _sse_body([*events, "data: [DONE]"])


def test_openai_provider_streams_deltas_and_usage():
    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = json.loads(request.content)
        body = _openai_chunks(["ANSWER: hi", " there [S1]"], {"prompt_tokens": 30, "completion_tokens": 6})
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    provider = OpenAICompatibleProvider(_settings(), client=httpx.Client(transport=httpx.MockTransport(handler)))
    *deltas, final = list(provider.stream(_request()))
    assert deltas == ["ANSWER: hi", " there [S1]"]
    assert final.text == "ANSWER: hi there [S1]" and final.model == "llama3.2:3b"
    assert final.usage.input_tokens == 30 and final.usage.output_tokens == 6 and not final.usage.estimated
    assert seen["body"]["stream"] is True and seen["body"]["messages"][0]["role"] == "system"


def test_openai_stream_without_usage_estimates_tokens():
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=_openai_chunks(["ANSWER: ok [S1]"])))
    provider = OpenAICompatibleProvider(_settings(), client=httpx.Client(transport=transport))
    final = list(provider.stream(_request()))[-1]
    assert final.usage.estimated and final.usage.output_tokens > 0


def test_openai_stream_http_error_does_not_leak_key():
    transport = httpx.MockTransport(lambda r: httpx.Response(401, json={"error": f"bad key {SECRET}"}))
    provider = OpenAICompatibleProvider(_settings(), client=httpx.Client(transport=transport))
    with pytest.raises(ProviderError) as exc:
        list(provider.stream(_request()))
    assert "sk-test" not in exc.value.message and "401" in exc.value.message


def test_openai_stream_malformed_chunk():
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=b"data: {not json}\n\n"))
    provider = OpenAICompatibleProvider(_settings(), client=httpx.Client(transport=transport))
    with pytest.raises(ProviderError):
        list(provider.stream(_request()))


def test_openai_stream_network_error():
    def handler(request):
        raise httpx.ReadTimeout("slow model")

    provider = OpenAICompatibleProvider(_settings(), client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(ProviderError):
        list(provider.stream(_request()))


def _anthropic_events(texts: list[str]) -> bytes:
    events = [
        {"type": "message_start", "message": {"model": "anthropic-test-model", "usage": {"input_tokens": 25}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        *({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": t}} for t in texts),
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 7}},
        {"type": "message_stop"},
    ]
    return _sse_body([f"event: {e['type']}\ndata: {json.dumps(e)}" for e in events])


def test_anthropic_provider_streams_deltas_and_usage():
    def handler(request: httpx.Request):
        assert request.headers["x-api-key"] == SECRET
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, content=_anthropic_events(["ANSWER: Yes", " [S2]."]))

    provider = AnthropicProvider(_settings(), client=httpx.Client(transport=httpx.MockTransport(handler)))
    *deltas, final = list(provider.stream(_request()))
    assert deltas == ["ANSWER: Yes", " [S2]."] and final.text == "ANSWER: Yes [S2]."
    assert final.model == "anthropic-test-model" and final.usage.input_tokens == 25 and final.usage.output_tokens == 7


def test_anthropic_stream_error_event():
    body = _sse_body(['event: error\ndata: {"type": "error", "error": {"type": "overloaded_error"}}'])
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=body))
    provider = AnthropicProvider(_settings(), client=httpx.Client(transport=transport))
    with pytest.raises(ProviderError):
        list(provider.stream(_request()))


def test_anthropic_stream_requires_key():
    with pytest.raises(ProviderError):
        list(AnthropicProvider(Settings(llm_api_key=None)).stream(_request()))


# --- SSE endpoint ----------------------------------------------------------------------------
POLICY = make_pdf(
    [
        "Eligibility. Employees become eligible for remote work after completing a 90-day onboarding period.",
        "Equipment. Each remote employee receives a one-time home office stipend of 600 euros.",
    ]
)


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


@pytest.fixture
def conversation(client):
    doc = client.post(f"{API}/documents", files={"file": ("policy.pdf", POLICY, "application/pdf")}).json()
    return client.post(f"{API}/conversations", json={"document_ids": [doc["document"]["id"]]}).json()


def ask_stream(client, conv_id, question):
    resp = client.post(f"{API}/conversations/{conv_id}/messages/stream", json={"question": question})
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/event-stream")
    return parse_sse(resp.text)


@pytest.mark.db
def test_stream_endpoint_streams_tokens_then_persisted_turn(client, conversation):
    events = ask_stream(client, conversation["id"], "How much is the home office stipend?")
    kinds = [k for k, _ in events]
    assert kinds[0] == "start" and kinds[-1] == "done" and kinds.count("token") > 3
    streamed = "".join(d["text"] for k, d in events if k == "token")
    turn = events[-1][1]
    answer = turn["assistant_message"]
    assert answer["evidence_status"] == "supported" and "600 euros" in answer["content"]
    assert streamed == answer["content"]  # the mock's preview equals its validated answer
    assert answer["citations"][0]["page_number"] == 2

    detail = client.get(f"{API}/conversations/{conversation['id']}").json()
    assert [m["id"] for m in detail["messages"]] == [turn["user_message"]["id"], answer["id"]]
    assert detail["title"] == "How much is the home office stipend?"


@pytest.mark.db
def test_stream_endpoint_matches_non_streaming_answer(client, conversation):
    question = "How much is the home office stipend?"
    streamed = ask_stream(client, conversation["id"], question)[-1][1]["assistant_message"]
    plain = client.post(f"{API}/conversations/{conversation['id']}/messages", json={"question": question}).json()
    plain = plain["assistant_message"]
    for key in ("content", "inference", "evidence_status", "citations"):
        assert streamed[key] == plain[key]


@pytest.mark.db
def test_stream_endpoint_respects_insufficient_evidence_gate(client, conversation):
    events = ask_stream(client, conversation["id"], "What is the parental leave policy?")
    assert [k for k, _ in events] == ["start", "done"]
    assert events[-1][1]["assistant_message"]["evidence_status"] == "insufficient_evidence"
    assert len(client.get(f"{API}/conversations/{conversation['id']}").json()["messages"]) == 2


@pytest.mark.db
def test_stream_endpoint_validation_and_missing_conversation(client, conversation):
    missing = client.post(f"{API}/conversations/{uuid.uuid4()}/messages/stream", json={"question": "x"})
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "not_found"
    empty = client.post(f"{API}/conversations/{conversation['id']}/messages/stream", json={"question": "  "})
    assert empty.status_code == 422


@pytest.mark.db
def test_stream_endpoint_with_openai_provider_over_mocked_http(client, conversation, monkeypatch):
    def handler(request: httpx.Request):
        body = _openai_chunks(["ANSWER: The stipend", " is 600 euros (S1).", "\nINFERENCE: None"])
        return httpx.Response(200, content=body)

    provider = OpenAICompatibleProvider(
        Settings(llm_provider="openai", llm_model="llama3.2:3b"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr("app.services.qa.get_llm_provider", lambda: provider)
    events = ask_stream(client, conversation["id"], "How much is the home office stipend?")
    streamed = "".join(d["text"] for k, d in events if k == "token")
    answer = events[-1][1]["assistant_message"]
    assert streamed == "The stipend is 600 euros (S1)."  # raw preview
    assert answer["content"] == "The stipend is 600 euros [S1]."  # validated final answer
    assert answer["evidence_status"] == "supported" and answer["usage"]["model"] == "llama3.2:3b"


@pytest.mark.db
def test_stream_endpoint_reports_provider_errors_and_stores_nothing(client, conversation, monkeypatch):
    transport = httpx.MockTransport(lambda r: httpx.Response(503, json={"error": "loading model"}))
    provider = OpenAICompatibleProvider(Settings(llm_provider="openai"), client=httpx.Client(transport=transport))
    monkeypatch.setattr("app.services.qa.get_llm_provider", lambda: provider)
    events = ask_stream(client, conversation["id"], "How much is the home office stipend?")
    assert [k for k, _ in events] == ["start", "error"]
    assert events[-1][1]["error"]["code"] == "llm_provider_error"
    assert client.get(f"{API}/conversations/{conversation['id']}").json()["messages"] == []
