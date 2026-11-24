"""Tests for the Ollama LLM client (``study_assistant.llm``).

Unit tests replace ``ollama.Client`` with ``StubClient``, so they never touch the network.
The ``live`` test talks to a real Ollama server and is deselected by default.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import ollama
import pytest

from study_assistant.config import get_settings
from study_assistant.llm import (
    LLMClient,
    LLMError,
    LLMUnavailableError,
    ModelNotFoundError,
    OllamaClient,
)
from study_assistant.models import LLMGrade

HOST = "http://localhost:11434"
MODEL = "gemma2:9b"
RETRY_MESSAGE = "Your previous reply was not valid JSON for the schema. Reply again with JSON only."
VALID_GRADE = '{"score": 80, "feedback": "iyi"}'


class StubClient:
    """Stands in for ``ollama.Client``: replays queued replies, records every call."""

    instances: list[StubClient] = []
    replies: list[str | Exception] = []
    show_error: Exception | None = None

    def __init__(self, host: str | None = None, **kwargs: Any) -> None:
        self.host = host
        self.kwargs = kwargs
        self.chat_calls: list[dict[str, Any]] = []
        self.show_calls: list[str] = []
        StubClient.instances.append(self)

    def chat(self, **kwargs: Any) -> ollama.ChatResponse:
        self.chat_calls.append(kwargs)
        reply = StubClient.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ollama.ChatResponse(message=ollama.Message(role="assistant", content=reply))

    def show(self, model: str) -> None:
        self.show_calls.append(model)
        if StubClient.show_error is not None:
            raise StubClient.show_error


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> type[StubClient]:
    StubClient.instances = []
    StubClient.replies = []
    StubClient.show_error = None
    monkeypatch.setattr(ollama, "Client", StubClient)
    return StubClient


def make_client() -> OllamaClient:
    return OllamaClient(host=HOST, model=MODEL, timeout=30.0)


def test_returns_validated_model(stub: type[StubClient]) -> None:
    stub.replies = [VALID_GRADE]

    grade = make_client().complete_json("sys", "user", LLMGrade)

    assert grade == LLMGrade(score=80, feedback="iyi")


def test_client_created_once_with_host_and_timeout(stub: type[StubClient]) -> None:
    stub.replies = [VALID_GRADE, VALID_GRADE]
    client = make_client()

    client.complete_json("sys", "user", LLMGrade)
    client.complete_json("sys", "user", LLMGrade)

    assert len(stub.instances) == 1
    assert stub.instances[0].host == HOST
    assert stub.instances[0].kwargs == {"timeout": 30.0}


def test_retries_once_on_bad_json(stub: type[StubClient]) -> None:
    stub.replies = ["nope", VALID_GRADE]

    grade = make_client().complete_json("sys", "user", LLMGrade)

    assert grade.score == 80
    calls = stub.instances[0].chat_calls
    assert len(calls) == 2
    retry_messages = calls[1]["messages"]
    assert retry_messages[:2] == calls[0]["messages"]
    assert retry_messages[2] == {"role": "assistant", "content": "nope"}
    assert retry_messages[-1] == {"role": "user", "content": RETRY_MESSAGE}


def test_retries_on_json_that_does_not_match_schema(stub: type[StubClient]) -> None:
    stub.replies = ['{"score": "high"}', VALID_GRADE]

    assert make_client().complete_json("sys", "user", LLMGrade).feedback == "iyi"
    assert len(stub.instances[0].chat_calls) == 2


def test_raises_after_two_bad_replies(stub: type[StubClient]) -> None:
    stub.replies = ["nope", "still nope"]

    with pytest.raises(LLMError, match="valid JSON"):
        make_client().complete_json("sys", "user", LLMGrade)
    assert len(stub.instances[0].chat_calls) == 2


def test_sends_schema_and_temperature_zero(stub: type[StubClient]) -> None:
    stub.replies = [VALID_GRADE]

    make_client().complete_json("You grade answers.", "Grade this.", LLMGrade)

    call = stub.instances[0].chat_calls[0]
    assert call["model"] == MODEL
    assert call["format"] == LLMGrade.model_json_schema()
    assert call["options"] == {"temperature": 0, "num_ctx": 8192}
    assert call["messages"] == [
        {"role": "system", "content": "You grade answers."},
        {"role": "user", "content": "Grade this."},
    ]


@pytest.mark.parametrize(
    "error",
    [ConnectionError("Failed to connect to Ollama."), httpx.ConnectError("refused")],
)
def test_connection_error_maps_to_unavailable(stub: type[StubClient], error: Exception) -> None:
    stub.replies = [error]

    with pytest.raises(LLMUnavailableError) as exc_info:
        make_client().complete_json("sys", "user", LLMGrade)

    message = str(exc_info.value)
    assert "ollama serve" in message
    assert HOST in message


def test_404_maps_to_model_not_found(stub: type[StubClient]) -> None:
    stub.replies = [ollama.ResponseError("model 'gemma2:9b' not found", 404)]

    with pytest.raises(ModelNotFoundError) as exc_info:
        make_client().complete_json("sys", "user", LLMGrade)

    assert str(exc_info.value) == "Model 'gemma2:9b' is not installed. Run: ollama pull gemma2:9b"


def test_other_response_error_maps_to_llm_error(stub: type[StubClient]) -> None:
    stub.replies = [ollama.ResponseError("out of memory", 500)]

    with pytest.raises(LLMError, match="out of memory") as exc_info:
        make_client().complete_json("sys", "user", LLMGrade)

    assert type(exc_info.value) is LLMError


def test_timeout_maps_to_llm_error(stub: type[StubClient]) -> None:
    stub.replies = [httpx.ReadTimeout("timed out")]

    with pytest.raises(LLMError, match=r"took too long to answer \(timeout 30s\)"):
        make_client().complete_json("sys", "user", LLMGrade)


def test_check_passes_when_model_is_installed(stub: type[StubClient]) -> None:
    client = make_client()

    client.check()

    assert stub.instances[0].show_calls == [MODEL]


def test_check_maps_connection_error(stub: type[StubClient]) -> None:
    stub.show_error = ConnectionError("Failed to connect to Ollama.")

    with pytest.raises(LLMUnavailableError, match="ollama serve"):
        make_client().check()


def test_check_maps_404(stub: type[StubClient]) -> None:
    stub.show_error = ollama.ResponseError("model 'gemma2:9b' not found", 404)

    with pytest.raises(ModelNotFoundError, match="ollama pull gemma2:9b"):
        make_client().check()


def test_ollama_client_satisfies_protocol(stub: type[StubClient]) -> None:
    client: LLMClient = make_client()

    assert callable(client.complete_json)
    assert callable(client.check)


@pytest.mark.live
def test_live_ollama_roundtrip() -> None:
    settings = get_settings()
    client = OllamaClient(settings.ollama_host, settings.llm_model, settings.llm_timeout_s)
    client.check()

    started = time.perf_counter()
    grade = client.complete_json(
        system="You grade a student's answer. Reply with JSON: an integer score 0-100 and "
        "one short sentence of feedback.",
        user="Question: What is 2 + 2?\nStudent answer: 4",
        schema=LLMGrade,
    )
    elapsed = time.perf_counter() - started

    print(f"\nlive roundtrip: {elapsed:.1f}s -> {grade!r}")
    assert isinstance(grade, LLMGrade)
    assert 0 <= grade.score <= 100
    assert grade.feedback.strip()
