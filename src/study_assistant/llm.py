"""Local LLM access through Ollama, with structured (JSON-schema) output and clear errors.

``LLMClient`` is the small interface the rest of the app depends on; ``OllamaClient`` is the
real implementation and ``tests/fakes.py::FakeLLM`` the test double.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Protocol, TypeVar

import httpx
import ollama
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")

RETRY_MESSAGE = "Your previous reply was not valid JSON for the schema. Reply again with JSON only."
CHAT_OPTIONS: dict[str, Any] = {"temperature": 0, "num_ctx": 8192}


class LLMError(RuntimeError):
    """The local model could not produce a usable answer."""


class LLMUnavailableError(LLMError):
    """The Ollama server cannot be reached."""


class ModelNotFoundError(LLMError):
    """The configured model is not installed in Ollama."""


class LLMClient(Protocol):
    """What the app needs from an LLM: schema-validated JSON replies and a health check."""

    def complete_json(self, system: str, user: str, schema: type[T]) -> T: ...

    def check(self) -> None: ...


class OllamaClient:
    """``LLMClient`` backed by a local Ollama server."""

    def __init__(self, host: str, model: str, timeout: float) -> None:
        self.host = host
        self.model = model
        self.timeout = timeout
        self._client = ollama.Client(host=host, timeout=timeout)

    def complete_json(self, system: str, user: str, schema: type[T]) -> T:
        """Ask the model for JSON matching ``schema``; retry once if the reply is invalid."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        reply = self._chat(messages, schema)
        try:
            return schema.model_validate_json(reply)
        except (ValidationError, json.JSONDecodeError):
            pass

        retry_messages = [
            *messages,
            {"role": "assistant", "content": reply},
            {"role": "user", "content": RETRY_MESSAGE},
        ]
        reply = self._chat(retry_messages, schema)
        try:
            return schema.model_validate_json(reply)
        except (ValidationError, json.JSONDecodeError) as exc:
            raise LLMError(
                f"The local model did not return valid JSON for {schema.__name__} after a retry."
            ) from exc

    def check(self) -> None:
        """Raise ``LLMUnavailableError``/``ModelNotFoundError`` unless the model is ready."""
        self._call(lambda: self._client.show(self.model))

    def _chat(self, messages: list[dict[str, str]], schema: type[BaseModel]) -> str:
        response = self._call(
            lambda: self._client.chat(
                model=self.model,
                messages=messages,
                format=schema.model_json_schema(),
                options=CHAT_OPTIONS,
            )
        )
        return response.message.content or ""

    def _call(self, fn: Callable[[], R]) -> R:
        """Run one Ollama request, translating library errors into ``LLMError`` subclasses."""
        try:
            return fn()
        except (httpx.ConnectError, ConnectionError) as exc:
            raise LLMUnavailableError(
                f"Ollama is not running at {self.host}. Start it with: ollama serve"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMError(
                f"The local model took too long to answer (timeout {self.timeout:g}s)."
            ) from exc
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                raise ModelNotFoundError(
                    f"Model '{self.model}' is not installed. Run: ollama pull {self.model}"
                ) from exc
            raise LLMError(f"Ollama returned an error: {exc.error}") from exc
