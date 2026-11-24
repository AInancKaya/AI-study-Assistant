"""Tests for the Streamlit UI's HTTP client (``httpx.MockTransport``, no network)."""

import json

import httpx
import pytest

from ui.api_client import ApiClient, ApiError

BASE_URL = "http://api.test"


def _client(handler: httpx.MockTransport | None = None) -> ApiClient:
    return ApiClient(BASE_URL, transport=handler)


def _recorder(response: httpx.Response) -> tuple[list[httpx.Request], httpx.MockTransport]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return response

    return seen, httpx.MockTransport(handler)


def test_ask_posts_json() -> None:
    answer = {"answer": "Plato", "found": True, "citations": []}
    seen, transport = _recorder(httpx.Response(200, json=answer))

    result = _client(transport).ask("Who was Socrates' student?", doc_id="abc")

    assert result == answer
    request = seen[0]
    assert request.method == "POST"
    assert request.url.path == "/ask"
    assert json.loads(request.content) == {
        "question": "Who was Socrates' student?",
        "doc_id": "abc",
    }


def test_error_detail_raised_as_api_error() -> None:
    _, transport = _recorder(httpx.Response(404, json={"detail": "Document not found"}))

    with pytest.raises(ApiError) as excinfo:
        _client(transport).ask("anything", doc_id="missing")

    assert excinfo.value.status == 404
    assert excinfo.value.detail == "Document not found"


def test_validation_error_list_is_stringified() -> None:
    body = {"detail": [{"loc": ["body", "question"], "msg": "Field required"}]}
    _, transport = _recorder(httpx.Response(422, json=body))

    with pytest.raises(ApiError) as excinfo:
        _client(transport).ask("")

    assert excinfo.value.status == 422
    assert excinfo.value.detail == "Field required"


def test_non_json_error_uses_response_text() -> None:
    _, transport = _recorder(httpx.Response(500, text="Internal Server Error"))

    with pytest.raises(ApiError) as excinfo:
        _client(transport).health()

    assert excinfo.value.detail == "Internal Server Error"


def test_upload_sends_multipart() -> None:
    seen, transport = _recorder(httpx.Response(201, json={"doc_id": "x", "pages": 2}))

    result = _client(transport).upload("notes.pdf", b"%PDF-1.4 fake")

    assert result == {"doc_id": "x", "pages": 2}
    request = seen[0]
    assert request.method == "POST"
    assert request.url.path == "/documents"
    assert request.headers["content-type"].startswith("multipart/form-data")
    body = request.content
    assert b'name="file"; filename="notes.pdf"' in body
    assert b"Content-Type: application/pdf" in body
    assert b"%PDF-1.4 fake" in body


def test_delete_returns_none_on_204() -> None:
    seen, transport = _recorder(httpx.Response(204))

    assert _client(transport).delete("abc") is None
    assert seen[0].method == "DELETE"
    assert seen[0].url.path == "/documents/abc"


def test_unreachable_api_raises_status_zero() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(ApiError) as excinfo:
        _client(httpx.MockTransport(handler)).documents()

    assert excinfo.value.status == 0
    assert excinfo.value.detail == f"Cannot reach the API at {BASE_URL}. Run: make api"
