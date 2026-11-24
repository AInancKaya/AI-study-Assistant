"""Thin HTTP client the Streamlit UI uses to talk to the FastAPI backend.

Every method returns plain parsed JSON (dicts and lists). Any failure is raised as
``ApiError`` so the UI only has to catch one exception type.
"""

from typing import Any
from urllib.parse import quote

import httpx

UNREACHABLE_HINT = "Run: make api"


class ApiError(Exception):
    """A failed API call. ``status`` is 0 when the API could not be reached at all."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


class ApiClient:
    """Calls the study assistant API. Keeps one ``httpx.Client`` for its whole life."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 300,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout, transport=transport
        )

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health").json()

    def upload(self, filename: str, data: bytes) -> dict[str, Any]:
        files = {"file": (filename, data, "application/pdf")}
        return self._request("POST", "/documents", files=files).json()

    def documents(self) -> list[dict[str, Any]]:
        return self._request("GET", "/documents").json()

    def delete(self, doc_id: str) -> None:
        self._request("DELETE", f"/documents/{quote(doc_id, safe='')}")

    def ask(self, question: str, doc_id: str | None = None) -> dict[str, Any]:
        body = {"question": question, "doc_id": doc_id}
        return self._request("POST", "/ask", json=body).json()

    def quiz(self, doc_id: str, n_mc: int, n_open: int) -> dict[str, Any]:
        body = {"doc_id": doc_id, "n_multiple_choice": n_mc, "n_open_ended": n_open}
        return self._request("POST", "/quiz", json=body).json()

    def grade(
        self,
        quiz_id: str,
        question_id: str,
        answer: str | None = None,
        selected_option: int | None = None,
    ) -> dict[str, Any]:
        body = {
            "quiz_id": quiz_id,
            "question_id": question_id,
            "answer": answer,
            "selected_option": selected_option,
        }
        return self._request("POST", "/grade", json=body).json()

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._http.request(method, path, **kwargs)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            detail = f"Cannot reach the API at {self._base_url}. {UNREACHABLE_HINT}"
            raise ApiError(0, detail) from exc
        if not response.is_success:
            raise ApiError(response.status_code, _error_detail(response))
        return response


def _error_detail(response: httpx.Response) -> str:
    """Return FastAPI's ``detail`` as text; fall back to the raw body."""
    try:
        payload = response.json()
    except ValueError:
        return response.text
    if not isinstance(payload, dict) or "detail" not in payload:
        return response.text
    detail = payload["detail"]
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        # FastAPI 422: a list of {"loc", "msg", "type"} dicts.
        return "; ".join(_validation_message(item) for item in detail)
    return str(detail)


def _validation_message(item: Any) -> str:
    if isinstance(item, dict) and "msg" in item:
        return str(item["msg"])
    return str(item)
