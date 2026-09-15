"""OpenAI-shaped HTTP errors for the `/v1` endpoints.

Unlike the Ollama endpoints (which translate runtime failures into assistant text),
`/v1` answers with real HTTP status codes and the OpenAI error object
``{"error": {"message", "type", "param", "code"}}`` so SDK retries and agents behave.
"""

from typing import Any

from flask import Response, jsonify, request
from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError
from werkzeug.exceptions import HTTPException

from ollama_adapter import state
from ollama_adapter.prompt_renderer import PromptRenderError

_HTTP_NOT_FOUND = 404
_HTTP_METHOD_NOT_ALLOWED = 405
_HTTP_INTERNAL_ERROR = 500
_HTTP_BAD_GATEWAY = 502
_HTTP_GATEWAY_TIMEOUT = 504
_FORWARDED_UPSTREAM_HEADERS = ("retry-after", "retry-after-ms")
_ERROR_FIELDS = ("message", "type", "param", "code")


class OpenAIHTTPError(Exception):
    """An error ready to be sent as an OpenAI-format HTTP response."""

    def __init__(  # noqa: PLR0913 — mirrors the OpenAI error object fields
        self,
        status: int,
        message: str,
        *,
        type_: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
        headers: dict[str, str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Store the HTTP status, OpenAI error fields, extra response headers and extra body keys."""
        super().__init__(message)
        self.status = status
        self.message = message
        self.type = type_
        self.param = param
        self.code = code
        self.headers = headers or {}
        self.extra = extra or {}

    def body(self) -> dict[str, Any]:
        """Return the JSON body ``{"error": {...}}``."""
        error = {**self.extra, "message": self.message, "type": self.type, "param": self.param, "code": self.code}
        return {"error": error}


def _default_type(status: int) -> str:
    return "server_error" if status >= _HTTP_INTERNAL_ERROR else "invalid_request_error"


def _from_status_error(exc: APIStatusError) -> OpenAIHTTPError:
    """Pass an upstream HTTP error through with its status, body and retry headers."""
    status = exc.status_code
    body: Any = exc.body
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        body = body["error"]

    headers = {h: v for h in _FORWARDED_UPSTREAM_HEADERS if (v := exc.response.headers.get(h)) is not None}

    if not isinstance(body, dict):
        return OpenAIHTTPError(status, exc.message, type_=_default_type(status), headers=headers)

    message = body.get("message")
    return OpenAIHTTPError(
        status,
        message if isinstance(message, str) and message else exc.message,
        type_=body.get("type") or _default_type(status),
        param=body.get("param"),
        code=body.get("code"),
        headers=headers,
        extra={k: v for k, v in body.items() if k not in _ERROR_FIELDS},
    )


def _from_http_exception(exc: HTTPException) -> OpenAIHTTPError:
    """Map Flask/Werkzeug routing errors (unknown URL, wrong method, ...) onto OpenAI errors."""
    status = exc.code or _HTTP_INTERNAL_ERROR
    if status == _HTTP_NOT_FOUND:
        return OpenAIHTTPError(status, f"Unknown request URL: {request.method} {request.path}", code="unknown_url")
    if status == _HTTP_METHOD_NOT_ALLOWED:
        valid_methods = getattr(exc, "valid_methods", None)
        return OpenAIHTTPError(
            status,
            f"Method {request.method} is not allowed for {request.path}",
            code="method_not_allowed",
            headers={"Allow": ", ".join(valid_methods)} if valid_methods else None,
        )
    return OpenAIHTTPError(status, exc.description or exc.name, type_=_default_type(status))


def to_openai_error(exc: BaseException) -> OpenAIHTTPError:  # noqa: PLR0911
    """Map any exception raised while serving `/v1` onto an OpenAI-format HTTP error."""
    if isinstance(exc, OpenAIHTTPError):
        return exc
    if isinstance(exc, HTTPException):
        return _from_http_exception(exc)
    if isinstance(exc, PromptRenderError):
        return OpenAIHTTPError(
            _HTTP_INTERNAL_ERROR,
            f"[PROMPT ERROR] {exc}",
            type_="prompt_render_error",
            code="prompt_render_error",
            headers={"x-should-retry": "false"},
        )
    # APITimeoutError subclasses APIConnectionError, so it must be checked first.
    if isinstance(exc, APITimeoutError):
        return OpenAIHTTPError(_HTTP_GATEWAY_TIMEOUT, str(exc), type_="timeout_error", code="upstream_timeout")
    if isinstance(exc, APIConnectionError):
        return OpenAIHTTPError(
            _HTTP_BAD_GATEWAY, str(exc), type_="api_connection_error", code="upstream_connection_error"
        )
    if isinstance(exc, APIStatusError):
        return _from_status_error(exc)
    if isinstance(exc, APIError):
        body = exc.body if isinstance(exc.body, dict) else {}
        message = body.get("message")
        return OpenAIHTTPError(
            _HTTP_BAD_GATEWAY,
            message if isinstance(message, str) and message else exc.message,
            type_=body.get("type") or "api_error",
            param=body.get("param"),
            code=body.get("code"),
            extra={k: v for k, v in body.items() if k not in _ERROR_FIELDS},
        )
    return OpenAIHTTPError(_HTTP_INTERNAL_ERROR, "Internal server error", type_="server_error")


def render_error(err: OpenAIHTTPError) -> tuple[Response, int]:
    """Build the Flask response for an already-mapped error (no logging)."""
    response = jsonify(err.body())
    for name, value in err.headers.items():
        response.headers[name] = value
    return response, err.status


def error_response(exc: BaseException) -> tuple[Response, int]:
    """Build the Flask response for `exc`, logging server-side failures with a stack trace."""
    err = to_openai_error(exc)
    expected = isinstance(exc, OpenAIHTTPError | APIStatusError | HTTPException)
    if err.status >= _HTTP_INTERNAL_ERROR or not expected:
        state.logger.error("OpenAI endpoint error: %s", err.message, exc_info=exc)
    else:
        state.logger.warning("OpenAI endpoint error (HTTP %d): %s", err.status, err.message)
    return render_error(err)
