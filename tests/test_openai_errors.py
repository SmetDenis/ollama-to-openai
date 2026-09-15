"""Tests for ollama_adapter.openai_errors."""

import httpx
import openai
import pytest

from ollama_adapter.openai_errors import OpenAIHTTPError, error_response, to_openai_error
from ollama_adapter.prompt_renderer import PromptRenderError

from .conftest import make_status_error

_REQUEST = httpx.Request("POST", "http://upstream/v1/chat/completions")


def test_openai_http_error_body_shape():
    err = OpenAIHTTPError(400, "bad", param="messages")
    assert err.body() == {
        "error": {"message": "bad", "type": "invalid_request_error", "param": "messages", "code": None}
    }


def test_prompt_render_error():
    err = to_openai_error(PromptRenderError("missing var", source="inline"))
    assert err.status == 500
    assert err.type == "prompt_render_error"
    assert err.message == "[PROMPT ERROR] missing var"
    assert err.headers == {"x-should-retry": "false"}


def test_timeout_checked_before_connection():
    err = to_openai_error(openai.APITimeoutError(request=_REQUEST))
    assert (err.status, err.code) == (504, "upstream_timeout")


def test_connection_error():
    err = to_openai_error(openai.APIConnectionError(request=_REQUEST))
    assert (err.status, err.type) == (502, "api_connection_error")


@pytest.mark.parametrize(
    ("cls", "status"),
    [
        (openai.RateLimitError, 429),
        (openai.AuthenticationError, 401),
        (openai.PermissionDeniedError, 403),
        (openai.InternalServerError, 503),
    ],
)
def test_status_error_passthrough(cls, status):
    body = {"message": "upstream says no", "type": "some_type", "param": None, "code": "c1", "provider": "x"}
    err = to_openai_error(make_status_error(cls, status, body, headers={"retry-after": "7", "x-other": "1"}))
    assert err.status == status
    assert err.body() == {
        "error": {"provider": "x", "message": "upstream says no", "type": "some_type", "param": None, "code": "c1"}
    }
    assert err.headers == {"retry-after": "7"}


def test_status_error_fills_missing_type():
    err = to_openai_error(make_status_error(openai.InternalServerError, 500, {"message": "boom"}))
    assert err.type == "server_error"
    err = to_openai_error(make_status_error(openai.BadRequestError, 400, {"message": "bad"}))
    assert err.type == "invalid_request_error"


def test_status_error_wrapped_body_unwrapped():
    err = to_openai_error(make_status_error(openai.NotFoundError, 404, {"error": {"message": "nope"}}))
    assert err.message == "nope"


def test_status_error_non_dict_body():
    exc = make_status_error(openai.InternalServerError, 502, "gateway html")
    err = to_openai_error(exc)
    assert (err.status, err.message, err.type) == (502, exc.message, "server_error")


def test_generic_api_error_from_stream_event():
    exc = openai.APIError("stream failed", _REQUEST, body={"message": "overloaded", "type": "overloaded_error"})
    err = to_openai_error(exc)
    assert (err.status, err.message, err.type) == (502, "overloaded", "overloaded_error")


def test_unexpected_error_hides_details():
    err = to_openai_error(RuntimeError("secret internals"))
    assert (err.status, err.type, err.message) == (500, "server_error", "Internal server error")


@pytest.mark.usefixtures("app_context")
def test_error_response_sets_headers():
    response, status = error_response(PromptRenderError("x", source="inline"))
    assert status == 500
    assert response.headers["x-should-retry"] == "false"
    assert response.get_json()["error"]["type"] == "prompt_render_error"
