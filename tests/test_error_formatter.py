"""Tests for ollama_adapter.error_formatter."""

from unittest.mock import MagicMock

import httpx
import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

from ollama_adapter import state
from ollama_adapter.error_formatter import (
    DEFAULT_PREFIX,
    ErrorPresentation,
    categorize_error,
    format_error_text,
    is_enabled,
)


def _api_status_error(cls, status_code: int, message: str):
    """Build an openai HTTP-status exception with a synthetic response."""
    request = httpx.Request("POST", "http://test/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return cls(message=message, response=response, body={"message": message})


def _connection_error(message: str) -> APIConnectionError:
    request = httpx.Request("POST", "http://test/v1/chat/completions")
    return APIConnectionError(request=request, message=message)


def _timeout_error() -> APITimeoutError:
    request = httpx.Request("POST", "http://test/v1/chat/completions")
    return APITimeoutError(request=request)


# ---------------------------------------------------------------------------
# categorize_error
# ---------------------------------------------------------------------------


class TestCategorize:
    @pytest.mark.parametrize(
        ("exc_factory", "expected_category"),
        [
            (lambda: _api_status_error(RateLimitError, 429, "quota exceeded"), "Rate limit"),
            (lambda: _api_status_error(AuthenticationError, 401, "bad key"), "Auth"),
            (lambda: _api_status_error(PermissionDeniedError, 403, "forbidden"), "Permission denied"),
            (lambda: _api_status_error(NotFoundError, 404, "no such model"), "Not found"),
            (lambda: _api_status_error(UnprocessableEntityError, 422, "bad shape"), "Unprocessable"),
            (lambda: _api_status_error(BadRequestError, 400, "bad payload"), "Bad request"),
            (lambda: _api_status_error(ConflictError, 409, "conflict"), "Conflict"),
            (lambda: _api_status_error(InternalServerError, 500, "upstream boom"), "Upstream 5xx"),
            (lambda: _connection_error("dns failed"), "Connection"),
            (_timeout_error, "Timeout"),
            (lambda: RuntimeError("generic boom"), "Unexpected"),
            (lambda: ValueError("not a number"), "Unexpected"),
        ],
    )
    def test_known_categories(self, exc_factory, expected_category):
        presentation = categorize_error(exc_factory())
        assert presentation.category == expected_category

    def test_message_falls_back_to_class_name_when_empty(self):
        presentation = categorize_error(RuntimeError(""))
        assert presentation.message == "RuntimeError"


# ---------------------------------------------------------------------------
# ErrorPresentation.render
# ---------------------------------------------------------------------------


class TestRender:
    def test_full_render(self):
        out = ErrorPresentation("Rate limit", "quota exceeded").render(
            "[LLM ERROR]", include_type=True, show_details=True,
        )
        assert out == "[LLM ERROR] Rate limit: quota exceeded"

    def test_no_type(self):
        out = ErrorPresentation("Auth", "bad key").render(
            "[LLM ERROR]", include_type=False, show_details=True,
        )
        assert out == "[LLM ERROR] bad key"

    def test_no_details(self):
        out = ErrorPresentation("Auth", "bad key").render(
            "[LLM ERROR]", include_type=True, show_details=False,
        )
        assert out == "[LLM ERROR] Auth"

    def test_no_type_no_details_falls_back_to_category(self):
        out = ErrorPresentation("Auth", "bad key").render(
            "[LLM ERROR]", include_type=False, show_details=False,
        )
        assert out == "[LLM ERROR] Auth"

    def test_custom_prefix(self):
        out = ErrorPresentation("Timeout", "slow").render(
            "[oops]", include_type=True, show_details=True,
        )
        assert out == "[oops] Timeout: slow"


# ---------------------------------------------------------------------------
# is_enabled
# ---------------------------------------------------------------------------


class TestIsEnabled:
    def test_missing_section_defaults_true(self):
        state.CONFIG = {}
        assert is_enabled() is True

    def test_explicit_true(self):
        state.CONFIG = {"error_handling": {"enabled": True}}
        assert is_enabled() is True

    def test_explicit_false(self):
        state.CONFIG = {"error_handling": {"enabled": False}}
        assert is_enabled() is False

    def test_non_dict_section_treated_as_default(self):
        state.CONFIG = {"error_handling": "yes please"}
        assert is_enabled() is True


# ---------------------------------------------------------------------------
# format_error_text
# ---------------------------------------------------------------------------


class TestFormatErrorText:
    def test_default_prefix_used_when_no_section(self):
        state.CONFIG = {}
        text = format_error_text(RuntimeError("kaboom"))
        assert text.startswith(DEFAULT_PREFIX)
        assert "Unexpected" in text
        assert "kaboom" in text

    def test_respects_custom_prefix(self):
        state.CONFIG = {"error_handling": {"prefix": "<<oops>>"}}
        text = format_error_text(RuntimeError("kaboom"))
        assert text.startswith("<<oops>>")

    def test_strips_details_when_show_details_false(self):
        state.CONFIG = {"error_handling": {"show_details": False}}
        text = format_error_text(RuntimeError("secret detail"))
        assert "secret detail" not in text

    def test_omits_category_when_include_type_false(self):
        state.CONFIG = {"error_handling": {"include_type": False}}
        text = format_error_text(RuntimeError("hello"))
        assert "Unexpected" not in text
        assert "hello" in text

    def test_empty_string_prefix_falls_back_to_default(self):
        state.CONFIG = {"error_handling": {"prefix": "   "}}
        text = format_error_text(RuntimeError("boom"))
        assert text.startswith(DEFAULT_PREFIX)

    def test_format_for_openai_rate_limit(self):
        state.CONFIG = {}
        exc = _api_status_error(RateLimitError, 429, "quota exceeded")
        text = format_error_text(exc)
        assert text.startswith(DEFAULT_PREFIX)
        assert "Rate limit" in text
        assert "quota exceeded" in text


# ---------------------------------------------------------------------------
# Magic mock with no extra context to ensure we don't crash on weird exceptions
# ---------------------------------------------------------------------------


def test_categorize_handles_exception_without_args():
    class CustomError(Exception):
        pass

    presentation = categorize_error(CustomError())
    assert presentation.category == "Unexpected"
    assert presentation.message == "CustomError"


def test_categorize_handles_magicmock_subclass_safely():
    # Mocked exceptions should not match isinstance checks; we fall to generic.
    mock_exc = MagicMock(spec=Exception)
    presentation = categorize_error(mock_exc)
    assert presentation.category == "Unexpected"
