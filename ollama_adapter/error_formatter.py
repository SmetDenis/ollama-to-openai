"""Translate runtime errors into user-facing assistant content.

Mirrors the existing `PromptRenderError → "[PROMPT ERROR] ..."` pattern so that
clients (notably Raycast) receive a readable explanation instead of an HTTP 500
when the upstream LLM provider — or the proxy itself — fails.
"""

from dataclasses import dataclass
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
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

DEFAULT_PREFIX = "[LLM ERROR]"


@dataclass(frozen=True)
class ErrorPresentation:
    """Categorised runtime error ready to be rendered as assistant content."""

    category: str
    message: str

    def render(self, prefix: str, *, include_type: bool, show_details: bool) -> str:
        """Compose the visible text from the configured prefix, category, and details."""
        details_part = self.message.strip() if show_details else ""

        if include_type and details_part:
            body = f"{self.category}: {details_part}"
        elif include_type:
            body = self.category
        elif details_part:
            body = details_part
        else:
            body = self.category

        return f"{prefix} {body}".rstrip()


def _extract_message(exc: BaseException) -> str:
    """Return a non-empty human-readable message for the exception."""
    raw = str(exc).strip()
    if raw:
        return raw
    return exc.__class__.__name__


# Ordered most-specific → least-specific. `APIError` is the OpenAI SDK base
# class for HTTP-shaped failures and therefore sits last; anything not matched
# falls through to "Unexpected".
_CATEGORY_TABLE: tuple[tuple[type[BaseException], str], ...] = (
    (RateLimitError, "Rate limit"),
    (AuthenticationError, "Auth"),
    (PermissionDeniedError, "Permission denied"),
    (NotFoundError, "Not found"),
    (UnprocessableEntityError, "Unprocessable"),
    (BadRequestError, "Bad request"),
    (ConflictError, "Conflict"),
    (APITimeoutError, "Timeout"),
    (APIConnectionError, "Connection"),
    (InternalServerError, "Upstream 5xx"),
    (APIError, "API"),
)


def categorize_error(exc: BaseException) -> ErrorPresentation:
    """Translate an exception (OpenAI SDK or generic) into a user-facing presentation."""
    msg = _extract_message(exc)
    for cls, category in _CATEGORY_TABLE:
        if isinstance(exc, cls):
            return ErrorPresentation(category, msg)
    return ErrorPresentation("Unexpected", msg)


def _config() -> dict[str, Any]:
    """Return the `error_handling` config dict (empty when unset)."""
    cfg = state.CONFIG.get("error_handling")
    if isinstance(cfg, dict):
        return cfg
    return {}


def is_enabled() -> bool:
    """Whether runtime errors should be translated into LLM-style responses.

    Defaults to True when the section is missing.
    """
    cfg = _config()
    return bool(cfg.get("enabled", True))


def format_error_text(exc: BaseException) -> str:
    """Compose the user-visible assistant text for an exception."""
    cfg = _config()
    prefix_raw = cfg.get("prefix")
    prefix = prefix_raw if isinstance(prefix_raw, str) and prefix_raw.strip() else DEFAULT_PREFIX
    include_type = bool(cfg.get("include_type", True))
    show_details = bool(cfg.get("show_details", True))

    presentation = categorize_error(exc)
    return presentation.render(prefix, include_type=include_type, show_details=show_details)
