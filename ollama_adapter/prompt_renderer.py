"""Jinja2-based prompt renderer with sandboxing, hot-reload, and explicit errors."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from jinja2 import (
    FileSystemLoader,
    StrictUndefined,
    TemplateNotFound,
    TemplateSyntaxError,
    UndefinedError,
)
from jinja2.exceptions import SecurityError
from jinja2.sandbox import SandboxedEnvironment


class PromptRenderError(Exception):
    """Raised when prompt rendering fails.

    Carries a human-readable message safe to expose to API clients and the
    template path or the literal 'inline' identifier in `source`.
    """

    def __init__(self, message: str, *, source: str) -> None:
        """Store the client-safe `message` and the originating `source` identifier."""
        super().__init__(message)
        self.message = message
        self.source = source

    def __str__(self) -> str:
        """Return the client-safe message (without exception type prefix)."""
        return self.message


def init_jinja_env(base_dir: Path) -> SandboxedEnvironment:
    """Construct a SandboxedEnvironment rooted at `base_dir`.

    `cache_size=0` and `auto_reload=True` preserve hot-reload semantics:
    every render re-reads the template from disk.
    """
    return SandboxedEnvironment(
        loader=FileSystemLoader(str(base_dir), encoding="utf-8"),
        undefined=StrictUndefined,
        autoescape=False,
        auto_reload=True,
        cache_size=0,
        keep_trailing_newline=False,
        trim_blocks=False,
        lstrip_blocks=False,
    )


def render_file(env: SandboxedEnvironment, template_path: str, variables: dict[str, Any]) -> str:
    """Render a template file relative to env's base directory."""
    return _wrap_render(
        lambda: env.get_template(template_path).render(variables),
        f"template '{template_path}'",
        template_path,
    )


def render_inline(env: SandboxedEnvironment, text: str, variables: dict[str, Any]) -> str:
    """Render an inline template string, resolving includes via env's loader."""
    return _wrap_render(
        lambda: env.from_string(text).render(variables),
        "inline prompt",
        "inline",
    )


def _wrap_render(render_callable: Callable[[], str], source_label: str, source_attr: str) -> str:
    """Run `render_callable`, translating Jinja2 exceptions into `PromptRenderError`."""
    try:
        return render_callable()
    except TemplateNotFound as exc:
        msg = f"{source_label}: template not found: '{exc.name}'"
        raise PromptRenderError(msg, source=source_attr) from exc
    except TemplateSyntaxError as exc:
        msg = f"{source_label} line {exc.lineno}: syntax error: {exc.message}"
        raise PromptRenderError(msg, source=source_attr) from exc
    except UndefinedError as exc:
        msg = f"{source_label}: {exc.message or 'undefined value'}"
        raise PromptRenderError(msg, source=source_attr) from exc
    except SecurityError as exc:
        msg = f"{source_label}: security violation: {exc!s}"
        raise PromptRenderError(msg, source=source_attr) from exc
    except RecursionError as exc:
        msg = f"{source_label}: recursion limit exceeded (possible include cycle)"
        raise PromptRenderError(msg, source=source_attr) from exc
    except UnicodeDecodeError as exc:
        msg = f"{source_label}: not a valid UTF-8 text file"
        raise PromptRenderError(msg, source=source_attr) from exc
