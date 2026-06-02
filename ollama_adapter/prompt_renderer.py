"""Jinja2-based prompt renderer with sandboxing, hot-reload, and explicit errors."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from jinja2 import (
    Environment,
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


def _make_env(loader: FileSystemLoader) -> SandboxedEnvironment:
    """Construct a SandboxedEnvironment with hot-reload semantics for `loader`."""
    return SandboxedEnvironment(
        loader=loader,
        undefined=StrictUndefined,
        autoescape=False,
        auto_reload=True,
        cache_size=0,
        keep_trailing_newline=False,
        trim_blocks=False,
        lstrip_blocks=False,
    )


def init_jinja_env(base_dir: Path) -> SandboxedEnvironment:
    """Construct a SandboxedEnvironment rooted at `base_dir`.

    `cache_size=0` and `auto_reload=True` preserve hot-reload semantics:
    every render re-reads the template from disk.
    """
    return _make_env(FileSystemLoader(str(base_dir), encoding="utf-8"))


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


class _DebugMarkerLoader(FileSystemLoader):
    """FileSystemLoader that wraps each template's source in visible boundary markers.

    The template matching `root_name` is wrapped as a 'file:' boundary; all others
    (resolved through `{% include %}`) as 'include:' boundaries. Markers are literal
    text, so they pass through rendering verbatim and includes nest automatically.
    """

    def __init__(self, searchpath: list[str] | str, root_name: str | None) -> None:
        super().__init__(searchpath, encoding="utf-8")
        self._root_name = root_name

    def get_source(self, environment: Environment, template: str) -> tuple[str, str, Callable[[], bool]]:
        source, filename, uptodate = super().get_source(environment, template)
        if template == self._root_name:
            begin, end = f"── BEGIN file: {template} ──", f"── END file: {template} ──"
        else:
            begin, end = f"── include: {template} ──", f"── end include: {template} ──"
        sep = "" if source.endswith("\n") else "\n"
        return f"{begin}\n{source}{sep}{end}", filename, uptodate


def _build_debug_env(env: SandboxedEnvironment, root_name: str | None) -> SandboxedEnvironment:
    """Build a sibling env that marks include boundaries, reusing `env`'s searchpath."""
    if not isinstance(env.loader, FileSystemLoader):
        msg = f"_build_debug_env requires a FileSystemLoader-backed env, got {type(env.loader).__name__}"
        raise TypeError(msg)
    return _make_env(_DebugMarkerLoader(env.loader.searchpath, root_name))


def render_file_debug(env: SandboxedEnvironment, template_path: str, variables: dict[str, Any]) -> str:
    """Render a template file with include-boundary markers (for debug output).

    Note: the injected markers shift template line numbers, so a PromptRenderError's reported
    line may be offset by the marker lines above it.
    """
    debug_env = _build_debug_env(env, root_name=template_path)
    return _wrap_render(
        lambda: debug_env.get_template(template_path).render(variables),
        f"template '{template_path}'",
        template_path,
    )


def render_inline_debug(env: SandboxedEnvironment, text: str, variables: dict[str, Any]) -> str:
    """Render an inline template with include-boundary markers (for debug output).

    The inline root is wrapped manually since `from_string` bypasses the loader.
    """
    debug_env = _build_debug_env(env, root_name=None)
    inner = _wrap_render(
        lambda: debug_env.from_string(text).render(variables),
        "inline prompt",
        "inline",
    )
    return f"── BEGIN inline system_prompt ──\n{inner}\n── END inline system_prompt ──"


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
