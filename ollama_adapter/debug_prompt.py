"""Debug short-circuit: compile and display the prompt instead of calling the model.

When a client's input reduces to the keyword `debug` (after stripping markup),
the adapter returns the fully compiled `messages` array rendered as a fenced text block — with visible markers at
every `{% include %}` boundary — instead of forwarding to the upstream model.
"""

import json
import re
from typing import Any

from ollama_adapter import state
from ollama_adapter.models import _collect_prompt_vars, _select_prompt_source, place_system_message
from ollama_adapter.prompt_renderer import PromptRenderError, render_file_debug, render_inline_debug

_TAG_RE = re.compile(r"<[^>]*>")
_DEBUG_KEYWORD = "debug"


def _strip_tags(text: str) -> str:
    """Remove every XML/HTML-like tag (the Raycast `<user_input>` envelope, etc.).

    Tag-level, not envelope-level: e.g. ``de<br>bug`` collapses to ``debug``. Harmless
    in practice (real clients never split the keyword with tags); see the spec's risks.
    """
    return _TAG_RE.sub("", text)


def is_debug_trigger(text: Any) -> bool:
    """Return True when `text`, stripped of markup and whitespace, equals `debug` (any case)."""
    if not isinstance(text, str):
        return False
    return _strip_tags(text).strip().lower() == _DEBUG_KEYWORD


def last_user_text(messages: list[dict[str, Any]]) -> str | None:
    """Return the last message's string content iff it is a user message; else None."""
    if not messages:
        return None
    last = messages[-1]
    if isinstance(last, dict) and last.get("role") == "user":
        content = last.get("content")
        if isinstance(content, str):
            return content
    return None


def _render_system_with_markers(adapter_params: dict[str, Any], model_id: str) -> tuple[str | None, bool]:
    """Render the configured system prompt with include markers.

    Returns ``(content, rendered_ok)``. ``content`` is None when no system prompt is
    configured. On render failure, ``content`` is a single PROMPT RENDER ERROR marker
    (non-fatal — the broken template is what the user wants to see) and ``rendered_ok``
    is False so callers can suppress success-only annotations (e.g. the caching note).
    """
    kind, value = _select_prompt_source(adapter_params, model_id)
    if kind is None:
        return None, False

    env = state.jinja_env
    assert env is not None, "Jinja2 environment is not initialized"  # noqa: S101
    variables = _collect_prompt_vars(adapter_params)

    assert value is not None  # noqa: S101
    try:
        if kind == "file":
            return render_file_debug(env, value, variables), True
        return render_inline_debug(env, value, variables), True
    except PromptRenderError as exc:
        return f"── PROMPT RENDER ERROR: {exc} ──", False


def _format_messages(messages: list[dict[str, Any]]) -> str:
    """Render the messages array as role-labelled sections; non-string content as JSON."""
    sections: list[str] = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "?") if isinstance(msg, dict) else "?"
        content = msg.get("content") if isinstance(msg, dict) else msg
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, indent=2)
        sections.append(f"═══ message[{i}] role={role} ═══\n{content}")
    return "\n\n".join(sections)


def _wrap_in_fence(text: str) -> str:
    """Wrap `text` in a fenced code block longer than any backtick run inside it."""
    longest = run = 0
    for ch in text:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    fence = "`" * max(3, longest + 1)
    return f"{fence}\n{text}\n{fence}"


def build_debug_content(messages: list[dict[str, Any]], adapter_params: dict[str, Any], model_id: str) -> str:
    """Assemble the full debug output: compiled messages array in a fenced block."""
    rendered_system, rendered_ok = _render_system_with_markers(adapter_params, model_id)
    if rendered_system is not None:
        if rendered_ok and adapter_params.get("prompt_caching"):
            note = "── note: prompt_caching enabled (sent as cache_control text block) ──"
            rendered_system = f"{note}\n{rendered_system}"
        final_messages = place_system_message(messages, rendered_system)
    else:
        final_messages = messages
    return _wrap_in_fence(_format_messages(final_messages))
