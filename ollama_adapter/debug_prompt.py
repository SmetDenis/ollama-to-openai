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

# Field names whose values are masked in debug output. Matched as WHOLE words after
# splitting the key on separators and camelCase boundaries, so "X-Auth-Token" and
# "api_key" hit while look-alikes such as "max_tokens" or "monkey" do not.
_SENSITIVE_WORDS = frozenset(
    {
        "authorization",
        "auth",
        "token",
        "apikey",
        "key",
        "secret",
        "password",
        "passwd",
        "pwd",
        "cookie",
        "credential",
        "credentials",
        "bearer",
        "session",
        "sessionid",
    }
)
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_WORD_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_MASK_KEEP_EDGE = 4


def _is_sensitive_key(key: str) -> bool:
    """Return True when `key` names a secret-bearing field (auth, api key, token, ...).

    The key is split into words on separators and camelCase boundaries; a hit requires
    a whole word from `_SENSITIVE_WORDS` (so "max_tokens" -> {max, tokens} is safe).
    """
    normalized = _CAMEL_BOUNDARY_RE.sub("_", key).lower()
    words = {w for w in _WORD_SPLIT_RE.split(normalized) if w}
    return bool(words & _SENSITIVE_WORDS)


def _mask_value(value: Any) -> str:
    """Mask a secret value, keeping the first/last few chars as a fingerprint.

    Short values (<= 2*edge) collapse to ``****`` so nothing meaningful leaks.
    """
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if len(text) <= _MASK_KEEP_EDGE * 2:
        return "****"
    return f"{text[:_MASK_KEEP_EDGE]}…{text[-_MASK_KEEP_EDGE:]}"


def _mask_secrets(obj: Any) -> Any:
    """Return a deep copy of `obj` with values under sensitive keys masked.

    Recurses through dicts and lists; any value whose key looks secret-bearing
    (see `_is_sensitive_key`) is replaced by `_mask_value`. Scalars pass through.
    """
    if isinstance(obj, dict):
        return {k: (_mask_value(v) if _is_sensitive_key(str(k)) else _mask_secrets(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_mask_secrets(item) for item in obj]
    return obj


def _render_json_section(title: str, data: dict[str, Any]) -> str:
    """Render a titled config section as pretty-printed JSON with secrets masked."""
    body = json.dumps(_mask_secrets(data), ensure_ascii=False, indent=2)
    return f"═══ {title} ═══\n{body}"


def build_config_view(
    model_id: str, openai_params: dict[str, Any], adapter_params: dict[str, Any], headers: dict[str, Any]
) -> dict[str, Any]:
    """Shape the final combined model configuration (post IP-routing) for debug output.

    `resolved_model` is the upstream model actually used; `params` is everything sent
    to the OpenAI SDK except the internal `model_id` key.
    """
    return {
        "requested_model": model_id,
        "resolved_model": openai_params.get("model_id"),
        "params": {k: v for k, v in openai_params.items() if k != "model_id"},
        "headers": dict(headers or {}),
        "adapter": dict(adapter_params or {}),
    }


def build_outgoing_view(
    openai_params: dict[str, Any],
    extra_body: dict[str, Any] | None,
    merged_headers: dict[str, Any] | None,
) -> dict[str, Any]:
    """Shape the request that actually goes upstream (body + merged trace headers).

    `messages` are omitted here on purpose — they are rendered verbatim below as the
    ``message[i]`` sections.
    """
    return {
        "model": openai_params.get("model_id"),
        "extra_body": dict(extra_body or {}),
        "extra_headers": dict(merged_headers or {}),
    }


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


def build_debug_content(
    messages: list[dict[str, Any]],
    adapter_params: dict[str, Any],
    model_id: str,
    *,
    config_view: dict[str, Any] | None = None,
    outgoing_view: dict[str, Any] | None = None,
) -> str:
    """Assemble the full debug output in a single fenced block.

    Optional `config_view` (final combined config) and `outgoing_view` (what actually
    goes upstream) are rendered as JSON sections above the compiled `messages` array.
    """
    sections: list[str] = []
    if config_view is not None:
        sections.append(_render_json_section("model config", config_view))
    if outgoing_view is not None:
        sections.append(_render_json_section("outgoing request", outgoing_view))

    rendered_system, rendered_ok = _render_system_with_markers(adapter_params, model_id)
    if rendered_system is not None:
        if rendered_ok and adapter_params.get("prompt_caching"):
            note = "── note: prompt_caching enabled (sent as cache_control text block) ──"
            rendered_system = f"{note}\n{rendered_system}"
        final_messages = place_system_message(messages, rendered_system)
    else:
        final_messages = messages
    sections.append(_format_messages(final_messages))

    return _wrap_in_fence("\n\n".join(sections))
