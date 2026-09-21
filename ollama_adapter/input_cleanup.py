r"""Strip client-added label prefixes (Raycast's ``Text:`` line) from the incoming user text.

Some clients prepend a label line to the text they send — Raycast sends::

    Text:
    <what the user actually typed>

The label is noise: it reaches the model and it also breaks the `debug` keyword
check (``"Text:\ndebug"`` is not ``"debug"``). This module removes it from the
**last user message only**, and the routes call it right after input validation —
before debug detection and before the upstream request is assembled.

Matching is deliberately narrow (see the design spec): the label must be the whole
first line — prefix plus an optional run of spaces/tabs, then a line break — and it
is never stripped when nothing but whitespace would remain.
"""

import re
from functools import lru_cache
from typing import Any

from ollama_adapter import state

# Labels stripped when the `input_cleanup` section (or its `strip_prefixes` key) is absent.
DEFAULT_STRIP_PREFIXES = ("Text:", "Текст:")

INPUT_CLEANUP_KEYS = frozenset({"enabled", "strip_prefixes"})


def _section() -> dict[str, Any]:
    """Return the `input_cleanup` config section, or an empty dict when absent/malformed."""
    cfg = state.CONFIG.get("input_cleanup")
    if isinstance(cfg, dict):
        return cfg
    return {}


def configured_prefixes() -> tuple[str, ...]:
    """Return the label prefixes to strip (empty when the feature is switched off)."""
    cfg = _section()
    if not cfg.get("enabled", True):
        return ()
    raw = cfg.get("strip_prefixes")
    if raw is None:
        return DEFAULT_STRIP_PREFIXES
    if not isinstance(raw, list):
        return ()
    return tuple(p for p in raw if isinstance(p, str) and p.strip())


@lru_cache(maxsize=64)
def _prefix_pattern(prefix: str) -> re.Pattern[str]:
    r"""Compile the matcher for one label: leading blanks, the prefix, its own line break.

    Case-insensitive, so ``text:`` and ``ТЕКСТ:`` match too. The trailing ``\s*``
    swallows the blank lines between the label and the real text.
    """
    return re.compile(rf"^\s*{re.escape(prefix)}[ \t]*\r?\n\s*", re.IGNORECASE)


def _strip_one(text: str, prefix: str) -> str | None:
    """Return `text` without the `prefix` label line, or None when it does not apply."""
    match = _prefix_pattern(prefix).match(text)
    if match is None:
        return None
    rest = text[match.end() :]
    if not rest.strip():
        # The label was the entire message; keeping it beats sending an empty request.
        return None
    return rest


def strip_input_prefix(text: str) -> str:
    """Return `text` with a leading client label line removed (unchanged when there is none)."""
    if not isinstance(text, str) or not text:
        return text
    for prefix in configured_prefixes():
        stripped = _strip_one(text, prefix)
        if stripped is not None:
            state.logger.debug("Stripped client input prefix %r from the last user message", prefix)
            return stripped
    return text


def _clean_content_parts(parts: list[Any]) -> list[Any] | None:
    """Strip the label from the first text part; None when nothing changed.

    Only the first text part can carry the label — later parts are not at the start
    of the message.
    """
    for i, part in enumerate(parts):
        if not (isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)):
            continue
        cleaned = strip_input_prefix(part["text"])
        if cleaned == part["text"]:
            return None
        return [*parts[:i], {**part, "text": cleaned}, *parts[i + 1 :]]
    return None


def clean_last_user_message(messages: list[Any]) -> list[Any]:
    """Return `messages` with the label stripped from the last message iff it is a user one.

    The input is never mutated: on a hit a new list with a copied last message is
    returned, otherwise the original list is passed through. Handles both string
    content and an OpenAI content-part list.
    """
    if not messages:
        return messages
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        return messages

    content = last.get("content")
    new_content: Any
    if isinstance(content, str):
        cleaned = strip_input_prefix(content)
        if cleaned == content:
            return messages
        new_content = cleaned
    elif isinstance(content, list):
        cleaned_parts = _clean_content_parts(content)
        if cleaned_parts is None:
            return messages
        new_content = cleaned_parts
    else:
        return messages

    return [*messages[:-1], {**last, "content": new_content}]
