"""Thinking tag removal for streaming and non-streaming responses."""

import json
import re
from collections.abc import Callable, Generator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ollama_adapter import state

_THINKING_PATTERN = re.compile(r"^\s*<think(?:ing)?>(.*?)</think(?:ing)?>\s*", re.DOTALL | re.IGNORECASE)
_PREVIEW_LENGTH = 100


def remove_thinking_tags(content: str | None, model_id: str, *, remove_enabled: bool) -> str | None:
    """Remove <think>/<thinking> tags from the beginning of content if enabled.

    Log removed thinking content at DEBUG level.
    """
    if not remove_enabled or not content:
        return content

    match = _THINKING_PATTERN.match(content)
    if not match:
        return content

    thinking_content = match.group(1)
    cleaned_content = content[match.end() :]
    preview = (
        thinking_content[:_PREVIEW_LENGTH] + "..." if len(thinking_content) > _PREVIEW_LENGTH else thinking_content
    )
    state.logger.debug(
        "Removed thinking tags from model '%s'. Thinking content (%d chars): %s",
        model_id,
        len(thinking_content),
        preview,
    )
    return cleaned_content


class _StreamState(StrEnum):
    """State machine states for streaming tag removal."""

    DETECTING_OPEN_TAG = "DETECTING_OPEN_TAG"
    BUFFERING_THINKING = "BUFFERING_THINKING"
    DETECTING_CLOSE_TAG = "DETECTING_CLOSE_TAG"
    STREAMING_NORMAL = "STREAMING_NORMAL"


_OPEN_TAGS = ("<think>", "<thinking>")
_CLOSE_TAGS = ("</think>", "</thinking>")
_CLOSE_TAG_START = "</"
_BUFFER_FLUSH_SIZE = 1000


class ThinkingTagFilter:
    """Incremental filter that removes a leading `<think>`/`<thinking>` block from streamed text.

    Protocol-agnostic: `feed()` returns the text that is safe to emit now (possibly empty while
    a tag is being detected or thinking content is buffered); `flush()` returns whatever must be
    emitted when the stream ends. Every piece of input is re-dispatched immediately after a state
    transition, so text that shares a chunk with a tag boundary is never held back or lost.
    """

    def __init__(self, model_id: str) -> None:
        """Create a filter in tag-detection state; `model_id` is used for logging only."""
        self._model_id = model_id
        self._state = _StreamState.DETECTING_OPEN_TAG
        self._buffer = ""
        self._thinking_buffer = ""
        self._close_tag_buffer = ""
        self._strip_leading = False

    def feed(self, text: str) -> str:
        """Consume a streamed text fragment and return the part that can be emitted now."""
        if self._state == _StreamState.DETECTING_OPEN_TAG:
            return self._detect_open(text)
        if self._state == _StreamState.BUFFERING_THINKING:
            return self._buffer_thinking(text)
        if self._state == _StreamState.DETECTING_CLOSE_TAG:
            return self._detect_close(text)
        if self._strip_leading:
            text = text.lstrip()
            self._strip_leading = not text
        return text

    def flush(self) -> str:
        """Return the remaining text at end of stream and reset to pass-through mode."""
        state_at_end = self._state
        self._state = _StreamState.STREAMING_NORMAL
        if state_at_end == _StreamState.DETECTING_OPEN_TAG:
            out, self._buffer = self._buffer, ""
            return out
        if state_at_end in (_StreamState.BUFFERING_THINKING, _StreamState.DETECTING_CLOSE_TAG):
            fallback = self._thinking_buffer + self._buffer + self._close_tag_buffer
            self._thinking_buffer = self._buffer = self._close_tag_buffer = ""
            state.logger.warning(
                "Stream ended while buffering thinking content for model '%s'. "
                "No closing tag found. Outputting %d chars as fallback.",
                self._model_id,
                len(fallback),
            )
            return fallback if fallback.strip() else ""
        return ""

    def _detect_open(self, text: str) -> str:
        self._buffer += text
        stripped_lower = self._buffer.lstrip().lower()

        for tag in _OPEN_TAGS:
            if stripped_lower.startswith(tag):
                whitespace_len = len(self._buffer) - len(self._buffer.lstrip())
                rest = self._buffer[whitespace_len + len(tag) :]
                self._buffer = ""
                self._thinking_buffer = ""
                self._state = _StreamState.BUFFERING_THINKING
                state.logger.debug("Detected opening %s tag for model '%s'", tag, self._model_id)
                return self._buffer_thinking(rest)

        if any(tag.startswith(stripped_lower) for tag in _OPEN_TAGS):
            return ""

        self._state = _StreamState.STREAMING_NORMAL
        state.logger.debug("No thinking tag detected for model '%s'", self._model_id)
        out, self._buffer = self._buffer, ""
        return out

    def _buffer_thinking(self, text: str) -> str:
        self._buffer += text

        while _CLOSE_TAG_START in self._buffer:
            close_idx = self._buffer.index(_CLOSE_TAG_START)
            self._thinking_buffer += self._buffer[:close_idx]
            candidate = self._buffer[close_idx:]
            self._buffer = ""
            candidate_lower = candidate.lower()

            for tag in _CLOSE_TAGS:
                if candidate_lower.startswith(tag):
                    return self._finish_thinking(candidate[len(tag) :])

            if any(tag.startswith(candidate_lower) for tag in _CLOSE_TAGS):
                self._close_tag_buffer = candidate
                self._state = _StreamState.DETECTING_CLOSE_TAG
                return ""

            # Not a closing tag after all: keep "</" as thinking text and rescan the rest.
            self._thinking_buffer += _CLOSE_TAG_START
            self._buffer = candidate[len(_CLOSE_TAG_START) :]

        if len(self._buffer) > _BUFFER_FLUSH_SIZE:
            # Keep a trailing "<" so a close tag split right after it is still detected.
            keep = 1 if self._buffer.endswith("<") else 0
            cut = len(self._buffer) - keep
            self._thinking_buffer += self._buffer[:cut]
            self._buffer = self._buffer[cut:]
        return ""

    def _detect_close(self, text: str) -> str:
        candidate = self._close_tag_buffer + text
        self._close_tag_buffer = ""
        self._state = _StreamState.BUFFERING_THINKING
        return self._buffer_thinking(candidate)

    def _finish_thinking(self, after_tag: str) -> str:
        remainder = after_tag.lstrip()
        self._log_removed_thinking()
        self._state = _StreamState.STREAMING_NORMAL
        self._thinking_buffer = ""
        # Mirror the non-streaming regex: whitespace after the close tag is dropped
        # even when it arrives in later fragments.
        self._strip_leading = not remainder
        return remainder

    def _log_removed_thinking(self) -> None:
        preview = (
            self._thinking_buffer[:_PREVIEW_LENGTH] + "..."
            if len(self._thinking_buffer) > _PREVIEW_LENGTH
            else self._thinking_buffer
        )
        state.logger.debug(
            "Removed thinking tags from model '%s'. Thinking content (%d chars): %s",
            self._model_id,
            len(self._thinking_buffer),
            preview,
        )


def _record_usage(chunk: Any, usage: dict[str, int]) -> None:
    if chunk.usage:
        usage["prompt_tokens"] = chunk.usage.prompt_tokens
        usage["completion_tokens"] = chunk.usage.completion_tokens


@dataclass(frozen=True)
class StreamContext:
    """Configuration for stream processing."""

    model_id: str
    display_name: str
    make_chunk: Callable[[str, str], dict[str, Any]]
    remove_tags: bool = False


def process_stream(
    response_stream: Any,
    ctx: StreamContext,
    usage: dict[str, int],
) -> Generator[str]:
    """Process OpenAI streaming response into Ollama ndjson lines, optionally removing thinking tags.

    Shared logic for both /api/chat and /api/generate endpoints.

    Args:
        response_stream: OpenAI streaming response iterator.
        ctx: Stream processing configuration.
        usage: Mutable dict to store prompt_tokens and completion_tokens.

    Yields:
        JSON-encoded response lines ending with newline.

    """
    tag_filter = ThinkingTagFilter(ctx.model_id) if ctx.remove_tags else None

    def emit(content: str) -> str:
        return json.dumps(ctx.make_chunk(ctx.display_name, content)) + "\n"

    for chunk in response_stream:
        _record_usage(chunk, usage)
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if not content:
            continue
        out = tag_filter.feed(content) if tag_filter else content
        if out:
            yield emit(out)

    if tag_filter and (tail := tag_filter.flush()):
        yield emit(tail)
