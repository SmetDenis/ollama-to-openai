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

    INITIAL = "INITIAL"
    DETECTING_OPEN_TAG = "DETECTING_OPEN_TAG"
    BUFFERING_THINKING = "BUFFERING_THINKING"
    DETECTING_CLOSE_TAG = "DETECTING_CLOSE_TAG"
    STREAMING_NORMAL = "STREAMING_NORMAL"


_OPEN_TAGS = ("<think>", "<thinking>")
_CLOSE_TAGS = ("</think>", "</thinking>")
_MAX_DETECT_LENGTH = 20
_BUFFER_FLUSH_SIZE = 1000
_MAX_CLOSE_TAG_LENGTH = 15


class _StreamProcessor:
    """State machine that removes thinking tags from a streaming response."""

    def __init__(self, model_id: str, display_name: str, make_chunk: Callable[[str, str], dict[str, Any]]) -> None:
        self._model_id = model_id
        self._display_name = display_name
        self._make_chunk = make_chunk
        self._state = _StreamState.INITIAL
        self._buffer = ""
        self._thinking_buffer = ""
        self._close_tag_buffer = ""

    def process(self, response_stream: Any, usage: dict[str, int]) -> Generator[str]:
        """Process the entire stream, yielding JSON-encoded chunks."""
        for chunk in response_stream:
            if chunk.usage:
                usage["prompt_tokens"] = chunk.usage.prompt_tokens
                usage["completion_tokens"] = chunk.usage.completion_tokens

            if not chunk.choices:
                continue

            content = chunk.choices[0].delta.content
            if not content:
                continue

            yield from self._handle_content(content)

        yield from self._flush_remaining()

    def _handle_content(self, content: str) -> Generator[str]:
        """Dispatch content to the appropriate state handler."""
        handler = {
            _StreamState.INITIAL: self._handle_initial,
            _StreamState.DETECTING_OPEN_TAG: self._handle_detecting_open,
            _StreamState.BUFFERING_THINKING: self._handle_buffering,
            _StreamState.DETECTING_CLOSE_TAG: self._handle_detecting_close,
            _StreamState.STREAMING_NORMAL: self._handle_normal,
        }[self._state]
        yield from handler(content)

    def _handle_initial(self, content: str) -> Generator[str]:
        self._buffer = content
        self._state = _StreamState.DETECTING_OPEN_TAG
        yield from ()

    def _handle_detecting_open(self, content: str) -> Generator[str]:
        self._buffer += content
        buffer_lower = self._buffer.lstrip().lower()

        for tag in _OPEN_TAGS:
            if buffer_lower.startswith(tag):
                whitespace_len = len(self._buffer) - len(self._buffer.lstrip())
                self._buffer = self._buffer[whitespace_len + len(tag) :]
                self._thinking_buffer = ""
                self._state = _StreamState.BUFFERING_THINKING
                state.logger.debug("Detected opening %s tag for model '%s'", tag, self._model_id)
                return

        stripped = self._buffer.lstrip()
        if len(buffer_lower) > _MAX_DETECT_LENGTH or (stripped and stripped[0] != "<"):
            self._state = _StreamState.STREAMING_NORMAL
            state.logger.debug("No thinking tag detected for model '%s'", self._model_id)
            if self._buffer:
                yield self._emit(self._buffer)
                self._buffer = ""

    def _handle_buffering(self, content: str) -> Generator[str]:
        self._buffer += content

        if "</" in self._buffer:
            close_idx = self._buffer.index("</")
            self._thinking_buffer += self._buffer[:close_idx]
            self._close_tag_buffer = self._buffer[close_idx:]
            self._buffer = ""
            self._state = _StreamState.DETECTING_CLOSE_TAG
        elif len(self._buffer) > _BUFFER_FLUSH_SIZE:
            self._thinking_buffer += self._buffer
            self._buffer = ""
        yield from ()

    def _handle_detecting_close(self, content: str) -> Generator[str]:
        self._close_tag_buffer += content
        close_lower = self._close_tag_buffer.lower()

        for tag in _CLOSE_TAGS:
            if close_lower.startswith(tag):
                remainder = self._close_tag_buffer[len(tag) :].lstrip()
                self._log_removed_thinking()
                self._state = _StreamState.STREAMING_NORMAL
                self._thinking_buffer = ""
                self._close_tag_buffer = ""
                if remainder:
                    yield self._emit(remainder)
                return

        is_not_prefix = ">" in self._close_tag_buffer and not any(
            close_lower.startswith(t[: len(close_lower)]) for t in _CLOSE_TAGS
        )
        if len(self._close_tag_buffer) > _MAX_CLOSE_TAG_LENGTH or is_not_prefix:
            self._thinking_buffer += self._close_tag_buffer
            self._close_tag_buffer = ""
            self._state = _StreamState.BUFFERING_THINKING

    def _handle_normal(self, content: str) -> Generator[str]:
        yield self._emit(content)

    def _flush_remaining(self) -> Generator[str]:
        """Flush any buffered content at end of stream."""
        if self._state == _StreamState.DETECTING_OPEN_TAG and self._buffer:
            yield self._emit(self._buffer)
        elif self._state in (_StreamState.BUFFERING_THINKING, _StreamState.DETECTING_CLOSE_TAG):
            fallback = self._thinking_buffer + self._close_tag_buffer
            state.logger.warning(
                "Stream ended while buffering thinking content for model '%s'. "
                "No closing tag found. Outputting %d chars as fallback.",
                self._model_id,
                len(fallback),
            )
            if fallback.strip():
                yield self._emit(fallback)

    def _emit(self, content: str) -> str:
        return json.dumps(self._make_chunk(self._display_name, content)) + "\n"

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
    """Process OpenAI streaming response, optionally removing thinking tags.

    Shared logic for both /api/chat and /api/generate endpoints.

    Args:
        response_stream: OpenAI streaming response iterator.
        ctx: Stream processing configuration.
        usage: Mutable dict to store prompt_tokens and completion_tokens.

    Yields:
        JSON-encoded response lines ending with newline.

    """
    if ctx.remove_tags:
        processor = _StreamProcessor(ctx.model_id, ctx.display_name, ctx.make_chunk)
        yield from processor.process(response_stream, usage)
    else:
        for chunk in response_stream:
            if chunk.usage:
                usage["prompt_tokens"] = chunk.usage.prompt_tokens
                usage["completion_tokens"] = chunk.usage.completion_tokens

            if not chunk.choices:
                continue

            content = chunk.choices[0].delta.content
            if content:
                yield json.dumps(ctx.make_chunk(ctx.display_name, content)) + "\n"
