"""Tests for ollama_adapter.thinking module."""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from ollama_adapter.thinking import StreamContext, process_stream, remove_thinking_tags

from .conftest import make_mock_chunk


# ---------------------------------------------------------------------------
# remove_thinking_tags (non-streaming)
# ---------------------------------------------------------------------------


class TestRemoveThinkingTags:
    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            ("<think>reasoning</think>Hello", "Hello"),
            ("<thinking>reasoning</thinking>Hello", "Hello"),
            ("<THINK>reasoning</THINK>Hello", "Hello"),
            ("<Thinking>reasoning</Thinking>Hello", "Hello"),
        ],
    )
    def test_remove_tags(self, content, expected):
        assert remove_thinking_tags(content, "model-1", remove_enabled=True) == expected

    def test_remove_with_leading_whitespace(self):
        result = remove_thinking_tags("  <think>x</think>Hello", "m", remove_enabled=True)
        assert result == "Hello"

    def test_no_tags_unchanged(self):
        assert remove_thinking_tags("Hello world", "m", remove_enabled=True) == "Hello world"

    def test_disabled_returns_unchanged(self):
        content = "<think>x</think>Hello"
        assert remove_thinking_tags(content, "m", remove_enabled=False) == content

    def test_none_content(self):
        assert remove_thinking_tags(None, "m", remove_enabled=True) is None

    def test_empty_string(self):
        assert remove_thinking_tags("", "m", remove_enabled=True) == ""

    def test_tag_not_at_start(self):
        content = "Hello <think>x</think> world"
        assert remove_thinking_tags(content, "m", remove_enabled=True) == content

    def test_multiline_thinking(self):
        content = "<think>line1\nline2\nline3</think>Result"
        assert remove_thinking_tags(content, "m", remove_enabled=True) == "Result"


# ---------------------------------------------------------------------------
# Helpers for streaming tests
# ---------------------------------------------------------------------------


def _make_chunk_fn(display_name: str, content: str) -> dict[str, Any]:
    return {"model": display_name, "content": content, "done": False}


def _collect_content(chunks: list[str]) -> str:
    """Extract concatenated content from JSON chunk strings."""
    result = ""
    for line in chunks:
        data = json.loads(line)
        result += data.get("content", "")
    return result


def _run_stream(stream_chunks, *, remove_tags: bool = True) -> list[str]:
    """Run process_stream on a list of mock chunks, return yielded lines."""
    ctx = StreamContext(
        model_id="test-model",
        display_name="Test",
        make_chunk=_make_chunk_fn,
        remove_tags=remove_tags,
    )
    usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
    return list(process_stream(iter(stream_chunks), ctx, usage))


# ---------------------------------------------------------------------------
# _StreamProcessor FSM tests
# ---------------------------------------------------------------------------


class TestStreamProcessor:
    def test_no_tags_passthrough(self):
        chunks = [make_mock_chunk("Hello"), make_mock_chunk(" world")]
        lines = _run_stream(chunks, remove_tags=True)
        assert _collect_content(lines) == "Hello world"

    def test_thinking_then_content(self):
        chunks = [
            make_mock_chunk("<think>"),
            make_mock_chunk("reasoning"),
            make_mock_chunk("</think>"),
            make_mock_chunk("Answer"),
        ]
        lines = _run_stream(chunks)
        assert _collect_content(lines) == "Answer"

    def test_thinking_tag_variant(self):
        chunks = [
            make_mock_chunk("<thinking>"),
            make_mock_chunk("deep thought"),
            make_mock_chunk("</thinking>"),
            make_mock_chunk("Result"),
        ]
        lines = _run_stream(chunks)
        assert _collect_content(lines) == "Result"

    def test_tag_split_across_chunks(self):
        chunks = [
            make_mock_chunk("<thi"),
            make_mock_chunk("nk>"),
            make_mock_chunk("internal"),
            make_mock_chunk("</thi"),
            make_mock_chunk("nk>"),
            make_mock_chunk("output"),
        ]
        lines = _run_stream(chunks)
        assert _collect_content(lines) == "output"

    def test_long_detection_fallback(self):
        chunks = [make_mock_chunk("This is normal text that definitely isn't a tag")]
        lines = _run_stream(chunks)
        assert _collect_content(lines) == "This is normal text that definitely isn't a tag"

    def test_unclosed_tag_fallback(self):
        chunks = [
            make_mock_chunk("<think>"),
            make_mock_chunk("reasoning without close"),
        ]
        lines = _run_stream(chunks)
        content = _collect_content(lines)
        assert "reasoning without close" in content

    def test_thinking_with_remainder(self):
        chunks = [
            make_mock_chunk("<think>"),
            make_mock_chunk("thinking"),
            make_mock_chunk("</think>And the answer"),
        ]
        lines = _run_stream(chunks)
        content = _collect_content(lines)
        assert "And the answer" in content
        assert "thinking" not in content

    def test_whitespace_before_tag(self):
        chunks = [
            make_mock_chunk("  "),
            make_mock_chunk("<think>"),
            make_mock_chunk("x"),
            make_mock_chunk("</think>"),
            make_mock_chunk("Y"),
        ]
        lines = _run_stream(chunks)
        assert _collect_content(lines) == "Y"

    def test_empty_choices_skipped(self):
        chunks = [
            make_mock_chunk(None),
            make_mock_chunk("Hello"),
        ]
        lines = _run_stream(chunks, remove_tags=True)
        assert _collect_content(lines) == "Hello"

    def test_usage_tracking(self):
        usage_mock = MagicMock()
        usage_mock.prompt_tokens = 15
        usage_mock.completion_tokens = 20

        chunk = make_mock_chunk("Hi")
        chunk.usage = usage_mock

        ctx = StreamContext(
            model_id="m", display_name="M", make_chunk=_make_chunk_fn, remove_tags=False
        )
        usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        list(process_stream(iter([chunk]), ctx, usage))
        assert usage["prompt_tokens"] == 15
        assert usage["completion_tokens"] == 20


# ---------------------------------------------------------------------------
# process_stream integration
# ---------------------------------------------------------------------------


class TestProcessStream:
    def test_without_removal(self):
        chunks = [make_mock_chunk("<think>x</think>Hello")]
        lines = _run_stream(chunks, remove_tags=False)
        assert _collect_content(lines) == "<think>x</think>Hello"

    def test_json_format(self):
        chunks = [make_mock_chunk("Hi")]
        lines = _run_stream(chunks, remove_tags=False)
        assert len(lines) == 1
        assert lines[0].endswith("\n")
        data = json.loads(lines[0])
        assert "model" in data
        assert "content" in data
