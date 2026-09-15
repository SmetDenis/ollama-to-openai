"""Tests for ThinkingTagFilter (protocol-agnostic streaming tag removal)."""

import json

import pytest

from ollama_adapter.thinking import StreamContext, ThinkingTagFilter, process_stream

from .conftest import make_mock_chunk


def _run(fragments: list[str]) -> str:
    f = ThinkingTagFilter("m")
    return "".join(f.feed(x) for x in fragments) + f.flush()


@pytest.mark.parametrize(
    ("fragments", "expected"),
    [
        (["Hello", " world"], "Hello world"),
        (["<think>x</think>Hello"], "Hello"),
        (["<think>x</think>Hello", "!"], "Hello!"),
        (["<think>", "x</think>Hi"], "Hi"),
        (["<thi", "nk>", "internal", "</thi", "nk>", "output"], "output"),
        (["  ", "<THINKING>", "x", "</Thinking>", "  Y"], "Y"),
        (["<think>a </b> b</think>ans"], "ans"),
        (["<think>a </", "b> b </think>ans"], "ans"),
        (["<b>bold</b>"], "<b>bold</b>"),
        (["Hello <think>x</think>"], "Hello <think>x</think>"),
    ],
)
def test_feed_flush(fragments, expected):
    assert _run(fragments) == expected


def test_emits_as_soon_as_tag_is_ruled_out():
    f = ThinkingTagFilter("m")
    assert f.feed("<") == ""
    assert f.feed("p>text") == "<p>text"
    assert f.feed(" more") == " more"


def test_close_tag_and_answer_in_same_fragment_emit_immediately():
    f = ThinkingTagFilter("m")
    assert f.feed("<think>reason") == ""
    assert f.feed("</think>Answer") == "Answer"
    assert f.flush() == ""


def test_unclosed_tag_falls_back_to_thinking_text():
    assert _run(["<think>", "reasoning", "</"]) == "reasoning</"


def test_unclosed_whitespace_only_thinking_yields_nothing():
    assert _run(["<think>", "   "]) == ""


def test_whitespace_only_detection_buffer_is_flushed():
    assert _run(["  "]) == "  "


def test_long_thinking_split_close_tag_after_flush_threshold():
    body = "x" * 1500 + "<"
    assert _run(["<think>", body, "/think>done"]) == "done"


def test_process_stream_uses_filter():
    ctx = StreamContext(model_id="m", display_name="M", make_chunk=lambda _d, c: {"c": c}, remove_tags=True)
    lines = list(process_stream(iter([make_mock_chunk("<think>"), make_mock_chunk("x</think>Hi")]), ctx, {}))
    assert "".join(json.loads(line)["c"] for line in lines) == "Hi"


def test_many_non_closing_tags_in_one_fragment_do_not_recurse():
    body = "</b>" * 5000
    assert _run(["<think>", body + "</think>ok"]) == "ok"
