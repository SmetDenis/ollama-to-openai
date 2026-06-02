"""Tests for ollama_adapter.debug_prompt module."""

import pytest

from ollama_adapter.debug_prompt import (
    _wrap_in_fence,
    build_debug_content,
    is_debug_trigger,
    last_user_text,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("debug", True),
        ("DEBUG", True),
        ("  Debug  ", True),
        ('<user_input some=attr some="x">\ndebug\n</user_input>', True),
        ("<user_input><b>debug</b></user_input>", True),
        ("<user_input>please debug</user_input>", False),
        ("debug me", False),
        ("debugger", False),
        ("", False),
    ],
)
def test_is_debug_trigger(text, expected):
    assert is_debug_trigger(text) is expected


def test_is_debug_trigger_non_str():
    assert is_debug_trigger(None) is False
    assert is_debug_trigger([{"type": "text"}]) is False


class TestLastUserText:
    def test_returns_last_user_string(self):
        msgs = [{"role": "assistant", "content": "a"}, {"role": "user", "content": "hi"}]
        assert last_user_text(msgs) == "hi"

    def test_none_when_last_not_user(self):
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "a"}]
        assert last_user_text(msgs) is None

    def test_none_when_content_not_str(self):
        assert last_user_text([{"role": "user", "content": [{"type": "text"}]}]) is None

    def test_none_when_empty(self):
        assert last_user_text([]) is None


def test_wrap_in_fence_default_three_backticks():
    out = _wrap_in_fence("plain text")
    assert out.startswith("```\n")
    assert out.endswith("\n```")


def test_wrap_in_fence_grows_past_inner_backticks():
    out = _wrap_in_fence("a ``` b")
    assert out.startswith("````\n")
    assert out.endswith("\n````")


class TestBuildDebugContent:
    def test_inline_system_full_array(self, prompts_dir):
        messages = [{"role": "user", "content": "<user_input>\ndebug\n</user_input>"}]
        out = build_debug_content(messages, {"system_prompt_inline": "You are helpful."}, "m")
        assert out.startswith("```")
        assert out.rstrip().endswith("```")
        assert "═══ message[0] role=system ═══" in out
        assert "── BEGIN inline system_prompt ──" in out
        assert "You are helpful." in out
        assert "═══ message[1] role=user ═══" in out
        assert "<user_input>" in out  # trigger message shown verbatim with tags

    def test_no_system_source_leaves_array(self, prompts_dir):
        messages = [{"role": "user", "content": "<user_input>debug</user_input>"}]
        out = build_debug_content(messages, {}, "m")
        assert "role=system" not in out
        assert "═══ message[0] role=user ═══" in out
        assert "<user_input>debug</user_input>" in out

    def test_caching_note_present_when_enabled(self, prompts_dir):
        out = build_debug_content(
            [{"role": "user", "content": "debug"}],
            {"system_prompt_inline": "Hi", "prompt_caching": True},
            "m",
        )
        assert "── note: prompt_caching enabled" in out

    def test_render_error_shown_inline(self, prompts_dir):
        out = build_debug_content(
            [{"role": "user", "content": "debug"}],
            {"system_prompt_file": "missing.md"},
            "m",
        )
        assert "── PROMPT RENDER ERROR:" in out

    def test_non_string_content_as_json(self, prompts_dir):
        messages = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
        out = build_debug_content(messages, {}, "m")
        assert '"type": "text"' in out

    def test_caching_note_absent_on_render_error(self, prompts_dir):
        out = build_debug_content(
            [{"role": "user", "content": "debug"}],
            {"system_prompt_file": "missing.md", "prompt_caching": True},
            "m",
        )
        assert "── PROMPT RENDER ERROR:" in out
        assert "── note: prompt_caching enabled" not in out

    def test_replaces_existing_system_message(self, prompts_dir):
        messages = [
            {"role": "system", "content": "OLD"},
            {"role": "user", "content": "debug"},
        ]
        out = build_debug_content(messages, {"system_prompt_inline": "NEW SYSTEM"}, "m")
        assert "NEW SYSTEM" in out
        assert "OLD" not in out
        assert out.count("role=system") == 1
        assert "═══ message[0] role=system ═══" in out
        assert "═══ message[1] role=user ═══" in out

    def test_system_prompt_file_shows_file_markers(self, prompts_dir):
        (prompts_dir / "sys.md").write_text("FILE CONTENT")
        out = build_debug_content(
            [{"role": "user", "content": "debug"}],
            {"system_prompt_file": "sys.md"},
            "m",
        )
        assert "── BEGIN file: sys.md ──" in out
        assert "FILE CONTENT" in out
        assert "── END file: sys.md ──" in out

    def test_system_prompt_file_with_include_shows_nested_markers(self, prompts_dir):
        (prompts_dir / "snippet.md").write_text("SNIPPET BODY")
        (prompts_dir / "sys.md").write_text('ROOT BODY\n{% include "snippet.md" %}')
        out = build_debug_content(
            [{"role": "user", "content": "debug"}],
            {"system_prompt_file": "sys.md"},
            "m",
        )
        assert out.index("── BEGIN file: sys.md ──") < out.index("── include: snippet.md ──")
        assert out.index("SNIPPET BODY") < out.index("── end include: snippet.md ──")
        assert out.index("── end include: snippet.md ──") < out.index("── END file: sys.md ──")
