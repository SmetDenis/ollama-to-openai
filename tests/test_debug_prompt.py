"""Tests for ollama_adapter.debug_prompt module."""

import pytest

from ollama_adapter.debug_prompt import (
    _mask_secrets,
    _wrap_in_fence,
    build_config_view,
    build_debug_content,
    build_outgoing_view,
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


class TestMaskSecrets:
    def test_masks_authorization_keeping_edges(self):
        out = _mask_secrets({"Authorization": "Bearer sk-abcdef123456"})
        assert out["Authorization"] != "Bearer sk-abcdef123456"
        assert out["Authorization"].startswith("Bear")
        assert out["Authorization"].endswith("3456")

    def test_short_value_fully_masked(self):
        assert _mask_secrets({"api_key": "short"})["api_key"] == "****"

    def test_non_sensitive_untouched(self):
        assert _mask_secrets({"X-Provider": "openai"})["X-Provider"] == "openai"

    def test_key_forms_normalized(self):
        out = _mask_secrets({"X-API-Key": "abcdefghij", "api_key": "0123456789", "cookie": "sessionid"})
        assert out["X-API-Key"] == "abcd…ghij"
        assert out["api_key"] == "0123…6789"
        assert out["cookie"] == "****" or out["cookie"].startswith("sess")

    def test_recurses_into_nested_dicts_and_lists(self):
        out = _mask_secrets(
            {"headers": {"Authorization": "Bearer sk-longsecret-000"}, "list": [{"token": "abcdefghij"}]}
        )
        assert out["headers"]["Authorization"] != "Bearer sk-longsecret-000"
        assert out["list"][0]["token"] == "abcd…ghij"

    def test_does_not_mask_lookalike_keys(self):
        # legit params/headers that merely contain secret-word letters must survive intact
        data = {"max_tokens": 4096, "top_k": 5, "monkey": "value123456", "keynote": "value123456"}
        assert _mask_secrets(data) == data

    def test_masks_secret_words_as_whole_words(self):
        out = _mask_secrets({"access_token": "abcdefghij", "X-Auth-Token": "abcdefghij"})
        assert out["access_token"] == "abcd…ghij"
        assert out["X-Auth-Token"] == "abcd…ghij"


class TestBuildConfigView:
    def test_shape(self):
        openai_params = {"model_id": "openai/gpt-4o", "temperature": 0.7, "max_tokens": 100}
        adapter = {"remove_thinking_tags": True, "system_prompt_inline": "Hi"}
        headers = {"X-Provider": "openai"}
        view = build_config_view("GPT-4o", openai_params, adapter, headers)
        assert view["requested_model"] == "GPT-4o"
        assert view["resolved_model"] == "openai/gpt-4o"
        assert view["params"] == {"temperature": 0.7, "max_tokens": 100}
        assert "model_id" not in view["params"]
        assert view["headers"] == {"X-Provider": "openai"}
        assert view["adapter"] == {"remove_thinking_tags": True, "system_prompt_inline": "Hi"}

    def test_empty_config_still_exposes_resolved_model(self):
        view = build_config_view("gpt", {"model_id": "openai/raw"}, {}, {})
        assert view["resolved_model"] == "openai/raw"
        assert view["params"] == {}
        assert view["headers"] == {}
        assert view["adapter"] == {}


class TestBuildOutgoingView:
    def test_shape(self):
        openai_params = {"model_id": "openai/gpt-4o", "temperature": 0.7}
        extra_body = {"temperature": 0.7, "metadata": {"trace_name": "GPT-4o"}}
        merged = {"x-litellm-tags": "t", "Authorization": "Bearer xyz"}
        view = build_outgoing_view(openai_params, extra_body, merged)
        assert view["model"] == "openai/gpt-4o"
        assert view["extra_body"] == extra_body
        assert view["extra_headers"] == merged

    def test_none_values_become_empty(self):
        view = build_outgoing_view({"model_id": "m"}, None, None)
        assert view["extra_body"] == {}
        assert view["extra_headers"] == {}


class TestBuildDebugContentSections:
    def test_config_section_before_messages(self, prompts_dir):
        cfg = build_config_view("GPT-4o", {"model_id": "openai/gpt-4o", "temperature": 0.7}, {}, {})
        out = build_debug_content([{"role": "user", "content": "debug"}], {}, "GPT-4o", config_view=cfg)
        assert "═══ model config ═══" in out
        assert '"resolved_model": "openai/gpt-4o"' in out
        assert out.index("═══ model config ═══") < out.index("═══ message[0]")

    def test_outgoing_section_present(self, prompts_dir):
        view = build_outgoing_view({"model_id": "openai/gpt-4o"}, {"temperature": 0.7}, {})
        out = build_debug_content([{"role": "user", "content": "debug"}], {}, "GPT-4o", outgoing_view=view)
        assert "═══ outgoing request ═══" in out
        assert '"model": "openai/gpt-4o"' in out

    def test_secret_masked_in_rendered_output(self, prompts_dir):
        cfg = build_config_view("m", {"model_id": "x"}, {}, {"Authorization": "Bearer sk-supersecret-999"})
        out = build_debug_content([{"role": "user", "content": "debug"}], {}, "m", config_view=cfg)
        assert "sk-supersecret-999" not in out
        assert "Bear" in out

    def test_no_views_no_config_sections(self, prompts_dir):
        out = build_debug_content([{"role": "user", "content": "debug"}], {}, "m")
        assert "model config" not in out
        assert "outgoing request" not in out

    def test_both_sections_and_messages_ordered(self, prompts_dir):
        cfg = build_config_view("m", {"model_id": "x"}, {"system_prompt_inline": "SYS"}, {})
        out = build_debug_content(
            [{"role": "user", "content": "debug"}],
            {"system_prompt_inline": "SYS"},
            "m",
            config_view=cfg,
            outgoing_view=build_outgoing_view({"model_id": "x"}, {}, {}),
        )
        assert out.index("═══ model config ═══") < out.index("═══ outgoing request ═══")
        assert out.index("═══ outgoing request ═══") < out.index("═══ message[0]")
        assert "SYS" in out
