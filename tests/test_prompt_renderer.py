"""Tests for ollama_adapter.prompt_renderer module."""

import pytest

from ollama_adapter.prompt_renderer import (
    PromptRenderError,
    init_jinja_env,
    render_file,
    render_inline,
)


@pytest.fixture
def env(tmp_path):
    return init_jinja_env(tmp_path)


# ---------------------------------------------------------------------------
# render_inline
# ---------------------------------------------------------------------------


class TestRenderInline:
    def test_simple_substitution(self, env):
        assert render_inline(env, "Hello {{ name }}", {"name": "World"}) == "Hello World"

    def test_plain_text_passthrough(self, env):
        assert render_inline(env, "Just text", {}) == "Just text"

    def test_undefined_variable_raises(self, env):
        with pytest.raises(PromptRenderError) as exc_info:
            render_inline(env, "Hello {{ missing }}", {})
        assert exc_info.value.source == "inline"
        assert "missing" in str(exc_info.value)

    def test_syntax_error_raises_with_lineno(self, env):
        with pytest.raises(PromptRenderError) as exc_info:
            render_inline(env, "Line 1\n{% bad syntax %}", {})
        assert "line" in str(exc_info.value).lower()

    def test_sandbox_blocks_unsafe_attributes(self, env):
        with pytest.raises(PromptRenderError) as exc_info:
            render_inline(env, "{{ ''.__class__.__mro__ }}", {})
        assert "security" in str(exc_info.value).lower()

    def test_include_from_inline(self, env, tmp_path):
        (tmp_path / "snippet.md").write_text("snippet content")
        result = render_inline(env, '{% include "snippet.md" %}', {})
        assert result == "snippet content"


# ---------------------------------------------------------------------------
# render_file
# ---------------------------------------------------------------------------


class TestRenderFile:
    def test_simple_file(self, env, tmp_path):
        (tmp_path / "main.md").write_text("Main content")
        assert render_file(env, "main.md", {}) == "Main content"

    def test_with_variables(self, env, tmp_path):
        (tmp_path / "main.md").write_text("Hello {{ name }}")
        assert render_file(env, "main.md", {"name": "Bob"}) == "Hello Bob"

    def test_nested_include(self, env, tmp_path):
        (tmp_path / "leaf.md").write_text("leaf")
        (tmp_path / "mid.md").write_text('mid:{% include "leaf.md" %}')
        (tmp_path / "root.md").write_text('root:{% include "mid.md" %}')
        assert render_file(env, "root.md", {}) == "root:mid:leaf"

    def test_subdirectory_include(self, env, tmp_path):
        sub = tmp_path / "snippets"
        sub.mkdir()
        (sub / "intro.md").write_text("intro body")
        (tmp_path / "main.md").write_text('{% include "snippets/intro.md" %}')
        assert render_file(env, "main.md", {}) == "intro body"

    def test_cycle_detected(self, env, tmp_path):
        (tmp_path / "a.md").write_text('{% include "b.md" %}')
        (tmp_path / "b.md").write_text('{% include "a.md" %}')
        with pytest.raises(PromptRenderError) as exc_info:
            render_file(env, "a.md", {})
        assert "recursion" in str(exc_info.value).lower()

    def test_parent_directory_escape_rejected(self, env, tmp_path):
        outside = tmp_path.parent / "outside.md"
        outside.write_text("secret")
        (tmp_path / "main.md").write_text('{% include "../outside.md" %}')
        try:
            with pytest.raises(PromptRenderError):
                render_file(env, "main.md", {})
        finally:
            outside.unlink(missing_ok=True)

    def test_absolute_path_in_include_rejected(self, env, tmp_path):
        (tmp_path / "main.md").write_text('{% include "/etc/hosts" %}')
        with pytest.raises(PromptRenderError):
            render_file(env, "main.md", {})

    def test_missing_template_raises(self, env):
        with pytest.raises(PromptRenderError) as exc_info:
            render_file(env, "nope.md", {})
        assert "not found" in str(exc_info.value)
        assert exc_info.value.source == "nope.md"

    def test_binary_file_raises(self, env, tmp_path):
        (tmp_path / "bin.dat").write_bytes(b"\xff\xfe\x00\x01\x80")
        with pytest.raises(PromptRenderError) as exc_info:
            render_file(env, "bin.dat", {})
        assert "utf-8" in str(exc_info.value).lower()

    def test_undefined_variable_in_file_raises(self, env, tmp_path):
        (tmp_path / "main.md").write_text("Hello {{ missing }}")
        with pytest.raises(PromptRenderError) as exc_info:
            render_file(env, "main.md", {})
        assert exc_info.value.source == "main.md"


# ---------------------------------------------------------------------------
# PromptRenderError attributes
# ---------------------------------------------------------------------------


class TestPromptRenderError:
    def test_str_is_message(self):
        err = PromptRenderError("oops", source="x.md")
        assert str(err) == "oops"
        assert err.message == "oops"
        assert err.source == "x.md"
