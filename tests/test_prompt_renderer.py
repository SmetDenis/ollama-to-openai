"""Tests for ollama_adapter.prompt_renderer module."""

import pytest

from ollama_adapter.prompt_renderer import (
    PromptRenderError,
    init_jinja_env,
    render_file,
    render_file_debug,
    render_inline,
    render_inline_debug,
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


# ---------------------------------------------------------------------------
# render_file_debug
# ---------------------------------------------------------------------------


class TestRenderFileDebug:
    def test_single_file_wrapped_in_file_markers(self, env, tmp_path):
        (tmp_path / "main.md").write_text("Main content")
        result = render_file_debug(env, "main.md", {})
        assert "── BEGIN file: main.md ──" in result
        assert "Main content" in result
        assert "── END file: main.md ──" in result

    def test_nested_includes_nest_in_order(self, env, tmp_path):
        (tmp_path / "inner.md").write_text("INNER")
        (tmp_path / "outer.md").write_text('OUTER\n{% include "inner.md" %}')
        (tmp_path / "root.md").write_text('ROOT\n{% include "outer.md" %}')
        r = render_file_debug(env, "root.md", {})
        assert r.index("── BEGIN file: root.md ──") < r.index("── include: outer.md ──")
        assert r.index("── include: outer.md ──") < r.index("── include: inner.md ──")
        assert r.index("── end include: inner.md ──") < r.index("── end include: outer.md ──")
        assert r.index("── end include: outer.md ──") < r.index("── END file: root.md ──")

    def test_variables_substituted(self, env, tmp_path):
        (tmp_path / "main.md").write_text("Hello {{ name }}")
        result = render_file_debug(env, "main.md", {"name": "World"})
        assert "Hello World" in result

    def test_missing_file_raises(self, env):
        with pytest.raises(PromptRenderError) as exc_info:
            render_file_debug(env, "nope.md", {})
        assert "not found" in str(exc_info.value)


# ---------------------------------------------------------------------------
# render_inline_debug
# ---------------------------------------------------------------------------


class TestRenderInlineDebug:
    def test_inline_root_and_include_markers(self, env, tmp_path):
        (tmp_path / "snip.md").write_text("SNIP")
        result = render_inline_debug(env, 'HELLO {% include "snip.md" %}', {})
        assert "── BEGIN inline system_prompt ──" in result
        assert "── include: snip.md ──" in result
        assert "SNIP" in result
        assert "── end include: snip.md ──" in result
        assert "── END inline system_prompt ──" in result

    def test_inline_without_include(self, env):
        result = render_inline_debug(env, "Just text {{ x }}", {"x": "1"})
        assert "── BEGIN inline system_prompt ──" in result
        assert "Just text 1" in result
        assert "── END inline system_prompt ──" in result


# ---------------------------------------------------------------------------
# PromptRenderError attributes
# ---------------------------------------------------------------------------


class TestPromptRenderError:
    def test_str_is_message(self):
        err = PromptRenderError("oops", source="x.md")
        assert str(err) == "oops"
        assert err.message == "oops"
        assert err.source == "x.md"
