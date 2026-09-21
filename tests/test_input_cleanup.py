"""Tests for ollama_adapter.input_cleanup — stripping client label prefixes."""

import pytest

from ollama_adapter import state
from ollama_adapter.input_cleanup import (
    DEFAULT_STRIP_PREFIXES,
    clean_last_user_message,
    configured_prefixes,
    strip_input_prefix,
)


class TestConfiguredPrefixes:
    def test_defaults_without_section(self):
        state.CONFIG = {}
        assert configured_prefixes() == DEFAULT_STRIP_PREFIXES

    def test_defaults_when_only_enabled_given(self):
        state.CONFIG = {"input_cleanup": {"enabled": True}}
        assert configured_prefixes() == DEFAULT_STRIP_PREFIXES

    def test_disabled(self):
        state.CONFIG = {"input_cleanup": {"enabled": False}}
        assert configured_prefixes() == ()

    def test_custom_list_replaces_defaults(self):
        state.CONFIG = {"input_cleanup": {"strip_prefixes": ["Prompt:"]}}
        assert configured_prefixes() == ("Prompt:",)

    def test_empty_list_strips_nothing(self):
        state.CONFIG = {"input_cleanup": {"strip_prefixes": []}}
        assert configured_prefixes() == ()

    def test_malformed_section_ignored(self):
        state.CONFIG = {"input_cleanup": "nope"}
        assert configured_prefixes() == DEFAULT_STRIP_PREFIXES

    def test_non_list_prefixes_strip_nothing(self):
        state.CONFIG = {"input_cleanup": {"strip_prefixes": "Text:"}}
        assert configured_prefixes() == ()


class TestStripInputPrefix:
    @pytest.fixture(autouse=True)
    def _default_config(self):
        state.CONFIG = {}

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Text:\ndebug", "debug"),
            ("Текст:\ndebug", "debug"),
            ("text:\ndebug", "debug"),  # case-insensitive
            ("ТЕКСТ:\nпривет", "привет"),
            ("Text:\r\ndebug", "debug"),  # CRLF
            ("Text:   \ndebug", "debug"),  # trailing spaces on the label line
            ("Text:\n\n\ndebug", "debug"),  # blank lines after the label
            ("\nText:\ndebug", "debug"),  # leading blank line before the label
            ("Text:\nhello\nText:\nworld", "hello\nText:\nworld"),  # only the leading label
        ],
    )
    def test_strips(self, text, expected):
        assert strip_input_prefix(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "Text: debug",  # same line — not the configured shape
            "Text:",  # label only, nothing would remain
            "Text:\n   \n",  # only whitespace would remain
            "debug",
            "Please read. Text:\ndebug",  # not at the start
            "<user_input>Text:\ndebug</user_input>",  # inside markup — left alone by design
            "Textual:\ndebug",
            "",
        ],
    )
    def test_leaves_unchanged(self, text):
        assert strip_input_prefix(text) == text

    def test_respects_disabled_flag(self):
        state.CONFIG = {"input_cleanup": {"enabled": False}}
        assert strip_input_prefix("Text:\ndebug") == "Text:\ndebug"

    def test_custom_prefix(self):
        state.CONFIG = {"input_cleanup": {"strip_prefixes": ["Вопрос:"]}}
        assert strip_input_prefix("Вопрос:\nчто это") == "что это"
        assert strip_input_prefix("Text:\ndebug") == "Text:\ndebug"

    def test_non_string_passes_through(self):
        assert strip_input_prefix(None) is None  # type: ignore[arg-type]


class TestCleanLastUserMessage:
    @pytest.fixture(autouse=True)
    def _default_config(self):
        state.CONFIG = {}

    def test_cleans_last_user_string(self):
        messages = [{"role": "user", "content": "Text:\nhi"}]
        assert clean_last_user_message(messages) == [{"role": "user", "content": "hi"}]

    def test_does_not_mutate_input(self):
        messages = [{"role": "user", "content": "Text:\nhi"}]
        clean_last_user_message(messages)
        assert messages == [{"role": "user", "content": "Text:\nhi"}]

    def test_keeps_other_message_keys(self):
        messages = [{"role": "user", "content": "Text:\nhi", "name": "raycast"}]
        assert clean_last_user_message(messages)[-1]["name"] == "raycast"

    def test_earlier_messages_untouched(self):
        messages = [
            {"role": "user", "content": "Text:\nfirst"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "Text:\nsecond"},
        ]
        cleaned = clean_last_user_message(messages)
        assert cleaned[0]["content"] == "Text:\nfirst"
        assert cleaned[-1]["content"] == "second"

    def test_last_not_user_is_untouched(self):
        messages = [{"role": "user", "content": "Text:\nhi"}, {"role": "assistant", "content": "Text:\nno"}]
        assert clean_last_user_message(messages) == messages

    def test_content_parts_first_text_part(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                    {"type": "text", "text": "Text:\nwhat is this"},
                    {"type": "text", "text": "Text:\nkeep me"},
                ],
            }
        ]
        parts = clean_last_user_message(messages)[-1]["content"]
        assert parts[0]["type"] == "image_url"
        assert parts[1]["text"] == "what is this"
        assert parts[2]["text"] == "Text:\nkeep me"

    def test_content_parts_without_match(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        assert clean_last_user_message(messages) == messages

    def test_unsupported_content_type(self):
        messages = [{"role": "user", "content": None}]
        assert clean_last_user_message(messages) == messages

    def test_empty_list(self):
        assert clean_last_user_message([]) == []

    def test_non_dict_last_message(self):
        assert clean_last_user_message(["oops"]) == ["oops"]
