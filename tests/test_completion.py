"""Tests for ollama_adapter.completion (shared upstream pipeline)."""

from unittest.mock import MagicMock

import pytest
from flask import g

from ollama_adapter import state
from ollama_adapter.completion import (
    ResolvedModel,
    build_debug_text,
    build_extra_body,
    build_upstream_request,
    merge_client_params,
    open_chat_completion,
    prepare_messages,
    resolve_model,
)

TRACING = {"enabled": True, "send_trace_headers": True, "log_headers": True}


def _resolved(params=None, adapter=None, headers=None, model_id="up/model"):
    return ResolvedModel(
        openai_params={**(params or {}), "model_id": model_id},
        adapter_params=adapter or {},
        headers=headers or {},
    )


class TestMergeClientParams:
    def test_config_overrides_client(self):
        merged = merge_client_params(
            {"temperature": 1.0, "tools": [{"type": "function"}]},
            {"temperature": 0.2, "model_id": "x"},
            "M",
            streaming=False,
        )
        assert merged.extra_body == {"temperature": 0.2, "tools": [{"type": "function"}]}

    def test_reserved_keys_removed_from_both_sources(self):
        merged = merge_client_params(
            {"model": "a", "messages": [], "stream": True, "top_p": 1},
            {"model": "b", "stream": False, "model_id": "x"},
            "M",
            streaming=False,
        )
        assert merged.extra_body == {"top_p": 1}

    @pytest.mark.parametrize("key", ["api_key", "api_base", "base_url", "api_version"])
    def test_client_denylist_dropped(self, key):
        merged = merge_client_params({key: "evil", "n": 2}, {"model_id": "x"}, "M", streaming=False)
        assert merged.extra_body == {"n": 2}

    def test_config_may_set_denylisted_key(self):
        merged = merge_client_params({}, {"api_version": "2024", "model_id": "x"}, "M", streaming=False)
        assert merged.extra_body == {"api_version": "2024"}

    def test_streaming_forces_usage_and_records_client_intent(self):
        merged = merge_client_params({"stream_options": {"x": 1}}, {"model_id": "x"}, "M", streaming=True)
        assert merged.extra_body["stream_options"] == {"x": 1, "include_usage": True}
        assert merged.client_wants_usage is False

        merged = merge_client_params(
            {"stream_options": {"include_usage": True}}, {"model_id": "x"}, "M", streaming=True
        )
        assert merged.client_wants_usage is True

    def test_config_include_usage_false_still_forced(self):
        merged = merge_client_params(
            {}, {"stream_options": {"include_usage": False}, "model_id": "x"}, "M", streaming=True
        )
        assert merged.extra_body["stream_options"] == {"include_usage": True}
        assert merged.client_wants_usage is False

    def test_non_streaming_drops_stream_options(self):
        merged = merge_client_params(
            {"stream_options": {"include_usage": True}}, {"model_id": "x"}, "M", streaming=False
        )
        assert "stream_options" not in merged.extra_body
        assert merged.client_wants_usage is False

    def test_metadata_merge_without_tracing(self):
        merged = merge_client_params(
            {"metadata": {"a": 1, "b": 1}}, {"metadata": {"b": 2}, "model_id": "x"}, "M", streaming=False
        )
        assert merged.extra_body["metadata"] == {"a": 1, "b": 2}

    def test_metadata_merge_with_tracing(self, app):
        state.CONFIG = {**state.CONFIG, "tracing": TRACING}
        with app.test_request_context():
            merged = merge_client_params(
                {"metadata": {"trace_name": "client", "user": "u"}}, {"model_id": "x"}, "GPT", streaming=False
            )
        assert merged.extra_body["metadata"] == {"trace_name": "GPT", "user": "u", "adapter_model": "GPT"}

    def test_non_dict_config_metadata_wins(self):
        merged = merge_client_params({"metadata": {"a": 1}}, {"metadata": "raw", "model_id": "x"}, "M", streaming=False)
        assert merged.extra_body["metadata"] == "raw"


class TestBuildExtraBody:
    def test_legacy_shape(self):
        assert build_extra_body({"temperature": 0.1, "model_id": "x"}, "M") == {"temperature": 0.1}

    def test_trace_metadata_replaces(self, app):
        state.CONFIG = {**state.CONFIG, "tracing": TRACING}
        with app.test_request_context():
            body = build_extra_body({"metadata": {"a": 1}, "model_id": "x"}, "M")
        assert body["metadata"] == {"trace_name": "M", "adapter_model": "M"}


class TestResolveAndPrepare:
    def test_resolve_model_uses_config(self):
        state.CONFIG = {**state.CONFIG, "models": [{"name": "up/m", "custom_name": "M", "params": {"top_p": 1}}]}
        resolved = resolve_model("M", None)
        assert resolved.upstream_model == "up/m"
        assert resolved.openai_params == {"top_p": 1, "model_id": "up/m"}

    def test_prepare_messages_applies_prompt_and_caching(self):
        msgs = prepare_messages(
            [{"role": "user", "content": "hi"}],
            {"system_prompt_inline": "SYS", "prompt_caching": True},
            "m",
        )
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert msgs[1] == {"role": "user", "content": "hi"}

    def test_build_upstream_request_kwargs(self):
        req = build_upstream_request(
            display_name="M",
            resolved=_resolved(headers={"X-A": "1"}),
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            extra_body={},
            stream_options={"include_usage": True},
        )
        assert req.create_kwargs() == {
            "model": "up/model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "extra_body": None,
            "extra_headers": {"X-A": "1"},
        }


def _req(*, stream: bool):
    return build_upstream_request(
        display_name="M", resolved=_resolved(), messages=[], stream=stream, extra_body={"t": 1}
    )


class TestOpenChatCompletion:
    def test_plain_non_stream(self, mock_openai_client):
        state.client = mock_openai_client
        with open_chat_completion(_req(stream=False)) as resp:
            assert resp is mock_openai_client.chat.completions.create.return_value
        kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"] == {"t": 1}
        assert "stream_options" not in kwargs

    def test_plain_stream_is_closed(self, mock_openai_client):
        state.client = mock_openai_client
        stream = MagicMock()
        mock_openai_client.chat.completions.create.return_value = stream
        with pytest.raises(RuntimeError), open_chat_completion(_req(stream=True)):
            raise RuntimeError
        stream.close.assert_called_once()

    def test_raw_non_stream_captures_headers(self, app, mock_openai_client):
        state.client = mock_openai_client
        state.CONFIG = {**state.CONFIG, "tracing": TRACING}
        raw = mock_openai_client.chat.completions.with_raw_response.create.return_value
        raw.headers = {"x-litellm-model-id": "abc"}
        with app.test_request_context():
            with open_chat_completion(_req(stream=False)) as resp:
                assert resp is raw.parse.return_value
            assert g.litellm_response_headers == {"x-litellm-model-id": "abc"}

    def test_streaming_raw_exits_on_error(self, app, mock_openai_client):
        state.client = mock_openai_client
        state.CONFIG = {**state.CONFIG, "tracing": TRACING}
        cm = mock_openai_client.chat.completions.with_streaming_response.create.return_value
        cm.__enter__.return_value.headers = {}
        with app.test_request_context(), pytest.raises(RuntimeError), open_chat_completion(_req(stream=True)):
            raise RuntimeError
        cm.__exit__.assert_called_once()


def test_build_debug_text_includes_client_params():
    text = build_debug_text(
        model_id="M",
        display_name="M",
        messages=[{"role": "user", "content": "debug"}],
        resolved=_resolved(params={"temperature": 0.3}),
        extra_body={"temperature": 0.3, "tools": ["t"]},
    )
    assert '"tools"' in text
    assert "═══ outgoing request ═══" in text
