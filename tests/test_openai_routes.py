"""Integration tests for the OpenAI-compatible /v1 blueprint."""

from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest
from flask import Flask

from ollama_adapter import state
from ollama_adapter.app import create_app
from ollama_adapter.openai_routes import bp as openai_bp
from ollama_adapter.routes import bp as ollama_bp

from .conftest import collect_sse, make_sdk_chunk, make_sdk_completion, make_status_error

USAGE = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
USER = [{"role": "user", "content": "hi"}]


@pytest.fixture
def v1_client(minimal_config, mock_openai_client):
    state.CONFIG = dict(minimal_config)
    state.client = mock_openai_client
    state.CACHED_MODELS = []
    flask_app = Flask(__name__)
    flask_app.register_blueprint(ollama_bp)
    flask_app.register_blueprint(openai_bp)
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


def completion(content="Hello", **message_extra):
    return make_sdk_completion(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "upstream-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content, **message_extra},
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": USAGE,
        }
    )


def chunk(delta=None, *, finish=None, usage=None, choices=True):
    return make_sdk_chunk(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "upstream-model",
            "choices": [{"index": 0, "delta": delta or {}, "logprobs": None, "finish_reason": finish}]
            if choices
            else [],
            "usage": usage,
        }
    )


class StreamStub:
    """Upstream stream double that records close() and can fail mid-stream."""

    def __init__(self, items, fail_after=None):
        self._items = items
        self._fail_after = fail_after
        self.closed = False

    def __iter__(self):
        for i, item in enumerate(self._items):
            if self._fail_after is not None and i == self._fail_after:
                msg = "boom mid-stream"
                raise RuntimeError(msg)
            yield item

    def close(self):
        self.closed = True


def upstream_kwargs(mock_openai_client):
    return mock_openai_client.chat.completions.create.call_args.kwargs


# ---------------------------------------------------------------------------
# Auth / enable switch
# ---------------------------------------------------------------------------


class TestGuard:
    def test_no_keys_means_open(self, v1_client):
        assert v1_client.get("/v1/models").status_code == 200

    @pytest.mark.parametrize("header", [None, "Bearer wrong", "Basic sk-good", "Bearer "])
    def test_rejects_missing_or_wrong_key(self, v1_client, header):
        state.CONFIG["openai_api"] = {"api_keys": ["sk-good", "sk-other"]}
        headers = {"Authorization": header} if header else {}
        resp = v1_client.get("/v1/models", headers=headers)
        assert resp.status_code == 401
        assert resp.headers["WWW-Authenticate"] == "Bearer"
        assert resp.get_json()["error"]["code"] == "invalid_api_key"

    @pytest.mark.parametrize("header", ["Bearer sk-other", "bearer sk-good"])
    def test_accepts_valid_key(self, v1_client, header):
        state.CONFIG["openai_api"] = {"api_keys": ["sk-good", "sk-other"]}
        assert v1_client.get("/v1/models", headers={"Authorization": header}).status_code == 200

    def test_ollama_endpoints_not_gated(self, v1_client):
        state.CONFIG["openai_api"] = {"api_keys": ["sk-good"]}
        assert v1_client.get("/api/version").status_code == 200

    @pytest.mark.parametrize(
        ("method", "path"), [("get", "/v1/chat/completions"), ("post", "/v1/models"), ("get", "/v1/nope")]
    )
    def test_disabled_is_404_for_any_method_and_path(self, v1_client, method, path):
        state.CONFIG["openai_api"] = {"enabled": False}
        resp = getattr(v1_client, method)(path)
        assert resp.status_code == 404
        assert resp.get_json()["error"]["code"] == "unknown_url"

    def test_unknown_v1_path_requires_key(self, v1_client):
        state.CONFIG["openai_api"] = {"api_keys": ["sk-good"]}
        assert v1_client.get("/v1/nope").status_code == 401

    def test_guard_reads_openai_api_section_once(self, v1_client):
        class CountingConfig(dict):
            reads = 0

            def get(self, key, default=None):
                if key == "openai_api":
                    CountingConfig.reads += 1
                return super().get(key, default)

        state.CONFIG = CountingConfig({**state.CONFIG, "openai_api": {"api_keys": ["sk-good"]}})
        v1_client.get("/v1/nope")
        assert CountingConfig.reads == 1

    def test_disabled_returns_404_and_reacts_to_config_change(self, v1_client):
        state.CONFIG["openai_api"] = {"enabled": False}
        resp = v1_client.post("/v1/chat/completions", json={"model": "m", "messages": USER})
        assert resp.status_code == 404
        assert resp.get_json()["error"]["code"] == "unknown_url"
        state.CONFIG["openai_api"] = {"enabled": True}
        assert v1_client.get("/v1/models").status_code == 200

    def test_root_lists_v1_only_when_enabled(self, v1_client):
        assert "/v1/chat/completions" in v1_client.get("/").get_json()["endpoints"]
        state.CONFIG["openai_api"] = {"enabled": False}
        assert "/v1/chat/completions" not in v1_client.get("/").get_json()["endpoints"]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TestModels:
    def test_list_shape(self, v1_client):
        data = v1_client.get("/v1/models").get_json()
        assert data == {
            "object": "list",
            "data": [{"id": "openai/gpt-4o", "object": "model", "created": 1700000000, "owned_by": "ollama-openai"}],
        }

    def test_custom_names(self, v1_client):
        state.CONFIG["models"] = [{"name": "openai/gpt-4o", "custom_name": "GPT"}]
        assert [m["id"] for m in v1_client.get("/v1/models").get_json()["data"]] == ["GPT"]
        assert v1_client.get("/v1/models/openai/gpt-4o").get_json()["id"] == "GPT"

    def test_retrieve_with_slash(self, v1_client):
        assert v1_client.get("/v1/models/openai/gpt-4o").get_json()["id"] == "openai/gpt-4o"

    def test_retrieve_missing(self, v1_client):
        resp = v1_client.get("/v1/models/nope")
        assert resp.status_code == 404
        assert resp.get_json()["error"]["code"] == "model_not_found"

    def test_upstream_status_error_passthrough(self, v1_client, mock_openai_client):
        mock_openai_client.models.list.side_effect = make_status_error(
            openai.RateLimitError, 429, {"message": "slow down"}, headers={"retry-after": "5"}
        )
        resp = v1_client.get("/v1/models")
        assert resp.status_code == 429
        assert resp.headers["retry-after"] == "5"
        assert resp.get_json()["error"]["message"] == "slow down"

    def test_upstream_connection_error_502(self, v1_client, mock_openai_client):
        mock_openai_client.models.list.side_effect = openai.APIConnectionError(
            request=httpx.Request("GET", "http://upstream/v1/models")
        )
        resp = v1_client.get("/v1/models")
        assert resp.status_code == 502
        assert resp.get_json()["error"]["type"] == "api_connection_error"

    def test_empty_filtered_list_is_ok(self, v1_client):
        state.CONFIG["models"] = [{"name": "not-upstream"}]
        resp = v1_client.get("/v1/models")
        assert resp.status_code == 200
        assert resp.get_json() == {"object": "list", "data": []}

    def test_large_list_not_logged_verbatim(self, v1_client, mock_openai_client):
        models = []
        for i in range(50):
            m = MagicMock()
            m.id = f"provider/model-{i}"
            m.created = 1700000000
            models.append(m)
        mock_openai_client.models.list.return_value = MagicMock(data=models)
        with patch.object(state.logger, "info") as info:
            v1_client.get("/v1/models")
        logged = " ".join(str(c.args) for c in info.call_args_list)
        assert "<50 items>" in logged
        assert "model-49" not in logged


# ---------------------------------------------------------------------------
# Chat completions — non-streaming
# ---------------------------------------------------------------------------


class TestChatNonStreaming:
    def test_passthrough_and_model_rewrite(self, v1_client, mock_openai_client):
        mock_openai_client.chat.completions.create.return_value = completion("Hi", reasoning_content="why")
        resp = v1_client.post("/v1/chat/completions", json={"model": "client-name", "messages": USER})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["model"] == "client-name"
        assert data["choices"][0]["message"] == {"role": "assistant", "content": "Hi", "reasoning_content": "why"}
        assert data["usage"] == USAGE

    def test_client_params_merged_config_wins(self, v1_client, mock_openai_client):
        state.CONFIG["models"] = [{"name": "up/m", "custom_name": "M", "params": {"temperature": 0.1}}]
        mock_openai_client.chat.completions.create.return_value = completion()
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        v1_client.post(
            "/v1/chat/completions",
            json={
                "model": "M",
                "messages": USER,
                "temperature": 1.5,
                "tools": tools,
                "api_base": "http://evil",
                "stream_options": {"include_usage": True},
            },
        )
        kwargs = upstream_kwargs(mock_openai_client)
        assert kwargs["model"] == "up/m"
        assert kwargs["stream"] is False
        assert kwargs["extra_body"] == {"temperature": 0.1, "tools": tools}

    def test_remove_thinking_tags(self, v1_client, mock_openai_client):
        state.CONFIG["models"] = [{"name": "m", "remove_thinking_tags": True}]
        mock_openai_client.chat.completions.create.return_value = completion("<think>x</think>Answer")
        data = v1_client.post("/v1/chat/completions", json={"model": "m", "messages": USER}).get_json()
        assert data["choices"][0]["message"]["content"] == "Answer"

    def test_prepend_mode_and_ip_routing(self, v1_client, mock_openai_client):
        state.CONFIG["models"] = [
            {
                "name": "m",
                "system_prompt_inline": "CFG",
                "ip_routing": [{"ip": "10.0.0.9", "name": "routed", "system_prompt_mode": "prepend"}],
            }
        ]
        mock_openai_client.chat.completions.create.return_value = completion()
        messages = [{"role": "system", "content": "agent tools"}, *USER]
        v1_client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": messages},
            headers={"X-Forwarded-For": "10.0.0.9"},
        )
        kwargs = upstream_kwargs(mock_openai_client)
        assert kwargs["model"] == "routed"
        assert kwargs["messages"][0] == {"role": "system", "content": "CFG\n\nagent tools"}

    def test_tracing_headers_sent(self, v1_client, mock_openai_client):
        state.CONFIG["tracing"] = {"enabled": True, "send_trace_headers": True, "tags": "t1"}
        mock_openai_client.chat.completions.create.return_value = completion()
        v1_client.post("/v1/chat/completions", json={"model": "m", "messages": USER})
        kwargs = upstream_kwargs(mock_openai_client)
        assert kwargs["extra_headers"]["x-litellm-tags"] == "t1"
        assert kwargs["extra_body"]["metadata"]["adapter_model"] == "m"

    @pytest.mark.parametrize(
        ("payload", "param"),
        [
            ({"messages": USER}, "model"),
            ({"model": "m"}, "messages"),
            ({"model": "m", "messages": []}, "messages"),
            ({"model": "m", "messages": USER, "stream": "yes"}, "stream"),
        ],
    )
    def test_validation_errors(self, v1_client, payload, param):
        resp = v1_client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 400
        assert resp.get_json()["error"]["param"] == param

    def test_invalid_json(self, v1_client):
        resp = v1_client.post("/v1/chat/completions", data="not json", content_type="application/json")
        assert resp.status_code == 400
        assert resp.get_json()["error"]["type"] == "invalid_request_error"

    def test_upstream_status_error_passthrough(self, v1_client, mock_openai_client):
        mock_openai_client.chat.completions.create.side_effect = make_status_error(
            openai.RateLimitError, 429, {"message": "slow down", "type": "rate_limit"}, headers={"retry-after": "3"}
        )
        resp = v1_client.post("/v1/chat/completions", json={"model": "m", "messages": USER})
        assert resp.status_code == 429
        assert resp.headers["retry-after"] == "3"
        assert resp.get_json()["error"]["message"] == "slow down"

    def test_prompt_render_error_not_forwarded(self, v1_client, mock_openai_client):
        state.CONFIG["models"] = [{"name": "m", "system_prompt_inline": "{{ missing }}"}]
        resp = v1_client.post("/v1/chat/completions", json={"model": "m", "messages": USER})
        assert resp.status_code == 500
        assert resp.get_json()["error"]["type"] == "prompt_render_error"
        assert resp.headers["x-should-retry"] == "false"
        mock_openai_client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Chat completions — streaming
# ---------------------------------------------------------------------------


class TestChatStreaming:
    def _post(self, client, **extra):
        return client.post("/v1/chat/completions", json={"model": "m", "messages": USER, "stream": True, **extra})

    def test_sse_format_and_done(self, v1_client, mock_openai_client):
        mock_openai_client.chat.completions.create.return_value = StreamStub(
            [
                chunk({"role": "assistant", "content": ""}),
                chunk({"content": "Hel"}),
                chunk({"content": "lo"}),
                chunk(finish="stop"),
                chunk(choices=False, usage=USAGE),
            ]
        )
        resp = self._post(v1_client)
        assert resp.mimetype == "text/event-stream"
        assert resp.headers["Cache-Control"] == "no-cache"
        events = collect_sse(resp)
        assert events[-1] == "[DONE]"
        assert "".join(e["choices"][0]["delta"].get("content") or "" for e in events[:-1]) == "Hello"
        assert all(e["model"] == "m" and "usage" not in e for e in events[:-1])
        assert upstream_kwargs(mock_openai_client)["extra_body"]["stream_options"] == {"include_usage": True}

    def test_usage_forwarded_when_requested(self, v1_client, mock_openai_client):
        mock_openai_client.chat.completions.create.return_value = StreamStub(
            [chunk({"content": "x"}), chunk(choices=False, usage=USAGE)]
        )
        events = collect_sse(self._post(v1_client, stream_options={"include_usage": True}))
        assert events[-2]["usage"] == USAGE

    def test_tool_call_deltas_preserved(self, v1_client, mock_openai_client):
        tool_calls = [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
        mock_openai_client.chat.completions.create.return_value = StreamStub(
            [chunk({"tool_calls": tool_calls}), chunk(finish="tool_calls")]
        )
        events = collect_sse(self._post(v1_client))
        assert events[0]["choices"][0]["delta"]["tool_calls"] == tool_calls
        assert events[1]["choices"][0]["finish_reason"] == "tool_calls"

    def test_thinking_removed_in_stream(self, v1_client, mock_openai_client):
        state.CONFIG["models"] = [{"name": "m", "remove_thinking_tags": True}]
        mock_openai_client.chat.completions.create.return_value = StreamStub(
            [chunk({"content": "<think>"}), chunk({"content": "hidden</think>Shown"}), chunk(finish="stop")]
        )
        events = collect_sse(self._post(v1_client))
        assert "".join(e["choices"][0]["delta"].get("content") or "" for e in events[:-1]) == "Shown"

    def test_early_upstream_error_gets_http_status(self, v1_client, mock_openai_client):
        mock_openai_client.chat.completions.create.side_effect = make_status_error(
            openai.AuthenticationError, 401, {"message": "bad upstream key"}
        )
        resp = self._post(v1_client)
        assert resp.status_code == 401
        assert resp.get_json()["error"]["message"] == "bad upstream key"

    def test_error_in_first_chunk_gets_http_status_and_closes(self, v1_client, mock_openai_client):
        stub = StreamStub([chunk({"content": "x"})], fail_after=0)
        mock_openai_client.chat.completions.create.return_value = stub
        resp = self._post(v1_client)
        assert resp.status_code == 500
        assert stub.closed

    def test_mid_stream_error_event_without_done(self, v1_client, mock_openai_client):
        stub = StreamStub([chunk({"content": "a"}), chunk({"content": "b"})], fail_after=1)
        mock_openai_client.chat.completions.create.return_value = stub
        resp = self._post(v1_client)
        assert resp.status_code == 200
        events = collect_sse(resp)
        assert events[0]["choices"][0]["delta"]["content"] == "a"
        assert events[-1]["error"]["type"] == "server_error"
        assert "[DONE]" not in events
        assert stub.closed

    def test_stream_closed_when_client_disconnects(self, v1_client, mock_openai_client):
        stub = StreamStub([chunk({"content": "a"}), chunk({"content": "b"})])
        mock_openai_client.chat.completions.create.return_value = stub
        resp = v1_client.post(
            "/v1/chat/completions", json={"model": "m", "messages": USER, "stream": True}, buffered=False
        )
        next(resp.response)
        resp.close()
        assert stub.closed

    def test_tracing_streaming_response_used_and_exited(self, v1_client, mock_openai_client):
        state.CONFIG["tracing"] = {"enabled": True, "log_headers": True}
        cm = MagicMock()
        raw = cm.__enter__.return_value
        raw.headers = {"x-litellm-model-id": "abc"}
        raw.parse.return_value = StreamStub([chunk({"content": "ok"})])
        mock_openai_client.chat.completions.with_streaming_response.create.return_value = cm
        events = collect_sse(self._post(v1_client))
        assert events[0]["choices"][0]["delta"]["content"] == "ok"
        cm.__exit__.assert_called_once()


# ---------------------------------------------------------------------------
# Debug keyword
# ---------------------------------------------------------------------------


class TestDebug:
    def test_string_content(self, v1_client, mock_openai_client):
        state.CONFIG["models"] = [{"name": "m", "system_prompt_inline": "CFG"}]
        resp = v1_client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "debug"}], "top_p": 0.5},
        )
        content = resp.get_json()["choices"][0]["message"]["content"]
        assert "═══ model config ═══" in content
        assert '"top_p": 0.5' in content
        assert "CFG" in content
        mock_openai_client.chat.completions.create.assert_not_called()

    def test_content_parts_streaming(self, v1_client, mock_openai_client):
        messages = [{"role": "user", "content": [{"type": "text", "text": "<user_input>debug</user_input>"}]}]
        resp = v1_client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": messages, "stream": True, "stream_options": {"include_usage": True}},
        )
        events = collect_sse(resp)
        assert "═══ message[0] role=user ═══" in events[0]["choices"][0]["delta"]["content"]
        assert events[-2]["usage"]["total_tokens"] == 0
        assert events[-1] == "[DONE]"
        mock_openai_client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Legacy completions
# ---------------------------------------------------------------------------


class TestCompletions:
    def test_non_streaming(self, v1_client, mock_openai_client):
        state.CONFIG["models"] = [{"name": "m", "system_prompt_inline": "CFG"}]
        mock_openai_client.chat.completions.create.return_value = completion("Done")
        resp = v1_client.post("/v1/completions", json={"model": "m", "prompt": ["Say"], "max_tokens": 5, "echo": False})
        data = resp.get_json()
        assert data["object"] == "text_completion"
        assert data["choices"] == [{"text": "Done", "index": 0, "logprobs": None, "finish_reason": "stop"}]
        kwargs = upstream_kwargs(mock_openai_client)
        assert kwargs["messages"] == [{"role": "system", "content": "CFG"}, {"role": "user", "content": "Say"}]
        assert kwargs["extra_body"] == {"max_tokens": 5}

    def test_streaming(self, v1_client, mock_openai_client):
        mock_openai_client.chat.completions.create.return_value = StreamStub(
            [chunk({"role": "assistant", "content": ""}), chunk({"content": "ab"}), chunk(finish="stop")]
        )
        events = collect_sse(v1_client.post("/v1/completions", json={"model": "m", "prompt": "x", "stream": True}))
        assert [e["choices"][0]["text"] for e in events[:-1]] == ["ab", ""]
        assert events[0]["object"] == "text_completion"
        assert events[-1] == "[DONE]"

    @pytest.mark.parametrize(
        ("extra", "param"),
        [
            ({"prompt": ["a", "b"]}, "prompt"),
            ({"prompt": [1, 2]}, "prompt"),
            ({"prompt": "  "}, "prompt"),
            ({"prompt": "x", "echo": True}, "echo"),
            ({"prompt": "x", "suffix": "tail"}, "suffix"),
            ({"prompt": "x", "logprobs": 2}, "logprobs"),
            ({"prompt": "x", "logprobs": 0}, "logprobs"),
            ({"prompt": "x", "best_of": 3}, "best_of"),
        ],
    )
    def test_rejected(self, v1_client, extra, param):
        resp = v1_client.post("/v1/completions", json={"model": "m", **extra})
        assert resp.status_code == 400
        assert resp.get_json()["error"]["param"] == param

    @pytest.mark.parametrize("extra", [{"logprobs": False}, {"logprobs": None}, {"suffix": ""}, {"best_of": 1}])
    def test_falsy_legacy_params_accepted(self, v1_client, mock_openai_client, extra):
        mock_openai_client.chat.completions.create.return_value = completion()
        resp = v1_client.post("/v1/completions", json={"model": "m", "prompt": "x", **extra})
        assert resp.status_code == 200
        assert "logprobs" not in (upstream_kwargs(mock_openai_client)["extra_body"] or {})

    def test_debug_prompt(self, v1_client, mock_openai_client):
        data = v1_client.post("/v1/completions", json={"model": "m", "prompt": "DEBUG"}).get_json()
        assert data["object"] == "text_completion"
        assert "═══ message[0] role=user ═══" in data["choices"][0]["text"]
        mock_openai_client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# App-level routing errors under /v1
# ---------------------------------------------------------------------------


class TestAppRouting:
    def _client(self, config_file, minimal_config, mock_openai_client):
        state.CONFIG = dict(minimal_config)
        state.client = mock_openai_client
        with patch("ollama_adapter.app.init_state"):
            return create_app(str(config_file)).test_client()

    @pytest.mark.parametrize("path", ["/v1/embeddings", "/v1/responses"])
    def test_unknown_v1_route(self, config_file, minimal_config, mock_openai_client, path):
        resp = self._client(config_file, minimal_config, mock_openai_client).post(path, json={})
        assert resp.status_code == 404
        assert resp.get_json()["error"]["code"] == "unknown_url"

    def test_wrong_method(self, config_file, minimal_config, mock_openai_client):
        resp = self._client(config_file, minimal_config, mock_openai_client).get("/v1/chat/completions")
        assert resp.status_code == 405
        assert resp.get_json()["error"]["code"] == "method_not_allowed"
        assert "POST" in resp.headers["Allow"]

    def test_v1_registered_in_app(self, config_file, minimal_config, mock_openai_client):
        mock_openai_client.chat.completions.create.return_value = completion()
        client = self._client(config_file, minimal_config, mock_openai_client)
        assert client.post("/v1/chat/completions", json={"model": "m", "messages": USER}).status_code == 200


# ---------------------------------------------------------------------------
# Client label prefix ("Text:") cleanup
# ---------------------------------------------------------------------------


class TestInputPrefixCleanup:
    def test_chat_debug_behind_prefix(self, v1_client, mock_openai_client):
        state.CONFIG["models"] = [{"name": "m", "system_prompt_inline": "CFG"}]
        resp = v1_client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "Text:\ndebug"}]},
        )
        content = resp.get_json()["choices"][0]["message"]["content"]
        assert "═══ model config ═══" in content
        mock_openai_client.chat.completions.create.assert_not_called()

    def test_chat_prefix_removed_upstream(self, v1_client, mock_openai_client):
        mock_openai_client.chat.completions.create.return_value = completion()
        v1_client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "Text:\nhello"}]},
        )
        sent = mock_openai_client.chat.completions.create.call_args.kwargs["messages"]
        assert sent[-1]["content"] == "hello"

    def test_chat_content_parts_prefix_removed(self, v1_client, mock_openai_client):
        mock_openai_client.chat.completions.create.return_value = completion()
        messages = [{"role": "user", "content": [{"type": "text", "text": "Text:\nhello"}]}]
        v1_client.post("/v1/chat/completions", json={"model": "m", "messages": messages})
        sent = mock_openai_client.chat.completions.create.call_args.kwargs["messages"]
        assert sent[-1]["content"][0]["text"] == "hello"

    def test_legacy_completions_prefix_removed(self, v1_client, mock_openai_client):
        mock_openai_client.chat.completions.create.return_value = completion()
        v1_client.post("/v1/completions", json={"model": "m", "prompt": "Text:\nhello"})
        sent = mock_openai_client.chat.completions.create.call_args.kwargs["messages"]
        assert sent[-1]["content"] == "hello"

    def test_legacy_completions_debug_behind_prefix(self, v1_client, mock_openai_client):
        resp = v1_client.post("/v1/completions", json={"model": "m", "prompt": "Text:\ndebug"})
        assert "═══ model config ═══" in resp.get_json()["choices"][0]["text"]
        mock_openai_client.chat.completions.create.assert_not_called()
