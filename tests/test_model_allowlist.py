"""Tests for the configured-models allowlist (404 for unlisted models) and the replace-mode warning."""

from unittest.mock import patch

import pytest
from flask import Flask

from ollama_adapter import state
from ollama_adapter.completion import CLIENT_PARAM_DENYLIST, merge_client_params
from ollama_adapter.models import is_model_allowed
from ollama_adapter.openai_routes import bp as openai_bp
from ollama_adapter.routes import bp as ollama_bp

from .conftest import make_mock_completion, make_mock_embedding, make_sdk_completion

MODELS = [{"name": "up/listed", "custom_name": "Listed"}]
USER = [{"role": "user", "content": "hi"}]


@pytest.fixture
def both_client(minimal_config, mock_openai_client):
    state.CONFIG = {**minimal_config, "models": [dict(m) for m in MODELS]}
    state.client = mock_openai_client
    state.CACHED_MODELS = []
    app = Flask(__name__)
    app.register_blueprint(ollama_bp)
    app.register_blueprint(openai_bp)
    app.config["TESTING"] = True
    return app.test_client()


def sdk_completion():
    return make_sdk_completion(
        {
            "id": "c",
            "object": "chat.completion",
            "created": 1,
            "model": "up/listed",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        }
    )


class TestIsModelAllowed:
    def test_empty_models_allows_everything(self):
        state.CONFIG = {**state.CONFIG, "models": []}
        assert is_model_allowed("anything")

    @pytest.mark.parametrize(("model", "allowed"), [("Listed", True), ("up/listed", True), ("up/other", False)])
    def test_listed_by_custom_or_original_name(self, model, allowed):
        state.CONFIG = {**state.CONFIG, "models": MODELS}
        assert is_model_allowed(model) is allowed


class TestOllamaEndpoints404:
    @pytest.mark.parametrize(
        ("path", "payload"),
        [
            ("/api/chat", {"messages": USER}),
            ("/api/generate", {"prompt": "hi"}),
            ("/api/embed", {"input": "hi"}),
            ("/api/show", {}),
        ],
    )
    def test_unlisted_model_404(self, both_client, mock_openai_client, path, payload):
        resp = both_client.post(path, json={"model": "up/other", **payload})
        assert resp.status_code == 404
        assert resp.get_json() == {"error": 'model "up/other" not found'}
        mock_openai_client.chat.completions.create.assert_not_called()
        mock_openai_client.embeddings.create.assert_not_called()

    def test_unlisted_debug_also_404(self, both_client):
        assert (
            both_client.post(
                "/api/chat", json={"model": "up/other", "messages": [{"role": "user", "content": "debug"}]}
            ).status_code
            == 404
        )

    def test_listed_model_still_works(self, both_client, mock_openai_client):
        mock_openai_client.chat.completions.create.return_value = make_mock_completion("ok")
        mock_openai_client.embeddings.create.return_value = make_mock_embedding()
        assert both_client.post("/api/chat", json={"model": "Listed", "messages": USER}).status_code == 200
        assert both_client.post("/api/embed", json={"model": "up/listed", "input": "x"}).status_code == 200


class TestOpenAIEndpoints404:
    @pytest.mark.parametrize(
        ("path", "payload"),
        [
            ("/v1/chat/completions", {"messages": USER}),
            ("/v1/chat/completions", {"messages": [{"role": "user", "content": "debug"}]}),
            ("/v1/completions", {"prompt": "hi"}),
        ],
    )
    def test_unlisted_model_404(self, both_client, mock_openai_client, path, payload):
        resp = both_client.post(path, json={"model": "up/other", **payload})
        assert resp.status_code == 404
        assert resp.get_json()["error"]["code"] == "model_not_found"
        assert resp.get_json()["error"]["param"] == "model"
        mock_openai_client.chat.completions.create.assert_not_called()

    def test_listed_model_works(self, both_client, mock_openai_client):
        mock_openai_client.chat.completions.create.return_value = sdk_completion()
        assert both_client.post("/v1/chat/completions", json={"model": "Listed", "messages": USER}).status_code == 200


class TestReplaceWarning:
    def _config(self, mode=None):
        entry = {"name": "up/listed", "custom_name": "Listed", "system_prompt_inline": "CFG"}
        if mode:
            entry["system_prompt_mode"] = mode
        state.CONFIG["models"] = [entry]

    def _warnings(self, client, path, payload, mock_openai_client, response):
        mock_openai_client.chat.completions.create.return_value = response
        with patch.object(state.logger, "warning") as warn:
            client.post(path, json=payload)
        return [str(c.args[0]) for c in warn.call_args_list if "system_prompt_mode=replace" in str(c.args[0])]

    def test_v1_warns_when_replace_discards_client_system(self, both_client, mock_openai_client):
        self._config()
        payload = {"model": "Listed", "messages": [{"role": "developer", "content": "tool rules"}, *USER]}
        assert self._warnings(both_client, "/v1/chat/completions", payload, mock_openai_client, sdk_completion())

    @pytest.mark.parametrize(
        ("mode", "messages"),
        [
            ("prepend", [{"role": "system", "content": "tool rules"}, *USER]),
            (None, USER),
            (None, [{"role": "system", "content": "   "}, *USER]),
        ],
    )
    def test_v1_no_warning(self, both_client, mock_openai_client, mode, messages):
        self._config(mode)
        payload = {"model": "Listed", "messages": messages}
        assert not self._warnings(both_client, "/v1/chat/completions", payload, mock_openai_client, sdk_completion())

    def test_ollama_path_does_not_warn(self, both_client, mock_openai_client):
        self._config()
        payload = {"model": "Listed", "messages": [{"role": "system", "content": "client"}, *USER]}
        assert not self._warnings(both_client, "/api/chat", payload, mock_openai_client, make_mock_completion("ok"))


@pytest.mark.parametrize("key", sorted(CLIENT_PARAM_DENYLIST))
def test_extended_denylist_dropped(key):
    merged = merge_client_params({key: {"x": "y"}, "top_p": 1}, {"model_id": "m"}, "M", streaming=False)
    assert merged.extra_body == {"top_p": 1}


def test_denylist_covers_litellm_control_keys():
    assert {"custom_llm_provider", "extra_headers", "headers", "litellm_params"} <= CLIENT_PARAM_DENYLIST
