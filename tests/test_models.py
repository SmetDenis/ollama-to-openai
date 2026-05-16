"""Tests for ollama_adapter.models module."""

from typing import Any

import pytest

from ollama_adapter import state
from ollama_adapter.models import (
    _read_prompt_file,
    apply_ip_routing,
    apply_prompt_caching,
    apply_system_prompt,
    create_final_response,
    get_and_cache_models,
    get_display_name,
    get_model_config,
    resolve_model_name,
)

# ---------------------------------------------------------------------------
# resolve_model_name
# ---------------------------------------------------------------------------


class TestResolveModelName:
    def test_custom_to_original(self, full_config):
        state.CONFIG = full_config
        assert resolve_model_name("GPT-4o") == "openai/gpt-4o"

    def test_original_passthrough(self, full_config):
        state.CONFIG = full_config
        assert resolve_model_name("openai/gpt-4o") == "openai/gpt-4o"

    def test_unknown_passthrough(self, full_config):
        state.CONFIG = full_config
        assert resolve_model_name("unknown-model") == "unknown-model"

    def test_empty_models_config(self, minimal_config):
        state.CONFIG = minimal_config
        assert resolve_model_name("anything") == "anything"


# ---------------------------------------------------------------------------
# get_display_name
# ---------------------------------------------------------------------------


class TestGetDisplayName:
    def test_with_custom(self, full_config):
        state.CONFIG = full_config
        assert get_display_name("openai/gpt-4o") == "GPT-4o"

    def test_without_custom(self, full_config):
        state.CONFIG = full_config
        assert get_display_name("openai/gpt-3.5-turbo") == "openai/gpt-3.5-turbo"

    def test_unknown_model(self, full_config):
        state.CONFIG = full_config
        assert get_display_name("unknown") == "unknown"


# ---------------------------------------------------------------------------
# _read_prompt_file
# ---------------------------------------------------------------------------


class TestReadPromptFile:
    def test_md_extension(self, tmp_path, monkeypatch):
        prompt_file = tmp_path / "prompt.md"
        prompt_file.write_text("Be concise.")
        monkeypatch.chdir(tmp_path)
        assert _read_prompt_file("prompt.md") == "Be concise."

    def test_txt_extension(self, tmp_path, monkeypatch):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("From txt file.")
        monkeypatch.chdir(tmp_path)
        assert _read_prompt_file("prompt.txt") == "From txt file."

    def test_no_extension(self, tmp_path, monkeypatch):
        prompt_file = tmp_path / "PROMPT"
        prompt_file.write_text("No extension here.")
        monkeypatch.chdir(tmp_path)
        assert _read_prompt_file("PROMPT") == "No extension here."

    def test_absolute_path(self, tmp_path):
        prompt_file = tmp_path / "test.yml"
        prompt_file.write_text("Hello prompt")
        assert _read_prompt_file(str(prompt_file)) == "Hello prompt"

    def test_empty_file(self, tmp_path, monkeypatch):
        prompt_file = tmp_path / "empty.txt"
        prompt_file.write_text("   ")
        monkeypatch.chdir(tmp_path)
        assert _read_prompt_file("empty.txt") is None

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            _read_prompt_file("nonexistent.txt")


# ---------------------------------------------------------------------------
# apply_system_prompt
# ---------------------------------------------------------------------------


class TestApplySystemPrompt:
    def test_inline_replaces_existing(self):
        messages = [{"role": "system", "content": "old"}, {"role": "user", "content": "hi"}]
        result = apply_system_prompt(messages, {"system_prompt_inline": "new"}, "m")
        assert result[0]["content"] == "new"

    def test_inline_prepends_when_no_system(self):
        messages = [{"role": "user", "content": "hi"}]
        result = apply_system_prompt(messages, {"system_prompt_inline": "injected"}, "m")
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "injected"
        assert result[1]["role"] == "user"

    def test_file_any_extension(self, tmp_path, monkeypatch):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("From file.")
        monkeypatch.chdir(tmp_path)
        messages = [{"role": "user", "content": "hi"}]
        result = apply_system_prompt(messages, {"system_prompt_file": "prompt.txt"}, "m")
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "From file."

    def test_both_fields_prefer_file(self, tmp_path, monkeypatch, caplog):
        prompt_file = tmp_path / "p.txt"
        prompt_file.write_text("file wins")
        monkeypatch.chdir(tmp_path)
        messages = [{"role": "user", "content": "hi"}]
        with caplog.at_level("WARNING", logger="ollama_adapter"):
            result = apply_system_prompt(
                messages,
                {"system_prompt_inline": "inline loses", "system_prompt_file": "p.txt"},
                "m",
            )
        assert result[0]["content"] == "file wins"
        assert any("both 'system_prompt_inline' and 'system_prompt_file'" in r.message for r in caplog.records)

    def test_no_prompt_in_config(self):
        messages = [{"role": "user", "content": "hi"}]
        result = apply_system_prompt(messages, {}, "m")
        assert result == messages

    def test_empty_values_skip(self):
        messages = [{"role": "user", "content": "hi"}]
        result = apply_system_prompt(
            messages,
            {"system_prompt_inline": "   ", "system_prompt_file": ""},
            "m",
        )
        assert result == messages

    def test_file_not_found_graceful(self):
        messages = [{"role": "user", "content": "hi"}]
        result = apply_system_prompt(messages, {"system_prompt_file": "missing.txt"}, "m")
        assert result == messages

    def test_does_not_mutate_original(self):
        messages = [{"role": "user", "content": "hi"}]
        original = list(messages)
        apply_system_prompt(messages, {"system_prompt_inline": "new"}, "m")
        assert messages == original


# ---------------------------------------------------------------------------
# apply_prompt_caching
# ---------------------------------------------------------------------------


class TestApplyPromptCaching:
    def test_adds_cache_control(self):
        messages = [{"role": "system", "content": "Be helpful"}, {"role": "user", "content": "hi"}]
        result = apply_prompt_caching(messages, {"prompt_caching": True}, "m")
        sys_content = result[0]["content"]
        assert isinstance(sys_content, list)
        assert sys_content[0]["cache_control"] == {"type": "ephemeral"}
        assert sys_content[0]["text"] == "Be helpful"

    def test_disabled_no_change(self):
        messages = [{"role": "system", "content": "Be helpful"}]
        result = apply_prompt_caching(messages, {"prompt_caching": False}, "m")
        assert result[0]["content"] == "Be helpful"

    def test_no_system_message(self):
        messages = [{"role": "user", "content": "hi"}]
        result = apply_prompt_caching(messages, {"prompt_caching": True}, "m")
        assert result == messages

    def test_system_non_string_content(self):
        messages = [{"role": "system", "content": ["already", "a list"]}]
        result = apply_prompt_caching(messages, {"prompt_caching": True}, "m")
        assert result[0]["content"] == ["already", "a list"]

    def test_does_not_mutate_original(self):
        messages = [{"role": "system", "content": "x"}]
        original = [dict(m) for m in messages]
        apply_prompt_caching(messages, {"prompt_caching": True}, "m")
        assert messages == original


# ---------------------------------------------------------------------------
# get_and_cache_models
# ---------------------------------------------------------------------------


class TestGetAndCacheModels:
    def test_returns_cached(self, mock_openai_client):
        state.client = mock_openai_client
        state.CACHED_MODELS = [{"name": "cached"}]
        result = get_and_cache_models()
        assert result == [{"name": "cached"}]
        mock_openai_client.models.list.assert_not_called()

    def test_fetches_on_empty_cache(self, mock_openai_client, minimal_config):
        state.client = mock_openai_client
        state.CONFIG = minimal_config
        state.CACHED_MODELS = []
        result = get_and_cache_models()
        assert len(result) == 1
        assert result[0]["name"] == "openai/gpt-4o"

    def test_force_refresh(self, mock_openai_client, minimal_config):
        state.client = mock_openai_client
        state.CONFIG = minimal_config
        state.CACHED_MODELS = [{"name": "old"}]
        result = get_and_cache_models(force_refresh=True)
        assert result[0]["name"] == "openai/gpt-4o"
        mock_openai_client.models.list.assert_called_once()

    def test_filtered_by_models_config(self, mock_openai_client, full_config):
        state.client = mock_openai_client
        state.CONFIG = full_config
        state.CACHED_MODELS = []
        result = get_and_cache_models()
        found = [m["name"] for m in result]
        assert "GPT-4o" in found

    def test_api_error_returns_empty(self, mock_openai_client, minimal_config):
        state.client = mock_openai_client
        state.CONFIG = minimal_config
        state.CACHED_MODELS = []
        mock_openai_client.models.list.side_effect = RuntimeError("API down")
        result = get_and_cache_models()
        assert result == []

    def test_api_error_keeps_cache_on_refresh(self, mock_openai_client, minimal_config):
        state.client = mock_openai_client
        state.CONFIG = minimal_config
        state.CACHED_MODELS = [{"name": "preserved"}]
        mock_openai_client.models.list.side_effect = RuntimeError("API down")
        result = get_and_cache_models(force_refresh=True)
        assert result == [{"name": "preserved"}]


# ---------------------------------------------------------------------------
# apply_ip_routing
# ---------------------------------------------------------------------------


class TestApplyIpRouting:
    def _entry_with_routing(self) -> dict[str, Any]:
        return {
            "name": "openai/gpt-3.5-turbo",
            "params": {"temperature": 0.7, "max_tokens": 2000},
            "headers": {"X-Custom": "orig"},
            "ip_routing": [
                {"ip": "10.0.0.1", "name": "openai/gpt-4o", "params": {"temperature": 0.3}},
                {"ip": "10.0.0.2", "system_prompt_inline": "Be concise."},
            ],
        }

    def test_matching_ip(self):
        state.CONFIG = {}
        entry = self._entry_with_routing()
        result = apply_ip_routing(entry, "10.0.0.1")
        assert result["name"] == "openai/gpt-4o"

    def test_no_match(self):
        state.CONFIG = {}
        entry = self._entry_with_routing()
        result = apply_ip_routing(entry, "99.99.99.99")
        assert result["name"] == "openai/gpt-3.5-turbo"

    def test_no_ip_routing(self):
        state.CONFIG = {}
        entry = {"name": "model-a"}
        result = apply_ip_routing(entry, "10.0.0.1")
        assert result == entry

    def test_no_client_ip(self):
        state.CONFIG = {}
        entry = self._entry_with_routing()
        result = apply_ip_routing(entry, "")
        assert result == entry

    def test_params_shallow_merge(self):
        state.CONFIG = {}
        entry = self._entry_with_routing()
        result = apply_ip_routing(entry, "10.0.0.1")
        assert result["params"]["temperature"] == 0.3
        assert result["params"]["max_tokens"] == 2000

    def test_headers_shallow_merge(self):
        state.CONFIG = {}
        entry = {
            "name": "m",
            "headers": {"A": "1"},
            "ip_routing": [{"ip": "10.0.0.1", "headers": {"B": "2"}}],
        }
        result = apply_ip_routing(entry, "10.0.0.1")
        assert result["headers"] == {"A": "1", "B": "2"}

    def test_scalar_fields_replaced(self):
        state.CONFIG = {}
        entry = self._entry_with_routing()
        result = apply_ip_routing(entry, "10.0.0.2")
        assert result["system_prompt_inline"] == "Be concise."

    def test_client_alias_resolution(self, full_config):
        state.CONFIG = full_config
        entry = {
            "name": "m",
            "ip_routing": [{"ip": "office", "name": "openai/gpt-4o"}],
        }
        result = apply_ip_routing(entry, "192.168.1.100")
        assert result["name"] == "openai/gpt-4o"

    def test_does_not_mutate_original(self):
        state.CONFIG = {}
        entry = self._entry_with_routing()
        original_name = entry["name"]
        apply_ip_routing(entry, "10.0.0.1")
        assert entry["name"] == original_name

    def test_ip_routing_removed_from_result(self):
        state.CONFIG = {}
        entry = self._entry_with_routing()
        result = apply_ip_routing(entry, "10.0.0.1")
        assert "ip_routing" not in result


# ---------------------------------------------------------------------------
# get_model_config
# ---------------------------------------------------------------------------


class TestGetModelConfig:
    def test_known_by_custom_name(self, full_config):
        state.CONFIG = full_config
        openai_params, adapter_params, headers = get_model_config("GPT-4o")
        assert openai_params["model_id"] == "openai/gpt-4o"
        assert openai_params["temperature"] == 0.7

    def test_known_by_original_name(self, full_config):
        state.CONFIG = full_config
        openai_params, _, _ = get_model_config("openai/gpt-3.5-turbo")
        assert openai_params["model_id"] == "openai/gpt-3.5-turbo"

    def test_unknown_model(self, full_config):
        state.CONFIG = full_config
        openai_params, adapter_params, headers = get_model_config("unknown")
        assert openai_params == {"model_id": "unknown"}
        assert adapter_params == {}
        assert headers == {}

    def test_with_ip_routing(self, full_config):
        state.CONFIG = full_config
        openai_params, _, _ = get_model_config("openai/gpt-3.5-turbo", client_ip="192.168.1.100")
        assert openai_params["model_id"] == "openai/gpt-4o"
        assert openai_params["temperature"] == 0.3

    def test_adapter_params_extraction(self, full_config):
        state.CONFIG = full_config
        _, adapter_params, _ = get_model_config("Mini")
        assert adapter_params["remove_thinking_tags"] is True

    def test_system_prompt_in_adapter_params(self, full_config):
        state.CONFIG = full_config
        _, adapter_params, _ = get_model_config("openai/gpt-3.5-turbo")
        assert adapter_params["system_prompt_inline"] == "You are helpful."


# ---------------------------------------------------------------------------
# create_final_response
# ---------------------------------------------------------------------------


class TestCreateFinalResponse:
    def test_structure(self):
        resp = create_final_response("model-1", 10, 20, 1_000_000_000)
        assert resp["model"] == "model-1"
        assert resp["done"] is True
        assert resp["prompt_eval_count"] == 10
        assert resp["eval_count"] == 20
        assert resp["total_duration"] == 1_000_000_000
        assert "created_at" in resp

    def test_eval_duration_ratio(self):
        resp = create_final_response("m", 0, 0, 1_000_000_000)
        assert resp["eval_duration"] == 900_000_000

    def test_zero_duration(self):
        resp = create_final_response("m", 0, 0, 0)
        assert resp["eval_duration"] == 0

    def test_done_always_true(self):
        resp = create_final_response("m", 0, 0, 0)
        assert resp["done"] is True
