"""Tests for ollama_adapter.config module."""

from unittest.mock import patch

import pytest
import yaml

from ollama_adapter import state
from ollama_adapter.config import check_and_reload_config, init_state, load_config

# ---------------------------------------------------------------------------
# load_config: happy paths
# ---------------------------------------------------------------------------


class TestLoadConfigHappy:
    def test_minimal(self, config_file):
        config = load_config(str(config_file))
        assert config["server"]["host"] == "127.0.0.1"
        assert config["openai"]["api_key"] == "sk-test-key-000"

    def test_with_models(self, tmp_path, minimal_config):
        minimal_config["models"] = [{"name": "openai/gpt-4o"}]
        path = tmp_path / "c.yml"
        path.write_text(yaml.dump(minimal_config))
        config = load_config(str(path))
        assert len(config["models"]) == 1

    def test_with_clients(self, tmp_path, minimal_config):
        minimal_config["clients"] = {"office": ["10.0.0.1", "10.0.0.2"]}
        path = tmp_path / "c.yml"
        path.write_text(yaml.dump(minimal_config))
        config = load_config(str(path))
        assert config["clients"]["office"] == ["10.0.0.1", "10.0.0.2"]

    def test_with_tracing(self, tmp_path, minimal_config):
        minimal_config["tracing"] = {"enabled": True, "tags": "test"}
        path = tmp_path / "c.yml"
        path.write_text(yaml.dump(minimal_config))
        config = load_config(str(path))
        assert config["tracing"]["enabled"] is True

    def test_normalizes_custom_names(self, tmp_path, minimal_config):
        minimal_config["models"] = [{"name": "openai/gpt-4o", "custom_name": "  GPT  "}]
        path = tmp_path / "c.yml"
        path.write_text(yaml.dump(minimal_config))
        config = load_config(str(path))
        assert config["models"][0]["custom_name"] == "GPT"


# ---------------------------------------------------------------------------
# load_config: validation errors — required fields
# ---------------------------------------------------------------------------


class TestLoadConfigRequiredFields:
    @pytest.mark.parametrize(
        ("remove_key", "parent_key"),
        [
            ("api_key", "openai"),
            ("host", "server"),
            ("port", "server"),
        ],
    )
    def test_missing_required_field(self, tmp_path, minimal_config, remove_key, parent_key):
        del minimal_config[parent_key][remove_key]
        path = tmp_path / "c.yml"
        path.write_text(yaml.dump(minimal_config))
        with pytest.raises(ValueError, match="Missing required"):
            load_config(str(path))


# ---------------------------------------------------------------------------
# load_config: validation errors — models
# ---------------------------------------------------------------------------


class TestLoadConfigModels:
    def test_duplicate_custom_names(self, tmp_path, minimal_config):
        minimal_config["models"] = [
            {"name": "openai/gpt-4o", "custom_name": "GPT"},
            {"name": "openai/gpt-4o-mini", "custom_name": "GPT"},
        ]
        path = tmp_path / "c.yml"
        path.write_text(yaml.dump(minimal_config))
        with pytest.raises(ValueError, match="Duplicate custom_name"):
            load_config(str(path))


# ---------------------------------------------------------------------------
# load_config: validation errors — clients
# ---------------------------------------------------------------------------


class TestLoadConfigClients:
    def test_clients_not_dict(self, tmp_path, minimal_config):
        minimal_config["clients"] = ["10.0.0.1"]
        path = tmp_path / "c.yml"
        path.write_text(yaml.dump(minimal_config))
        with pytest.raises(TypeError, match="must be a dict"):
            load_config(str(path))

    def test_client_empty_ip_string(self, tmp_path, minimal_config):
        minimal_config["clients"] = {"office": "  "}
        path = tmp_path / "c.yml"
        path.write_text(yaml.dump(minimal_config))
        with pytest.raises(ValueError, match="cannot be empty"):
            load_config(str(path))

    def test_client_empty_ip_list(self, tmp_path, minimal_config):
        minimal_config["clients"] = {"office": []}
        path = tmp_path / "c.yml"
        path.write_text(yaml.dump(minimal_config))
        with pytest.raises(ValueError, match="cannot be empty"):
            load_config(str(path))

    def test_client_invalid_type(self, tmp_path, minimal_config):
        minimal_config["clients"] = {"office": 123}
        path = tmp_path / "c.yml"
        path.write_text(yaml.dump(minimal_config))
        with pytest.raises(TypeError, match="string or list"):
            load_config(str(path))

    def test_client_list_with_empty_entry(self, tmp_path, minimal_config):
        minimal_config["clients"] = {"office": ["10.0.0.1", ""]}
        path = tmp_path / "c.yml"
        path.write_text(yaml.dump(minimal_config))
        with pytest.raises(ValueError, match="non-empty string"):
            load_config(str(path))


# ---------------------------------------------------------------------------
# load_config: validation errors — ip_routing
# ---------------------------------------------------------------------------


class TestLoadConfigIpRouting:
    def _make_config(self, tmp_path, minimal_config, models):
        minimal_config["models"] = models
        path = tmp_path / "c.yml"
        path.write_text(yaml.dump(minimal_config))
        return str(path)

    def test_ip_routing_not_list(self, tmp_path, minimal_config):
        path = self._make_config(tmp_path, minimal_config, [{"name": "m", "ip_routing": {"ip": "1"}}])
        with pytest.raises(TypeError, match="must be a list"):
            load_config(path)

    def test_ip_routing_rule_not_dict(self, tmp_path, minimal_config):
        path = self._make_config(tmp_path, minimal_config, [{"name": "m", "ip_routing": ["10.0.0.1"]}])
        with pytest.raises(TypeError, match="must be a dict"):
            load_config(path)

    def test_ip_routing_missing_ip(self, tmp_path, minimal_config):
        path = self._make_config(tmp_path, minimal_config, [{"name": "m", "ip_routing": [{"name": "x"}]}])
        with pytest.raises(ValueError, match="non-empty 'ip' field"):
            load_config(path)

    def test_ip_routing_empty_ip(self, tmp_path, minimal_config):
        path = self._make_config(tmp_path, minimal_config, [{"name": "m", "ip_routing": [{"ip": "  "}]}])
        with pytest.raises(ValueError, match="non-empty 'ip' field"):
            load_config(path)

    def test_ip_routing_duplicate_ip(self, tmp_path, minimal_config):
        models = [{"name": "m", "ip_routing": [{"ip": "10.0.0.1"}, {"ip": "10.0.0.1"}]}]
        path = self._make_config(tmp_path, minimal_config, models)
        with pytest.raises(ValueError, match="duplicate IP"):
            load_config(path)

    def test_ip_routing_params_not_dict(self, tmp_path, minimal_config):
        models = [{"name": "m", "ip_routing": [{"ip": "10.0.0.1", "params": "bad"}]}]
        path = self._make_config(tmp_path, minimal_config, models)
        with pytest.raises(TypeError, match="must be a dict"):
            load_config(path)

    def test_ip_routing_headers_not_dict(self, tmp_path, minimal_config):
        models = [{"name": "m", "ip_routing": [{"ip": "10.0.0.1", "headers": ["bad"]}]}]
        path = self._make_config(tmp_path, minimal_config, models)
        with pytest.raises(TypeError, match="must be a dict"):
            load_config(path)


# ---------------------------------------------------------------------------
# load_config: validation errors — tracing
# ---------------------------------------------------------------------------


class TestLoadConfigTracing:
    def test_tracing_not_dict(self, tmp_path, minimal_config):
        minimal_config["tracing"] = "on"
        path = tmp_path / "c.yml"
        path.write_text(yaml.dump(minimal_config))
        with pytest.raises(TypeError, match="must be a dict"):
            load_config(str(path))

    def test_tracing_enabled_not_bool(self, tmp_path, minimal_config):
        minimal_config["tracing"] = {"enabled": "yes"}
        path = tmp_path / "c.yml"
        path.write_text(yaml.dump(minimal_config))
        with pytest.raises(ValueError, match="must be a boolean"):
            load_config(str(path))

    def test_tracing_tags_not_string(self, tmp_path, minimal_config):
        minimal_config["tracing"] = {"tags": ["a", "b"]}
        path = tmp_path / "c.yml"
        path.write_text(yaml.dump(minimal_config))
        with pytest.raises(ValueError, match="must be a string"):
            load_config(str(path))

    def test_tracing_grouping_valid_hourly(self, tmp_path, minimal_config):
        minimal_config["tracing"] = {"trace_grouping": "hourly"}
        path = tmp_path / "c.yml"
        path.write_text(yaml.dump(minimal_config))
        config = load_config(str(path))
        assert config["tracing"]["trace_grouping"] == "hourly"

    def test_tracing_grouping_valid_daily(self, tmp_path, minimal_config):
        minimal_config["tracing"] = {"trace_grouping": "daily"}
        path = tmp_path / "c.yml"
        path.write_text(yaml.dump(minimal_config))
        config = load_config(str(path))
        assert config["tracing"]["trace_grouping"] == "daily"

    def test_tracing_grouping_invalid(self, tmp_path, minimal_config):
        minimal_config["tracing"] = {"trace_grouping": "weekly"}
        path = tmp_path / "c.yml"
        path.write_text(yaml.dump(minimal_config))
        with pytest.raises(ValueError, match="trace_grouping"):
            load_config(str(path))

    def test_tracing_grouping_not_string(self, tmp_path, minimal_config):
        minimal_config["tracing"] = {"trace_grouping": 42}
        path = tmp_path / "c.yml"
        path.write_text(yaml.dump(minimal_config))
        with pytest.raises(ValueError, match="trace_grouping"):
            load_config(str(path))

    def test_tracing_timezone_not_string(self, tmp_path, minimal_config):
        minimal_config["tracing"] = {"timezone": 123}
        path = tmp_path / "c.yml"
        path.write_text(yaml.dump(minimal_config))
        with pytest.raises(ValueError, match="must be a string"):
            load_config(str(path))


# ---------------------------------------------------------------------------
# load_config: validation errors — prompts
# ---------------------------------------------------------------------------


class TestLoadConfigPrompts:
    def test_prompts_not_dict(self, tmp_path, minimal_config):
        minimal_config["prompts"] = ["bad"]
        path = tmp_path / "c.yml"
        path.write_text(yaml.dump(minimal_config))
        with pytest.raises(TypeError, match="'prompts' must be a dict"):
            load_config(str(path))

    def test_prompts_base_dir_must_be_string(self, tmp_path, minimal_config):
        minimal_config["prompts"] = {"base_dir": 42}
        path = tmp_path / "c.yml"
        path.write_text(yaml.dump(minimal_config))
        with pytest.raises(ValueError, match=r"prompts\.base_dir"):
            load_config(str(path))

    def test_prompts_vars_non_dict_warns(self, tmp_path, minimal_config, caplog):
        minimal_config["prompts"] = {"vars": ["bad"]}
        path = tmp_path / "c.yml"
        path.write_text(yaml.dump(minimal_config))
        with caplog.at_level("WARNING", logger="ollama_adapter"):
            config = load_config(str(path))
        assert config["prompts"]["vars"] == {}
        assert any("prompts.vars" in r.message for r in caplog.records)

    def test_model_prompt_vars_non_dict_warns(self, tmp_path, minimal_config, caplog):
        minimal_config["models"] = [{"name": "openai/gpt-4o", "prompt_vars": "bad"}]
        path = tmp_path / "c.yml"
        path.write_text(yaml.dump(minimal_config))
        with caplog.at_level("WARNING", logger="ollama_adapter"):
            config = load_config(str(path))
        assert config["models"][0]["prompt_vars"] == {}

    def test_ip_routing_prompt_vars_non_dict_raises(self, tmp_path, minimal_config):
        minimal_config["models"] = [
            {"name": "openai/gpt-4o", "ip_routing": [{"ip": "10.0.0.1", "prompt_vars": "bad"}]},
        ]
        path = tmp_path / "c.yml"
        path.write_text(yaml.dump(minimal_config))
        with pytest.raises(TypeError, match="prompt_vars"):
            load_config(str(path))


# ---------------------------------------------------------------------------
# load_config: file errors
# ---------------------------------------------------------------------------


class TestLoadConfigFileErrors:
    def test_file_not_found(self):
        with pytest.raises(OSError):
            load_config("/nonexistent/path/config.yml")

    def test_invalid_yaml(self, tmp_path):
        path = tmp_path / "bad.yml"
        path.write_text("{{invalid yaml::")
        with pytest.raises(yaml.YAMLError):
            load_config(str(path))


# ---------------------------------------------------------------------------
# init_state
# ---------------------------------------------------------------------------


class TestInitState:
    def test_success(self, config_file):
        with patch("ollama_adapter.config.OpenAI") as mock_cls:
            mock_cls.return_value = "fake-client"
            init_state(str(config_file))

        assert state.CONFIG["server"]["host"] == "127.0.0.1"
        assert state.client == "fake-client"
        assert state.config_file_path == str(config_file)
        assert state.last_config_mtime > 0
        assert state.jinja_env is not None

    def test_invalid_config_exits(self, tmp_path):
        path = tmp_path / "bad.yml"
        path.write_text(yaml.dump({"server": {}}))
        with pytest.raises(SystemExit) as exc_info:
            init_state(str(path))
        assert exc_info.value.code == 1

    def test_openai_error_exits(self, config_file):
        with (
            patch("ollama_adapter.config.OpenAI", side_effect=ValueError("bad key")),
            pytest.raises(SystemExit) as exc_info,
        ):
            init_state(str(config_file))
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# check_and_reload_config
# ---------------------------------------------------------------------------


class TestCheckAndReloadConfig:
    def test_no_change(self, config_file):
        with patch("ollama_adapter.config.OpenAI"):
            init_state(str(config_file))

        old_config = state.CONFIG.copy()
        check_and_reload_config()
        assert old_config == state.CONFIG

    def test_mtime_changed(self, config_file, minimal_config):
        with patch("ollama_adapter.config.OpenAI") as mock_cls:
            mock_cls.return_value = "client-v1"
            init_state(str(config_file))

        env_before = state.jinja_env

        minimal_config["server"]["port"] = 9999
        config_file.write_text(yaml.dump(minimal_config))

        with (
            patch("ollama_adapter.config.OpenAI") as mock_cls,
            patch("ollama_adapter.models.get_and_cache_models"),
        ):
            mock_cls.return_value = "client-v2"
            check_and_reload_config()

        assert state.CONFIG["server"]["port"] == 9999
        assert state.client == "client-v2"
        assert state.last_config_reload_time is not None
        assert state.jinja_env is not env_before

    def test_invalid_new_config_keeps_old(self, config_file):
        with patch("ollama_adapter.config.OpenAI"):
            init_state(str(config_file))

        old_config = state.CONFIG.copy()
        config_file.write_text(yaml.dump({"server": {}}))

        check_and_reload_config()
        assert old_config == state.CONFIG

    def test_file_disappeared(self, config_file):
        with patch("ollama_adapter.config.OpenAI"):
            init_state(str(config_file))

        state.config_file_path = "/nonexistent/path.yml"
        old_config = state.CONFIG.copy()
        check_and_reload_config()
        assert old_config == state.CONFIG

    def test_openai_client_error_keeps_old(self, config_file, minimal_config):
        with patch("ollama_adapter.config.OpenAI") as mock_cls:
            mock_cls.return_value = "client-v1"
            init_state(str(config_file))

        minimal_config["server"]["port"] = 8888
        config_file.write_text(yaml.dump(minimal_config))

        with patch("ollama_adapter.config.OpenAI", side_effect=ValueError("bad")):
            check_and_reload_config()

        assert state.client == "client-v1"
