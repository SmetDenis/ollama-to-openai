"""Tests for the `openai_api` config section validation."""

from unittest.mock import patch

import pytest
import yaml

from ollama_adapter import state
from ollama_adapter.config import load_config


def _load(tmp_path, minimal_config, section):
    path = tmp_path / "c.yml"
    path.write_text(yaml.dump({**minimal_config, "openai_api": section}), encoding="utf-8")
    return load_config(str(path))


def test_valid_section_strips_keys(tmp_path, minimal_config):
    config = _load(tmp_path, minimal_config, {"enabled": True, "api_keys": [" sk-1 ", "sk-2"]})
    assert config["openai_api"]["api_keys"] == ["sk-1", "sk-2"]


def test_section_absent_is_fine_but_warns(tmp_path, minimal_config):
    path = tmp_path / "c.yml"
    path.write_text(yaml.dump(minimal_config), encoding="utf-8")
    with patch.object(state.logger, "warning") as warn:
        assert "openai_api" not in load_config(str(path))
    assert any("without api_keys" in str(c.args[0]) for c in warn.call_args_list)


@pytest.mark.parametrize(
    ("section", "exc"),
    [
        ("yes", TypeError),
        ({"enabled": "true"}, ValueError),
        ({"api_keys": "sk-1"}, TypeError),
        ({"api_keys": ["ok", ""]}, ValueError),
        ({"api_keys": [123]}, ValueError),
    ],
)
def test_invalid(tmp_path, minimal_config, section, exc):
    with pytest.raises(exc, match="openai_api"):
        _load(tmp_path, minimal_config, section)


def test_warns_on_unknown_keys_and_missing_keys(tmp_path, minimal_config):
    with patch.object(state.logger, "warning") as warn:
        _load(tmp_path, minimal_config, {"enabled": True, "apikeys": ["x"]})
    messages = " ".join(str(c.args[0]) for c in warn.call_args_list)
    assert "unrecognized keys" in messages
    assert "without api_keys" in messages


def test_no_warning_when_disabled_or_keyed(tmp_path, minimal_config):
    with patch.object(state.logger, "warning") as warn:
        _load(tmp_path, minimal_config, {"enabled": False})
        _load(tmp_path, minimal_config, {"api_keys": ["sk"]})
    warn.assert_not_called()
