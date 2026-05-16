"""Shared fixtures for pytest tests."""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
from flask import Flask

from ollama_adapter import state
from ollama_adapter.routes import bp


@pytest.fixture(autouse=True)
def _reset_state():
    """Save and restore all mutable globals in state.py between tests."""
    original = {
        "CONFIG": state.CONFIG.copy(),
        "client": state.client,
        "CACHED_MODELS": state.CACHED_MODELS[:],
        "config_file_path": state.config_file_path,
        "last_config_mtime": state.last_config_mtime,
        "last_config_reload_time": state.last_config_reload_time,
    }
    yield
    state.CONFIG = original["CONFIG"]
    state.client = original["client"]
    state.CACHED_MODELS = original["CACHED_MODELS"]
    state.config_file_path = original["config_file_path"]
    state.last_config_mtime = original["last_config_mtime"]
    state.last_config_reload_time = original["last_config_reload_time"]


@pytest.fixture
def minimal_config() -> dict[str, Any]:
    return {
        "server": {"host": "127.0.0.1", "port": 11434},
        "openai": {"api_key": "sk-test-key-000", "base_url": "http://localhost:4000/v1"},
    }


@pytest.fixture
def full_config(minimal_config) -> dict[str, Any]:
    return {
        **minimal_config,
        "clients": {
            "office": ["192.168.1.100", "192.168.1.101"],
            "home": "10.0.0.5",
        },
        "logging": {"log_level": "DEBUG", "log_requests": True},
        "tracing": {
            "enabled": True,
            "log_headers": True,
            "send_trace_headers": True,
            "trace_id_prefix": "test",
            "tags": "test-tag",
        },
        "models": [
            {"name": "openai/gpt-4o", "custom_name": "GPT-4o", "params": {"temperature": 0.7}},
            {"name": "openai/gpt-4o-mini", "custom_name": "Mini", "remove_thinking_tags": True},
            {
                "name": "openai/gpt-3.5-turbo",
                "system_prompt_inline": "You are helpful.",
                "ip_routing": [
                    {"ip": "office", "name": "openai/gpt-4o", "params": {"temperature": 0.3}},
                    {"ip": "10.0.0.5", "system_prompt_inline": "Be concise."},
                ],
            },
        ],
    }


@pytest.fixture
def config_file(tmp_path, minimal_config):
    path = tmp_path / "config.yml"
    path.write_text(yaml.dump(minimal_config), encoding="utf-8")
    return path


@pytest.fixture
def mock_openai_client():
    client = MagicMock()
    mock_model = MagicMock()
    mock_model.id = "openai/gpt-4o"
    mock_model.created = 1700000000
    client.models.list.return_value = MagicMock(data=[mock_model])
    client.with_options.return_value = client
    return client


@pytest.fixture
def app(minimal_config, mock_openai_client):
    state.CONFIG = minimal_config
    state.client = mock_openai_client
    state.CACHED_MODELS = []

    flask_app = Flask(__name__)
    flask_app.register_blueprint(bp)
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def app_context(app):
    with app.app_context():
        yield


@pytest.fixture
def request_context(app):
    with app.test_request_context():
        yield


def make_mock_chunk(content=None, usage=None):
    """Create a mock OpenAI streaming chunk."""
    chunk = MagicMock()
    chunk.usage = usage
    if content is not None:
        choice = MagicMock()
        choice.delta.content = content
        chunk.choices = [choice]
    else:
        chunk.choices = []
    return chunk


def make_mock_completion(content="Hello", prompt_tokens=10, completion_tokens=5):
    """Create a mock non-streaming completion response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    return response


def make_mock_embedding(embeddings=None, prompt_tokens=5):
    """Create a mock embedding response."""
    if embeddings is None:
        embeddings = [[0.1, 0.2, 0.3]]
    response = MagicMock()
    response.data = [MagicMock(embedding=e) for e in embeddings]
    response.usage.prompt_tokens = prompt_tokens
    return response


def collect_stream(response) -> list[dict[str, Any]]:
    """Parse ndjson from a Flask test response into a list of dicts."""
    lines = response.data.decode("utf-8").strip().split("\n")
    return [json.loads(line) for line in lines if line.strip()]
