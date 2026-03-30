"""Tests for ollama_adapter.app module."""

from unittest.mock import patch

from flask import g

from ollama_adapter import state
from ollama_adapter.app import create_app


class TestGlobalErrorHandler:
    def test_unhandled_exception_returns_json_500(self, config_file, mock_openai_client):
        state.CONFIG = {"server": {"host": "0.0.0.0", "port": 11434}, "openai": {"api_key": "k"}}
        state.client = mock_openai_client

        with patch("ollama_adapter.app.init_state"):
            app = create_app(str(config_file))

        @app.route("/test-unhandled-error")
        def _raise_error():
            msg = "unexpected"
            raise RuntimeError(msg)

        with patch.object(state.logger, "exception") as mock_log, app.test_client() as client:
            resp = client.get("/test-unhandled-error")

        assert resp.status_code == 500
        assert resp.get_json()["error"] == "Internal server error"
        mock_log.assert_called_once_with("Unhandled exception")


class TestCreateApp:
    def test_returns_flask_app(self, config_file):
        with patch("ollama_adapter.app.init_state"):
            app = create_app(str(config_file))
        assert app is not None
        assert hasattr(app, "test_client")

    def test_before_request_calls_reload(self, config_file, mock_openai_client):
        state.CONFIG = {"server": {"host": "0.0.0.0", "port": 11434}, "openai": {"api_key": "k"}}
        state.client = mock_openai_client

        with patch("ollama_adapter.app.init_state"):
            app = create_app(str(config_file))

        with patch("ollama_adapter.app.check_and_reload_config") as mock_reload:
            with app.test_client() as client:
                client.get("/api/version")
            mock_reload.assert_called()


class TestTraceContext:
    def _make_app(self, config_file, config, mock_openai_client):
        state.CONFIG = config
        state.client = mock_openai_client
        with patch("ollama_adapter.app.init_state"):
            return create_app(str(config_file))

    def test_trace_enabled(self, config_file, mock_openai_client):
        config = {
            "server": {"host": "0.0.0.0", "port": 11434},
            "openai": {"api_key": "k"},
            "tracing": {"enabled": True, "trace_id_prefix": "oa"},
        }
        app = self._make_app(config_file, config, mock_openai_client)

        with app.test_client() as client:
            resp = client.get("/api/version")
            assert resp.status_code == 200

    def test_trace_disabled(self, config_file, mock_openai_client):
        config = {
            "server": {"host": "0.0.0.0", "port": 11434},
            "openai": {"api_key": "k"},
        }
        app = self._make_app(config_file, config, mock_openai_client)

        with app.test_request_context(), app.test_client() as client:
            client.get("/api/version")

    def test_incoming_trace_passthrough(self, config_file, mock_openai_client):
        config = {
            "server": {"host": "0.0.0.0", "port": 11434},
            "openai": {"api_key": "k"},
            "tracing": {"enabled": True, "trace_id_prefix": "oa"},
        }
        app = self._make_app(config_file, config, mock_openai_client)

        captured = {}

        @app.after_request
        def capture_g(response):
            captured["trace_id"] = getattr(g, "trace_id", None)
            return response

        with app.test_client() as client:
            client.get("/api/version", headers={"x-litellm-trace-id": "external-trace-123"})

        assert captured["trace_id"] == "external-trace-123"

    def test_custom_trace_prefix(self, config_file, mock_openai_client):
        config = {
            "server": {"host": "0.0.0.0", "port": 11434},
            "openai": {"api_key": "k"},
            "tracing": {"enabled": True, "trace_id_prefix": "custom"},
        }
        app = self._make_app(config_file, config, mock_openai_client)

        captured = {}

        @app.after_request
        def capture_g(response):
            captured["trace_id"] = getattr(g, "trace_id", None)
            captured["request_id"] = getattr(g, "request_id", None)
            return response

        with app.test_client() as client:
            client.get("/api/version")

        assert captured["trace_id"].startswith("custom_")
        assert captured["request_id"].startswith("req_")
