"""Tests for ollama_adapter.tracing module."""

import json

from flask import g

from ollama_adapter import state
from ollama_adapter.tracing import (
    build_trace_body_metadata,
    build_trace_headers,
    capture_litellm_headers,
    log_litellm_headers,
    tracing_enabled,
    tracing_log_headers_enabled,
)


# ---------------------------------------------------------------------------
# tracing_enabled
# ---------------------------------------------------------------------------


class TestTracingEnabled:
    def test_enabled(self):
        state.CONFIG = {"tracing": {"enabled": True}}
        assert tracing_enabled() is True

    def test_disabled(self):
        state.CONFIG = {"tracing": {"enabled": False}}
        assert tracing_enabled() is False

    def test_missing_tracing(self):
        state.CONFIG = {}
        assert tracing_enabled() is False

    def test_missing_enabled_key(self):
        state.CONFIG = {"tracing": {}}
        assert tracing_enabled() is False


# ---------------------------------------------------------------------------
# tracing_log_headers_enabled
# ---------------------------------------------------------------------------


class TestTracingLogHeadersEnabled:
    def test_both_true(self):
        state.CONFIG = {"tracing": {"enabled": True, "log_headers": True}}
        assert tracing_log_headers_enabled() is True

    def test_enabled_false(self):
        state.CONFIG = {"tracing": {"enabled": False, "log_headers": True}}
        assert tracing_log_headers_enabled() is False

    def test_log_headers_false(self):
        state.CONFIG = {"tracing": {"enabled": True, "log_headers": False}}
        assert tracing_log_headers_enabled() is False

    def test_missing(self):
        state.CONFIG = {}
        assert tracing_log_headers_enabled() is False


# ---------------------------------------------------------------------------
# build_trace_headers
# ---------------------------------------------------------------------------


class TestBuildTraceHeaders:
    def test_tracing_off_returns_extra(self):
        state.CONFIG = {"tracing": {"enabled": False}}
        extra = {"X-Custom": "value"}
        assert build_trace_headers(extra) == extra

    def test_send_headers_off(self):
        state.CONFIG = {"tracing": {"enabled": True, "send_trace_headers": False}}
        extra = {"X-Custom": "value"}
        assert build_trace_headers(extra) == extra

    def test_adds_trace_and_call_id(self, app):
        state.CONFIG = {"tracing": {"enabled": True, "send_trace_headers": True}}
        with app.test_request_context():
            g.trace_id = "trace_123"
            g.request_id = "req_456"
            result = build_trace_headers(None)
        assert result["x-litellm-trace-id"] == "trace_123"
        assert result["x-litellm-call-id"] == "req_456"

    def test_adds_tags(self, app):
        state.CONFIG = {"tracing": {"enabled": True, "send_trace_headers": True, "tags": "prod,v2"}}
        with app.test_request_context():
            g.trace_id = "t"
            g.request_id = "r"
            result = build_trace_headers(None)
        assert result["x-litellm-tags"] == "prod,v2"

    def test_adds_display_name_metadata(self, app):
        state.CONFIG = {"tracing": {"enabled": True, "send_trace_headers": True}}
        with app.test_request_context():
            g.trace_id = "t"
            g.request_id = "r"
            result = build_trace_headers(None, display_name="GPT-4o")
        meta = json.loads(result["x-litellm-spend-logs-metadata"])
        assert meta["adapter_model"] == "GPT-4o"

    def test_extra_headers_override_trace(self, app):
        state.CONFIG = {"tracing": {"enabled": True, "send_trace_headers": True}}
        extra = {"x-litellm-trace-id": "custom-trace"}
        with app.test_request_context():
            g.trace_id = "auto-trace"
            g.request_id = "r"
            result = build_trace_headers(extra)
        assert result["x-litellm-trace-id"] == "custom-trace"

    def test_no_trace_data(self, app):
        state.CONFIG = {"tracing": {"enabled": True, "send_trace_headers": True}}
        with app.test_request_context():
            g.trace_id = None
            g.request_id = None
            result = build_trace_headers(None)
        assert result is None


# ---------------------------------------------------------------------------
# build_trace_body_metadata
# ---------------------------------------------------------------------------


class TestBuildTraceBodyMetadata:
    def test_returns_metadata(self):
        state.CONFIG = {"tracing": {"enabled": True, "send_trace_headers": True}}
        result = build_trace_body_metadata("GPT-4o")
        assert result == {"trace_name": "GPT-4o", "adapter_model": "GPT-4o"}

    def test_disabled(self):
        state.CONFIG = {"tracing": {"enabled": False}}
        assert build_trace_body_metadata("GPT-4o") is None

    def test_no_display_name(self):
        state.CONFIG = {"tracing": {"enabled": True, "send_trace_headers": True}}
        assert build_trace_body_metadata(None) is None


# ---------------------------------------------------------------------------
# capture_litellm_headers
# ---------------------------------------------------------------------------


class TestCaptureLitellmHeaders:
    def test_captures_known(self, app):
        headers = {
            "x-litellm-call-id": "call-1",
            "x-litellm-model-id": "model-1",
            "x-litellm-response-cost": "0.01",
        }
        with app.test_request_context():
            g.litellm_response_headers = {}
            capture_litellm_headers(headers)
            captured = g.litellm_response_headers
        assert captured["x-litellm-call-id"] == "call-1"
        assert captured["x-litellm-model-id"] == "model-1"
        assert captured["x-litellm-response-cost"] == "0.01"

    def test_ignores_unknown(self, app):
        headers = {"x-custom": "value", "x-litellm-call-id": "call-1"}
        with app.test_request_context():
            g.litellm_response_headers = {}
            capture_litellm_headers(headers)
            captured = g.litellm_response_headers
        assert "x-custom" not in captured
        assert "x-litellm-call-id" in captured

    def test_handles_missing(self, app):
        headers = {}
        with app.test_request_context():
            g.litellm_response_headers = {}
            capture_litellm_headers(headers)
            captured = g.litellm_response_headers
        assert captured == {}


# ---------------------------------------------------------------------------
# log_litellm_headers
# ---------------------------------------------------------------------------


class TestLogLitellmHeaders:
    def test_no_headers_no_crash(self, app):
        with app.test_request_context():
            g.litellm_response_headers = {}
            log_litellm_headers()

    def test_outside_context_no_crash(self):
        log_litellm_headers()
