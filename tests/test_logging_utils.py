"""Tests for ollama_adapter.logging_utils module."""

import logging

from flask import g

from ollama_adapter.logging_utils import (
    TraceContextFilter,
    get_client_ip,
    validate_json_request,
    validate_model_parameter,
)


# ---------------------------------------------------------------------------
# validate_json_request
# ---------------------------------------------------------------------------


class TestValidateJsonRequest:
    def test_valid_json(self, app):
        with app.test_request_context(
            "/test", method="POST", json={"model": "gpt-4o"}
        ):
            result = validate_json_request()
            assert isinstance(result, dict)
            assert result["model"] == "gpt-4o"

    def test_not_json_content_type(self, app):
        with app.test_request_context(
            "/test", method="POST", data="plain text", content_type="text/plain"
        ):
            result = validate_json_request()
            assert isinstance(result, tuple)
            assert result[1] == 400

    def test_empty_body(self, app):
        with app.test_request_context(
            "/test", method="POST", data="{}", content_type="application/json"
        ):
            result = validate_json_request()
            assert isinstance(result, tuple)
            assert result[1] == 400


# ---------------------------------------------------------------------------
# validate_model_parameter
# ---------------------------------------------------------------------------


class TestValidateModelParameter:
    def test_valid_model(self):
        result = validate_model_parameter({"model": "gpt-4o"})
        assert result == "gpt-4o"

    def test_valid_name(self):
        result = validate_model_parameter({"name": "gpt-4o"})
        assert result == "gpt-4o"

    def test_model_takes_precedence(self):
        result = validate_model_parameter({"model": "a", "name": "b"})
        assert result == "a"

    def test_missing_both(self, app):
        with app.test_request_context():
            result = validate_model_parameter({})
            assert isinstance(result, tuple)
            assert result[1] == 400

    def test_not_string(self, app):
        with app.test_request_context():
            result = validate_model_parameter({"model": 123})
            assert isinstance(result, tuple)
            assert result[1] == 400

    def test_empty_string(self, app):
        with app.test_request_context():
            result = validate_model_parameter({"model": ""})
            assert isinstance(result, tuple)
            assert result[1] == 400

    def test_whitespace_only(self, app):
        with app.test_request_context():
            result = validate_model_parameter({"model": "   "})
            assert isinstance(result, tuple)
            assert result[1] == 400

    def test_stripped(self):
        result = validate_model_parameter({"model": "  gpt-4o  "})
        assert result == "gpt-4o"


# ---------------------------------------------------------------------------
# get_client_ip
# ---------------------------------------------------------------------------


class TestGetClientIp:
    def test_x_forwarded_for_single(self, app):
        with app.test_request_context(headers={"X-Forwarded-For": "10.0.0.1"}):
            assert get_client_ip() == "10.0.0.1"

    def test_x_forwarded_for_multiple(self, app):
        with app.test_request_context(headers={"X-Forwarded-For": "10.0.0.1, 10.0.0.2"}):
            assert get_client_ip() == "10.0.0.1"

    def test_x_real_ip(self, app):
        with app.test_request_context(headers={"X-Real-IP": "10.0.0.5"}):
            assert get_client_ip() == "10.0.0.5"

    def test_remote_addr_fallback(self, app):
        with app.test_request_context(environ_base={"REMOTE_ADDR": "127.0.0.1"}):
            assert get_client_ip() == "127.0.0.1"

    def test_forwarded_for_priority(self, app):
        with app.test_request_context(
            headers={"X-Forwarded-For": "10.0.0.1", "X-Real-IP": "10.0.0.2"}
        ):
            assert get_client_ip() == "10.0.0.1"

    def test_empty_forwarded_for_falls_through(self, app):
        with app.test_request_context(
            headers={"X-Forwarded-For": "", "X-Real-IP": "10.0.0.5"}
        ):
            assert get_client_ip() == "10.0.0.5"


# ---------------------------------------------------------------------------
# TraceContextFilter
# ---------------------------------------------------------------------------


class TestTraceContextFilter:
    def test_both_ids(self, app):
        f = TraceContextFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        with app.test_request_context():
            g.request_id = "req_abc"
            g.trace_id = "trace_xyz"
            f.filter(record)
        assert record.trace_context == "req_abc|trace_xyz"

    def test_request_id_only(self, app):
        f = TraceContextFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        with app.test_request_context():
            g.request_id = "req_abc"
            g.trace_id = None
            f.filter(record)
        assert record.trace_context == "req_abc"

    def test_no_ids(self, app):
        f = TraceContextFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        with app.test_request_context():
            f.filter(record)
        assert record.trace_context == "-"

    def test_outside_request_context(self):
        f = TraceContextFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        f.filter(record)
        assert record.trace_context == "-"

    def test_always_returns_true(self, app):
        f = TraceContextFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        with app.test_request_context():
            assert f.filter(record) is True
