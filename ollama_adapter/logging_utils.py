"""Request/response logging utilities and validation helpers."""

import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from flask import Response, g, jsonify, request

from ollama_adapter import state

_TRUNCATED_FIELDS = {"messages", "prompt"}
_TRUNCATE_THRESHOLD = 100
_LARGE_RESPONSE_THRESHOLD = 500
_RESPONSE_PREVIEW_LENGTH = 200


class TraceContextFilter(logging.Filter):
    """Inject request_id|trace_id into log records when inside a Flask request."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add trace_context attribute to the log record."""
        try:
            req_id = getattr(g, "request_id", None)
            trace_id = getattr(g, "trace_id", None)
            if req_id and trace_id:
                record.trace_context = f"{req_id}|{trace_id}"  # type: ignore[attr-defined]
            elif req_id:
                record.trace_context = req_id  # type: ignore[attr-defined]
            else:
                record.trace_context = "-"  # type: ignore[attr-defined]
        except RuntimeError:
            record.trace_context = "-"  # type: ignore[attr-defined]
        return True


def validate_json_request() -> dict | tuple[Response, int]:
    """Validate that request contains valid JSON.

    Return parsed data dict on success, or (error_response, status_code) on failure.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body cannot be empty"}), 400

    return data


def validate_model_parameter(data: dict) -> str | tuple[Response, int]:
    """Validate model parameter exists and is valid.

    Return cleaned model name on success, or (error_response, status_code) on failure.
    """
    model = data.get("model") or data.get("name")
    if not model:
        return jsonify({"error": "Parameter 'model' or 'name' is required"}), 400

    if not isinstance(model, str) or not model.strip():
        return jsonify({"error": "Parameter 'model' must be a non-empty string"}), 400

    return model.strip()


def get_client_ip() -> str | None:
    """Determine the real client IP from X-Forwarded-For, X-Real-IP, or remote_addr."""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()
        if client_ip:
            return client_ip
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    return request.remote_addr


def log_request(endpoint: str, method: str, data: Any = None) -> None:
    """Log incoming request details."""
    if not state.CONFIG.get("logging", {}).get("log_requests", True):
        return

    log_data: dict[str, Any] = {
        "endpoint": endpoint,
        "method": method,
        "client_ip": get_client_ip(),
        "user_agent": request.headers.get("User-Agent", "Unknown"),
    }

    if data:
        if isinstance(data, dict):
            safe_data = {
                k: f"<{type(v).__name__}>" if k in _TRUNCATED_FIELDS and len(str(v)) > _TRUNCATE_THRESHOLD else v
                for k, v in data.items()
            }
            log_data["request_data"] = safe_data
        else:
            log_data["request_data"] = f"<{type(data).__name__}>"

    state.logger.info("Request: %s", log_data)


def log_response(
    endpoint: str,
    status_code: int,
    response_data: Any = None,
    error: str | None = None,
) -> None:
    """Log response details."""
    if not state.CONFIG.get("logging", {}).get("log_requests", True):
        return

    log_data: dict[str, Any] = {"endpoint": endpoint, "status_code": status_code}

    if error:
        log_data["error"] = error
        state.logger.error("Response: %s", log_data)
    elif response_data:
        if isinstance(response_data, dict):
            if (
                any(key in response_data for key in ("models", "embeddings"))
                and len(str(response_data)) > _LARGE_RESPONSE_THRESHOLD
            ):
                log_data["response"] = {
                    k: f"<{len(v)} items>" if isinstance(v, list) else f"<{type(v).__name__}>"
                    for k, v in response_data.items()
                }
            else:
                log_data["response"] = response_data
        elif isinstance(response_data, str) and len(response_data) > _RESPONSE_PREVIEW_LENGTH:
            log_data["response"] = response_data[:_RESPONSE_PREVIEW_LENGTH] + "..."
        else:
            log_data["response"] = response_data

        state.logger.info("Response: %s", log_data)
    else:
        state.logger.info("Response: %s", log_data)


def log_endpoint(f: Callable) -> Callable:
    """Decorate endpoint to log requests and responses."""

    @wraps(f)
    def decorated_function(*args: Any, **kwargs: Any) -> Any:
        endpoint_path = request.path

        if request.is_json:
            try:
                data = request.get_json()
                log_request(endpoint_path, request.method, data)
            except (ValueError, RuntimeError):
                log_request(endpoint_path, request.method)
        else:
            log_request(endpoint_path, request.method)

        start_time = time.time()

        try:
            result = f(*args, **kwargs)
        except Exception:
            duration = time.time() - start_time
            state.logger.error("Endpoint %s failed after %.3fs", endpoint_path, duration)
            log_response(endpoint_path, 500, error="unhandled exception")
            raise

        duration = time.time() - start_time
        _log_result(endpoint_path, result)
        state.logger.info("Request completed in %.3fs", duration)
        return result

    return decorated_function


def _log_result(endpoint_path: str, result: Any) -> None:
    """Extract status code and response data from a route handler result and log it."""
    if isinstance(result, tuple):
        response, status_code = result
        if hasattr(response, "get_json"):
            try:
                log_response(endpoint_path, status_code, response.get_json())
            except (ValueError, RuntimeError):
                log_response(endpoint_path, status_code)
        else:
            log_response(endpoint_path, status_code)
        return

    status_code = 200
    if hasattr(result, "get_json"):
        try:
            log_response(endpoint_path, status_code, result.get_json())
        except (ValueError, RuntimeError):
            log_response(endpoint_path, status_code)
    elif isinstance(result, Response):
        log_response(endpoint_path, result.status_code)
    else:
        log_response(endpoint_path, status_code)
