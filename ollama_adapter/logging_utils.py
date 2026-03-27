import time
from functools import wraps
from flask import request, jsonify, Response, g

from ollama_adapter import state


class TraceContextFilter:
    """Injects request_id|trace_id into log records when inside a Flask request."""
    def filter(self, record):
        try:
            req_id = getattr(g, 'request_id', None)
            trace_id = getattr(g, 'trace_id', None)
            if req_id and trace_id:
                record.trace_context = f"{req_id}|{trace_id}"
            elif req_id:
                record.trace_context = req_id
            else:
                record.trace_context = "-"
        except RuntimeError:
            record.trace_context = "-"
        return True


def validate_json_request():
    """Validate that request contains valid JSON."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body cannot be empty"}), 400

    return data, None


def validate_model_parameter(data):
    """Validate model parameter exists and is valid."""
    model = data.get("model") or data.get("name")
    if not model:
        return None, jsonify({"error": "Parameter 'model' or 'name' is required"}), 400

    if not isinstance(model, str) or not model.strip():
        return None, jsonify({"error": "Parameter 'model' must be a non-empty string"}), 400

    return model.strip(), None


def get_client_ip():
    """Determine the real client IP address from the request.
    Checks X-Forwarded-For (first IP), then X-Real-IP, then request.remote_addr."""
    forwarded_for = request.headers.get('X-Forwarded-For')
    if forwarded_for:
        client_ip = forwarded_for.split(',')[0].strip()
        if client_ip:
            return client_ip
    real_ip = request.headers.get('X-Real-IP')
    if real_ip:
        return real_ip.strip()
    return request.remote_addr


def log_request(endpoint, method, data=None):
    """Log incoming request details."""
    if not state.CONFIG.get('logging', {}).get('log_requests', True):
        return

    log_data = {
        "endpoint": endpoint,
        "method": method,
        "client_ip": get_client_ip(),
        "user_agent": request.headers.get('User-Agent', 'Unknown')
    }

    if data:
        if isinstance(data, dict):
            safe_data = {k: f"<{type(v).__name__}>" if k in ['messages', 'prompt'] and len(str(v)) > 100
                        else v for k, v in data.items()}
            log_data["request_data"] = safe_data
        else:
            log_data["request_data"] = f"<{type(data).__name__}>"

    state.logger.info(f"Request: {log_data}")


def log_response(endpoint, status_code, response_data=None, error=None):
    """Log response details."""
    if not state.CONFIG.get('logging', {}).get('log_requests', True):
        return

    log_data = {
        "endpoint": endpoint,
        "status_code": status_code
    }

    if error:
        log_data["error"] = str(error)
        state.logger.error(f"Response: {log_data}")
    elif response_data:
        if isinstance(response_data, dict):
            if any(key in response_data for key in ['models', 'embeddings']) and len(str(response_data)) > 500:
                log_data["response"] = {k: f"<{len(v)} items>" if isinstance(v, list) else f"<{type(v).__name__}>"
                                      for k, v in response_data.items()}
            else:
                log_data["response"] = response_data
        elif isinstance(response_data, str) and len(response_data) > 200:
            log_data["response"] = response_data[:200] + "..."
        else:
            log_data["response"] = response_data

        state.logger.info(f"Response: {log_data}")
    else:
        state.logger.info(f"Response: {log_data}")


def log_endpoint(f):
    """Decorator to log requests and responses for endpoints."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        endpoint_path = request.path

        if request.is_json:
            try:
                data = request.get_json()
                log_request(endpoint_path, request.method, data)
            except Exception:
                log_request(endpoint_path, request.method)
        else:
            log_request(endpoint_path, request.method)

        start_time = time.time()

        try:
            result = f(*args, **kwargs)
            duration = time.time() - start_time

            if isinstance(result, tuple):
                response, status_code = result
                if hasattr(response, 'get_json'):
                    try:
                        response_data = response.get_json()
                        log_response(endpoint_path, status_code, response_data)
                    except Exception:
                        log_response(endpoint_path, status_code, None)
                else:
                    log_response(endpoint_path, status_code, None)

                state.logger.info(f"Request completed in {duration:.3f}s")
                return result
            else:
                status_code = 200
                if hasattr(result, 'get_json'):
                    try:
                        response_data = result.get_json()
                        log_response(endpoint_path, status_code, response_data)
                    except Exception:
                        log_response(endpoint_path, status_code, None)
                elif isinstance(result, Response):
                    status_code = result.status_code
                    log_response(endpoint_path, status_code, None)
                else:
                    log_response(endpoint_path, status_code, None)

                state.logger.info(f"Request completed in {duration:.3f}s")
                return result

        except Exception as e:
            duration = time.time() - start_time
            state.logger.error(f"Endpoint {endpoint_path} failed after {duration:.3f}s: {str(e)}")
            log_response(endpoint_path, 500, None, str(e))
            raise

    return decorated_function
