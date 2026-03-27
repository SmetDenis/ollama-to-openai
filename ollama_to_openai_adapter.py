import sys
import yaml
import json
import time
import re
import os
import uuid
import logging
import threading
from functools import wraps
from datetime import datetime
from flask import Flask, request, jsonify, Response, stream_with_context, g
from openai import OpenAI

# Will configure logging after loading config
logger = logging.getLogger(__name__)


class TraceContextFilter(logging.Filter):
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

# --- Input validation helpers ---

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

# --- Client IP detection ---

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

# --- Logging helpers ---

def log_request(endpoint, method, data=None):
    """Log incoming request details."""
    if not CONFIG.get('logging', {}).get('log_requests', True):
        return

    log_data = {
        "endpoint": endpoint,
        "method": method,
        "client_ip": get_client_ip(),
        "user_agent": request.headers.get('User-Agent', 'Unknown')
    }

    if data:
        # Don't log sensitive data in full, just structure
        if isinstance(data, dict):
            safe_data = {k: f"<{type(v).__name__}>" if k in ['messages', 'prompt'] and len(str(v)) > 100
                        else v for k, v in data.items()}
            log_data["request_data"] = safe_data
        else:
            log_data["request_data"] = f"<{type(data).__name__}>"

    logger.info(f"Request: {log_data}")

def log_response(endpoint, status_code, response_data=None, error=None):
    """Log response details."""
    if not CONFIG.get('logging', {}).get('log_requests', True):
        return

    log_data = {
        "endpoint": endpoint,
        "status_code": status_code
    }

    if error:
        log_data["error"] = str(error)
        logger.error(f"Response: {log_data}")
    elif response_data:
        # Truncate large responses for readability
        if isinstance(response_data, dict):
            # For streaming responses or large data, show structure only
            if any(key in response_data for key in ['models', 'embeddings']) and len(str(response_data)) > 500:
                log_data["response"] = {k: f"<{len(v)} items>" if isinstance(v, list) else f"<{type(v).__name__}>"
                                      for k, v in response_data.items()}
            else:
                log_data["response"] = response_data
        elif isinstance(response_data, str) and len(response_data) > 200:
            log_data["response"] = response_data[:200] + "..."
        else:
            log_data["response"] = response_data

        logger.info(f"Response: {log_data}")
    else:
        logger.info(f"Response: {log_data}")


def log_endpoint(f):
    """Decorator to log requests and responses for endpoints."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get the actual endpoint path
        endpoint_path = request.path

        # Log request with data if it's a JSON request
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
            # Call the original function
            result = f(*args, **kwargs)
            duration = time.time() - start_time

            # Handle different response types
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

                logger.info(f"Request completed in {duration:.3f}s")
                return result
            else:
                # Single response object
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

                logger.info(f"Request completed in {duration:.3f}s")
                return result

        except Exception as e:
            # Log error
            duration = time.time() - start_time
            logger.error(f"Endpoint {endpoint_path} failed after {duration:.3f}s: {str(e)}")
            log_response(endpoint_path, 500, None, str(e))
            raise

    return decorated_function

# --- Tracing helpers ---

LITELLM_RESPONSE_HEADERS = [
    'x-litellm-call-id',
    'x-litellm-model-id',
    'x-litellm-model-api-base',
    'x-litellm-response-cost',
    'x-litellm-response-duration-ms',
    'x-litellm-overhead-duration-ms',
    'x-litellm-version',
]


def _tracing_enabled():
    """Check if the tracing master switch is on."""
    return CONFIG.get('tracing', {}).get('enabled', False)


def _tracing_log_headers_enabled():
    """Check if LiteLLM response header extraction is enabled."""
    tracing = CONFIG.get('tracing', {})
    return tracing.get('enabled', False) and tracing.get('log_headers', False)


def build_trace_headers(extra_headers, display_name=None):
    """Merge LiteLLM tracing headers into extra_headers.
    Model-specific headers take precedence over tracing headers.
    display_name is sent via x-litellm-spend-logs-metadata for cost tracking."""
    tracing = CONFIG.get('tracing', {})
    if not tracing.get('enabled', False) or not tracing.get('send_trace_headers', False):
        return extra_headers

    trace_headers = {}
    trace_id = getattr(g, 'trace_id', None)
    request_id = getattr(g, 'request_id', None)

    if trace_id:
        trace_headers['x-litellm-trace-id'] = trace_id
    if request_id:
        trace_headers['x-litellm-call-id'] = request_id

    tags = tracing.get('tags', '')
    if tags:
        trace_headers['x-litellm-tags'] = tags

    if display_name:
        trace_headers['x-litellm-spend-logs-metadata'] = json.dumps({"adapter_model": display_name})

    if not trace_headers:
        return extra_headers

    merged = dict(trace_headers)
    if extra_headers:
        merged.update(extra_headers)
    return merged


def build_trace_body_metadata(display_name=None):
    """Return metadata dict for the request body (trace_name for Langfuse, adapter_model for LiteLLM UI)."""
    tracing = CONFIG.get('tracing', {})
    if not tracing.get('enabled', False) or not tracing.get('send_trace_headers', False):
        return None
    if not display_name:
        return None
    return {"trace_name": display_name, "adapter_model": display_name}


def _capture_litellm_headers(headers):
    """Extract LiteLLM response headers into g.litellm_response_headers."""
    captured = {}
    for h in LITELLM_RESPONSE_HEADERS:
        value = headers.get(h)
        if value is not None:
            captured[h] = value
    try:
        g.litellm_response_headers = captured
    except RuntimeError:
        pass


def _log_litellm_headers():
    """Log captured LiteLLM headers: compact at INFO, full at DEBUG."""
    try:
        headers = getattr(g, 'litellm_response_headers', {})
    except RuntimeError:
        return
    if not headers:
        return
    parts = []
    if 'x-litellm-model-id' in headers:
        parts.append(f"model_id={headers['x-litellm-model-id']}")
    if 'x-litellm-response-cost' in headers:
        parts.append(f"cost=${headers['x-litellm-response-cost']}")
    if 'x-litellm-response-duration-ms' in headers:
        parts.append(f"llm_duration={headers['x-litellm-response-duration-ms']}ms")
    if 'x-litellm-overhead-duration-ms' in headers:
        parts.append(f"overhead={headers['x-litellm-overhead-duration-ms']}ms")
    if 'x-litellm-call-id' in headers:
        parts.append(f"litellm_call_id={headers['x-litellm-call-id']}")
    if parts:
        logger.info(f"LiteLLM: {' '.join(parts)}")
    logger.debug(f"LiteLLM response headers: {headers}")


def remove_thinking_tags(content, model_id, remove_enabled):
    """
    Remove <think> or <thinking> tags from the beginning of content if enabled.
    Logs removed thinking content at DEBUG level.

    Args:
        content: The content to process (may be None or empty)
        model_id: Model identifier for logging
        remove_enabled: Boolean indicating if tag removal is enabled

    Returns:
        Cleaned content or original content if no tags found
    """
    # Early return if feature is disabled or content is empty
    if not remove_enabled or not content:
        return content

    # Regex pattern to match thinking tags at the beginning of content
    # Pattern explanation:
    # ^\s*          - Start of string with optional leading whitespace
    # <think(?:ing)?>  - Match <think> or <thinking>
    # (.*?)         - Capture content inside tags (non-greedy)
    # </think(?:ing)?> - Match closing </think> or </thinking>
    # \s*           - Optional trailing whitespace
    pattern = r'^\s*<think(?:ing)?>(.*?)</think(?:ing)?>\s*'

    match = re.match(pattern, content, re.DOTALL | re.IGNORECASE)
    if match:
        thinking_content = match.group(1)
        cleaned_content = content[match.end():]

        # Log removed thinking content (first 100 chars) at DEBUG level
        thinking_preview = thinking_content[:100] + "..." if len(thinking_content) > 100 else thinking_content
        logger.debug(f"Removed thinking tags from model '{model_id}'. Thinking content ({len(thinking_content)} chars): {thinking_preview}")

        return cleaned_content

    return content

def _process_stream(response_stream, remove_tags, model_id, display_name, make_chunk, usage):
    """
    Process OpenAI streaming response, optionally removing thinking tags.
    Shared logic for both /api/chat and /api/generate endpoints.

    Args:
        response_stream: OpenAI streaming response iterator
        remove_tags: Boolean — whether to strip <think>/<thinking> tags
        model_id: Model identifier for logging
        display_name: Display name for response chunks
        make_chunk: Callable(display_name, content) -> dict for endpoint-specific chunk format
        usage: Mutable dict to store {'prompt_tokens': int, 'completion_tokens': int}

    Yields:
        JSON-encoded response lines (str ending with '\\n')
    """
    if remove_tags:
        state = "INITIAL"
        buffer = ""
        thinking_buffer = ""
        close_tag_buffer = ""

        for chunk in response_stream:
            if chunk.usage:
                usage['prompt_tokens'] = chunk.usage.prompt_tokens
                usage['completion_tokens'] = chunk.usage.completion_tokens

            if not chunk.choices:
                continue

            content = chunk.choices[0].delta.content
            if not content:
                continue

            # STATE: INITIAL
            if state == "INITIAL":
                buffer = content
                state = "DETECTING_OPEN_TAG"
                continue

            # STATE: DETECTING_OPEN_TAG
            elif state == "DETECTING_OPEN_TAG":
                buffer += content
                buffer_lower = buffer.lstrip().lower()

                found_tag = False
                for tag in ["<think>", "<thinking>"]:
                    if buffer_lower.startswith(tag):
                        whitespace_len = len(buffer) - len(buffer.lstrip())
                        tag_end_pos = whitespace_len + len(tag)
                        buffer = buffer[tag_end_pos:]
                        thinking_buffer = ""
                        state = "BUFFERING_THINKING"
                        found_tag = True
                        logger.debug(f"Detected opening {tag} tag for model '{model_id}'")
                        break

                if found_tag:
                    continue

                if len(buffer_lower) > 20 or (len(buffer.lstrip()) > 0 and buffer.lstrip()[0] != '<'):
                    state = "STREAMING_NORMAL"
                    logger.debug(f"No thinking tag detected for model '{model_id}'")
                    if buffer:
                        yield json.dumps(make_chunk(display_name, buffer)) + '\n'
                        buffer = ""
                continue

            # STATE: BUFFERING_THINKING
            elif state == "BUFFERING_THINKING":
                buffer += content

                if "</" in buffer:
                    close_idx = buffer.index("</")
                    thinking_buffer += buffer[:close_idx]
                    close_tag_buffer = buffer[close_idx:]
                    buffer = ""
                    state = "DETECTING_CLOSE_TAG"
                else:
                    if len(buffer) > 1000:
                        thinking_buffer += buffer
                        buffer = ""
                continue

            # STATE: DETECTING_CLOSE_TAG
            elif state == "DETECTING_CLOSE_TAG":
                close_tag_buffer += content
                close_lower = close_tag_buffer.lower()

                found_close = False
                for tag in ["</think>", "</thinking>"]:
                    if close_lower.startswith(tag):
                        remainder = close_tag_buffer[len(tag):].lstrip()

                        preview = thinking_buffer[:100] + "..." if len(thinking_buffer) > 100 else thinking_buffer
                        logger.debug(f"Removed thinking tags from model '{model_id}'. Thinking content ({len(thinking_buffer)} chars): {preview}")

                        state = "STREAMING_NORMAL"
                        thinking_buffer = ""
                        close_tag_buffer = ""
                        found_close = True

                        if remainder:
                            yield json.dumps(make_chunk(display_name, remainder)) + '\n'
                        break

                if found_close:
                    continue

                if len(close_tag_buffer) > 15 or ('>' in close_tag_buffer and not any(close_lower.startswith(t[:len(close_lower)]) for t in ["</think>", "</thinking>"])):
                    thinking_buffer += close_tag_buffer
                    close_tag_buffer = ""
                    state = "BUFFERING_THINKING"
                continue

            # STATE: STREAMING_NORMAL
            elif state == "STREAMING_NORMAL":
                yield json.dumps(make_chunk(display_name, content)) + '\n'

        # End of stream — flush remaining buffered content
        if state == "DETECTING_OPEN_TAG" and buffer:
            yield json.dumps(make_chunk(display_name, buffer)) + '\n'
        elif state in ["BUFFERING_THINKING", "DETECTING_CLOSE_TAG"]:
            fallback = thinking_buffer + close_tag_buffer
            logger.warning(f"Stream ended while buffering thinking content for model '{model_id}'. No closing tag found. Outputting {len(fallback)} chars as fallback.")
            if fallback.strip():
                yield json.dumps(make_chunk(display_name, fallback)) + '\n'

    else:
        # Feature disabled — simple pass-through
        for chunk in response_stream:
            if chunk.usage:
                usage['prompt_tokens'] = chunk.usage.prompt_tokens
                usage['completion_tokens'] = chunk.usage.completion_tokens

            if not chunk.choices:
                continue

            content = chunk.choices[0].delta.content
            if content:
                yield json.dumps(make_chunk(display_name, content)) + '\n'


def resolve_system_prompt(value):
    """
    Resolve system_prompt value to actual prompt text.
    If value ends with '.md', reads content from file (relative to CWD).
    Otherwise returns the string as-is.

    Args:
        value: System prompt string or path to .md file

    Returns:
        Resolved prompt text, or None if empty

    Raises:
        FileNotFoundError: If .md file does not exist
        IOError: If .md file cannot be read
    """
    if not value or not isinstance(value, str):
        return None

    value = value.strip()
    if not value:
        return None

    if value.endswith('.md'):
        file_path = value if os.path.isabs(value) else os.path.join(os.getcwd(), value)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        return content if content else None

    return value

def apply_system_prompt(messages, adapter_params, model_id):
    """
    Apply system prompt from model config to messages list.
    If config has system_prompt and request has system message — replaces it.
    If config has system_prompt and request has no system message — prepends it.
    If config has no system_prompt — returns messages unchanged.

    Args:
        messages: List of message dicts (role/content)
        adapter_params: Adapter parameters from model config
        model_id: Model identifier for logging

    Returns:
        New list of messages (shallow copy, original not mutated)
    """
    system_prompt_value = adapter_params.get('system_prompt')
    if not system_prompt_value:
        return messages

    try:
        resolved = resolve_system_prompt(system_prompt_value)
    except FileNotFoundError:
        logger.error(f"System prompt file not found for model '{model_id}': {system_prompt_value}")
        return messages
    except Exception as e:
        logger.error(f"Failed to read system prompt for model '{model_id}': {e}")
        return messages

    if not resolved:
        return messages

    result = list(messages)
    system_msg = {"role": "system", "content": resolved}

    # Find existing system message index
    system_idx = None
    for i, msg in enumerate(result):
        if isinstance(msg, dict) and msg.get('role') == 'system':
            system_idx = i
            break

    if system_idx is not None:
        result[system_idx] = system_msg
        logger.debug(f"Replaced system message for model '{model_id}' with config system_prompt")
    else:
        result.insert(0, system_msg)
        logger.debug(f"Prepended system message for model '{model_id}' from config system_prompt")

    return result

def apply_prompt_caching(messages, adapter_params, model_id):
    """
    Add cache_control markers to system message content for provider-side prompt caching.

    Transforms system message content from string to array-of-blocks format
    with cache_control annotation. Enables prompt caching on Anthropic and
    Google Gemini via LiteLLM. OpenAI caching is automatic and unaffected.

    Transform:
        {"role": "system", "content": "text"}
    Into:
        {"role": "system", "content": [{"type": "text", "text": "text",
         "cache_control": {"type": "ephemeral"}}]}

    Args:
        messages: List of message dicts (role/content)
        adapter_params: Adapter parameters from model config
        model_id: Model identifier for logging

    Returns:
        New list of messages (shallow copy, original not mutated)
    """
    if not adapter_params.get('prompt_caching'):
        return messages

    result = list(messages)

    for i, msg in enumerate(result):
        if not isinstance(msg, dict) or msg.get('role') != 'system':
            continue

        content = msg.get('content')
        if not content or not isinstance(content, str):
            continue

        result[i] = {
            "role": "system",
            "content": [{
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"}
            }]
        }
        logger.debug(
            f"Added cache_control to system message for model '{model_id}' "
            f"({len(content)} chars)"
        )
        break  # Only cache the first system message

    return result

def load_config(path='config.yml'):
    """Loads and validates YAML configuration file. Raises on any error."""
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if not config.get('openai', {}).get('api_key'):
        raise ValueError("Missing required parameter 'openai.api_key' in config.yml")
    if not config.get('server', {}).get('host'):
        raise ValueError("Missing required parameter 'server.host' in config.yml")
    if not config.get('server', {}).get('port'):
        raise ValueError("Missing required parameter 'server.port' in config.yml")

    # Normalize custom_name values (strip whitespace)
    models_config = config.get('models', [])
    if models_config:
        for model in models_config:
            if isinstance(model, dict) and 'custom_name' in model:
                model['custom_name'] = model['custom_name'].strip()

    # Validate custom_name uniqueness
    if models_config:
        custom_names = []
        for model in models_config:
            if isinstance(model, dict) and 'custom_name' in model:
                custom_names.append(model['custom_name'])

        duplicates = [name for name in set(custom_names) if custom_names.count(name) > 1]
        if duplicates:
            raise ValueError(f"Duplicate custom_name values found: {duplicates}")

    # Validate clients section
    clients_config = config.get('clients')
    if clients_config is not None:
        if not isinstance(clients_config, dict):
            raise ValueError("'clients' must be a dict mapping alias names to IP addresses")
        for alias, ips in clients_config.items():
            if isinstance(ips, str):
                if not ips.strip():
                    raise ValueError(f"Client alias '{alias}': IP address cannot be empty")
                clients_config[alias] = ips.strip()
            elif isinstance(ips, list):
                if not ips:
                    raise ValueError(f"Client alias '{alias}': IP list cannot be empty")
                for i, ip in enumerate(ips):
                    if not isinstance(ip, str) or not ip.strip():
                        raise ValueError(f"Client alias '{alias}': ip[{i}] must be a non-empty string")
                clients_config[alias] = [ip.strip() for ip in ips]
            else:
                raise ValueError(f"Client alias '{alias}': value must be a string or list of strings")

    # Validate ip_routing in model entries
    if models_config:
        for idx, model in enumerate(models_config):
            if not isinstance(model, dict):
                continue
            ip_routing = model.get('ip_routing')
            if ip_routing is None:
                continue

            model_display = model.get('custom_name') or model.get('name', f'models[{idx}]')

            if not isinstance(ip_routing, list):
                raise ValueError(f"Model '{model_display}': 'ip_routing' must be a list")

            seen_ips = set()
            for rule_idx, rule in enumerate(ip_routing):
                if not isinstance(rule, dict):
                    raise ValueError(
                        f"Model '{model_display}': ip_routing[{rule_idx}] must be a dict")

                ip_value = rule.get('ip')
                if not ip_value or not isinstance(ip_value, str) or not ip_value.strip():
                    raise ValueError(
                        f"Model '{model_display}': ip_routing[{rule_idx}] "
                        f"must have a non-empty 'ip' field (alias name or direct IP)")

                ip_value = ip_value.strip()
                rule['ip'] = ip_value

                # Resolve to actual IPs for duplicate checking
                if clients_config and ip_value in clients_config:
                    resolved = clients_config[ip_value]
                    resolved_ips = [resolved] if isinstance(resolved, str) else resolved
                else:
                    resolved_ips = [ip_value]

                for ip in resolved_ips:
                    if ip in seen_ips:
                        raise ValueError(
                            f"Model '{model_display}': duplicate IP '{ip}' in ip_routing")
                    seen_ips.add(ip)

                if 'params' in rule and rule['params'] is not None:
                    if not isinstance(rule['params'], dict):
                        raise ValueError(
                            f"Model '{model_display}': ip_routing[{rule_idx}] 'params' must be a dict")

                if 'headers' in rule and rule['headers'] is not None:
                    if not isinstance(rule['headers'], dict):
                        raise ValueError(
                            f"Model '{model_display}': ip_routing[{rule_idx}] 'headers' must be a dict")

                # Warn about unrecognized keys (likely misplaced params like temperature)
                recognized_keys = {'ip', 'name', 'system_prompt', 'remove_thinking_tags', 'prompt_caching', 'params', 'headers'}
                unknown_keys = set(rule.keys()) - recognized_keys
                if unknown_keys:
                    logger.warning(
                        f"Model '{model_display}': ip_routing[{rule_idx}] has unrecognized keys "
                        f"{unknown_keys}. These will be ignored. "
                        f"Did you mean to put them inside 'params'?")

    # Validate tracing section
    tracing_config = config.get('tracing')
    if tracing_config is not None:
        if not isinstance(tracing_config, dict):
            raise ValueError("'tracing' must be a dict")
        for flag in ('enabled', 'log_headers', 'send_trace_headers'):
            if flag in tracing_config and not isinstance(tracing_config[flag], bool):
                raise ValueError(f"tracing.{flag} must be a boolean (true/false)")
        for field in ('trace_id_prefix', 'tags'):
            if field in tracing_config and not isinstance(tracing_config[field], str):
                raise ValueError(f"tracing.{field} must be a string")

    return config


try:
    CONFIG = load_config()
except Exception as e:
    logger.error(f"Failed to load config: {e}")
    sys.exit(1)

def _configure_log_format(config):
    """Configure log format and filters based on tracing config.
    When tracing is enabled, adds [request_id|trace_id] to every log line."""
    root_logger = logging.getLogger()
    log_level = getattr(logging, config.get('logging', {}).get('log_level', 'INFO').upper(), logging.INFO)
    root_logger.setLevel(log_level)

    tracing_on = config.get('tracing', {}).get('enabled', False)

    if tracing_on:
        fmt = '%(asctime)s - %(levelname)s - [%(trace_context)s] %(message)s'
    else:
        fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(fmt)

    if not root_logger.handlers:
        handler = logging.StreamHandler()
        root_logger.addHandler(handler)

    for handler in root_logger.handlers:
        handler.setFormatter(formatter)
        # Remove existing TraceContextFilter instances before adding
        handler.filters = [f for f in handler.filters if not isinstance(f, TraceContextFilter)]
        if tracing_on:
            handler.addFilter(TraceContextFilter())


_configure_log_format(CONFIG)

try:
    client = OpenAI(
        api_key=CONFIG['openai']['api_key'],
        base_url=CONFIG['openai'].get('base_url', 'https://api.openai.com/v1')
    )
except Exception as e:
    logger.error(f"Error initializing OpenAI client: {e}")
    sys.exit(1)

app = Flask(__name__)
CACHED_MODELS = []

# Config reload state
_config_file_path = 'config.yml'
_last_config_reload_time = None
_config_reload_lock = threading.Lock()

try:
    _last_config_mtime = os.path.getmtime(_config_file_path)
except OSError:
    _last_config_mtime = 0.0


def _check_and_reload_config():
    """Check config.yml mtime; reload everything from scratch if changed.
    Thread-safe: uses _config_reload_lock to prevent concurrent reloads."""
    global CONFIG, CACHED_MODELS, client, _last_config_mtime, _last_config_reload_time

    try:
        current_mtime = os.path.getmtime(_config_file_path)
    except OSError:
        return

    if current_mtime == _last_config_mtime:
        return

    with _config_reload_lock:
        # Re-check after acquiring lock (another thread may have reloaded already)
        if current_mtime == _last_config_mtime:
            return

        _last_config_mtime = current_mtime

        try:
            new_config = load_config(_config_file_path)
        except Exception as e:
            logger.warning(f"Config reload failed, keeping current config: {e}")
            return

        try:
            new_client = OpenAI(
                api_key=new_config['openai']['api_key'],
                base_url=new_config['openai'].get('base_url', 'https://api.openai.com/v1')
            )
        except Exception as e:
            logger.warning(f"Failed to recreate OpenAI client: {e}")
            return

        # Atomic swap: update all globals together
        CONFIG = new_config
        client = new_client

        _configure_log_format(new_config)

        try:
            get_and_cache_models(force_refresh=True)
        except Exception as e:
            logger.warning(f"Failed to rebuild model cache: {e}")

        _last_config_reload_time = datetime.now().isoformat() + "Z"
        logger.info("Config changed, reloaded")


@app.before_request
def _before_request_reload_config():
    _check_and_reload_config()


@app.before_request
def _before_request_set_trace_context():
    """Generate request_id and trace_id for every request when tracing is enabled."""
    tracing = CONFIG.get('tracing', {})
    if not tracing.get('enabled', False):
        g.request_id = None
        g.trace_id = None
        g.litellm_response_headers = {}
        return
    g.request_id = f"req_{uuid.uuid4().hex[:12]}"
    incoming_trace = request.headers.get('x-litellm-trace-id')
    if incoming_trace:
        g.trace_id = incoming_trace
    else:
        prefix = tracing.get('trace_id_prefix', 'oa')
        g.trace_id = f"{prefix}_{uuid.uuid4().hex[:12]}"
    g.litellm_response_headers = {}


def get_display_name(original_name):
    """
    Returns custom_name if set, otherwise original_name.

    Args:
        original_name: Original model name from OpenAI

    Returns:
        custom_name or original_name
    """
    models_config = CONFIG.get('models', [])
    for model in models_config:
        if isinstance(model, dict) and model.get('name') == original_name and 'custom_name' in model:
            return model['custom_name']
    return original_name

def resolve_model_name(client_name):
    """
    Resolves model name from client to original OpenAI name.
    Works with both custom_name and original names as synonyms.

    Args:
        client_name: Model name from client request

    Returns:
        Original model name for OpenAI API
    """
    models_config = CONFIG.get('models', [])

    # First check if this is a custom_name
    for model in models_config:
        if isinstance(model, dict) and model.get('custom_name') == client_name:
            return model['name']

    # If not found as custom_name, return as-is (may be original name)
    return client_name

def get_and_cache_models(force_refresh=False):
    """
    Fetches, filters, maps and caches model list.
    Updated according to Ollama API documentation.

    Args:
        force_refresh: If True, rebuilds cache atomically even if already populated.
    """
    global CACHED_MODELS
    if CACHED_MODELS and not force_refresh:
        return CACHED_MODELS

    action = "Refreshing" if force_refresh else "Requesting"
    logger.info(f"Model cache: {action} model list from OpenAI...")
    try:
        all_models_response = client.models.list().data
        models_config = CONFIG.get('models', [])

        model_details = {
            "parent_model": "",
            "format": "gguf",
            "family": "llama",
            "families": ["llama"],
            "parameter_size": "8.0B",
            "quantization_level": "Q4_0",
        }

        new_models = []
        if models_config:
            # Build lookup by OpenAI model id for metadata
            openai_models_by_id = {m.id: m for m in all_models_response}

            # Create one cached entry per config entry (supports multiple
            # entries for the same model with different custom_names)
            for model_entry in models_config:
                model_name = model_entry.get('name') if isinstance(model_entry, dict) else None
                if not model_name or model_name not in openai_models_by_id:
                    continue
                openai_model = openai_models_by_id[model_name]
                display_name = model_entry.get('custom_name', model_name)
                new_models.append({
                    "name": display_name,
                    "model": display_name,
                    "modified_at": datetime.fromtimestamp(openai_model.created).isoformat() + "Z",
                    "size": 0,
                    "digest": model_name,
                    "details": model_details,
                })
        else:
            # No models configured — expose all available OpenAI models
            for model in all_models_response:
                new_models.append({
                    "name": model.id,
                    "model": model.id,
                    "modified_at": datetime.fromtimestamp(model.created).isoformat() + "Z",
                    "size": 0,
                    "digest": model.id,
                    "details": model_details,
                })

        # Atomic swap — requests see either old or new list, never empty
        CACHED_MODELS = new_models
        logger.info(f"Models successfully loaded and cached. Found: {len(CACHED_MODELS)}")
        return CACHED_MODELS
    except Exception as e:
        logger.error(f"Critical error getting models from OpenAI: {e}")
        if force_refresh:
            logger.warning("Keeping previous model cache after refresh failure")
            return CACHED_MODELS
        return []


@app.route('/api/tags', methods=['GET', 'POST'])
@log_endpoint
def handle_tags():
    models = get_and_cache_models()
    if not models:
        return jsonify({"error": "Failed to get model list from OpenAI"}), 500

    if request.method == 'GET':
        return jsonify({"models": models})

    if request.method == 'POST':
        data, error = validate_json_request()
        if error:
            return error

        model_identifier, error = validate_model_parameter(data)
        if error:
            return error

        # Resolve model_identifier to original name
        original_name = resolve_model_name(model_identifier)
        # Search by display name (name/model) OR by digest (original name)
        found_model = next((
            m for m in models
            if m['name'] == model_identifier
            or m['model'] == model_identifier
            or m['digest'] == original_name
        ), None)
        return jsonify({"models": [found_model]}) if found_model else (jsonify({"error": "Model not found"}), 404)

@app.route('/api/show', methods=['POST'])
@log_endpoint
def show_model():
    """
    Emulates /api/show endpoint.
    Updated according to Ollama API documentation.
    """
    data, error = validate_json_request()
    if error:
        return error

    model_id, error = validate_model_parameter(data)
    if error:
        return error

    # Resolve to original name for modelfile comment
    original_name = resolve_model_name(model_id)

    response_data = {
        "modelfile": "\n".join([
            f'# Modelfile generated by "ollama show"',
            f'# To build a new Modelfile based on this one, replace the FROM line with:',
            f'# FROM {original_name}',
            '',
            f'FROM /Users/matt/.ollama/models/blobs/sha256:200765e1283640ffbd013184bf496e261032fa75b99498a9613be4e94d63ad52',
            'TEMPLATE """{{ .System }}',
            'USER: {{ .Prompt }}',
            'ASSISTANT: """',
            'PARAMETER num_ctx 100000',
            'PARAMETER stop "</s>"',
            'PARAMETER stop "USER:"',
            'PARAMETER stop "ASSISTANT:"'
        ]),

        "parameters": "\n".join([
            'num_keep                       24',
            'stop                           "<|start_header_id|>"',
            'stop                           "<|end_header_id|>"',
            'stop                           "<|eot_id|>"'
        ]),

        "template": "\n".join([
            '{{ if .System }}<|start_header_id|>system<|end_header_id|>',
            '',
            '{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>',
            '',
            '{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>',
            '',
            '{{ .Response }}<|eot_id|>'
        ]),

        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": "llama",
            "families": ["llama"],
            "parameter_size": "8.0B",
            "quantization_level": "Q4_0",
        },

        "model_info": {
            "general.architecture": "llama",
            "general.file_type": 2,
            "llama.context_length": 256000,
        },

        "capabilities": ["completion", "vision"]
    }
    return jsonify(response_data)

def resolve_ip_list(ip_value):
    """Resolve an ip_routing 'ip' field to a list of IP addresses.
    If ip_value matches a key in CONFIG['clients'], returns that client group's IPs.
    Otherwise treats ip_value as a direct IP address."""
    clients = CONFIG.get('clients', {}) or {}
    if isinstance(ip_value, str) and ip_value in clients:
        client_ips = clients[ip_value]
        if isinstance(client_ips, str):
            return [client_ips]
        return list(client_ips)
    if isinstance(ip_value, str):
        return [ip_value]
    return []

def apply_ip_routing(model_entry, client_ip):
    """Apply IP-based routing overrides to a model config entry.
    Returns a new dict with overrides merged; does not mutate the original.
    Scalar fields (name, system_prompt, remove_thinking_tags) are replaced.
    Dict fields (params, headers) are shallow-merged."""
    ip_routing = model_entry.get('ip_routing')
    if not ip_routing or not client_ip:
        return model_entry

    matching_rule = None
    for rule in ip_routing:
        resolved_ips = resolve_ip_list(rule.get('ip', ''))
        if client_ip in resolved_ips:
            matching_rule = rule
            break

    if matching_rule is None:
        return model_entry

    merged = dict(model_entry)
    merged.pop('ip_routing', None)

    for key in ('name', 'system_prompt', 'remove_thinking_tags', 'prompt_caching'):
        if key in matching_rule:
            merged[key] = matching_rule[key]

    for key in ('params', 'headers'):
        if key in matching_rule:
            parent_dict = dict(model_entry.get(key, {}) or {})
            override_dict = matching_rule[key]
            if override_dict is not None:
                parent_dict.update(override_dict)
            merged[key] = parent_dict

    logger.info(
        f"IP routing applied: client_ip={client_ip}, "
        f"model='{model_entry.get('custom_name') or model_entry.get('name')}', "
        f"routed_to='{merged.get('name')}'"
    )

    return merged

def get_model_config(model_id, client_ip=None):
    """
    Returns model configuration split into OpenAI params, adapter params, and headers.
    If client_ip is provided, applies IP-based routing overrides.

    Config format per model:
      - name (required): OpenAI model identifier
      - custom_name (optional): Display name for clients
      - remove_thinking_tags (optional): Remove <think>/<thinking> tags
      - system_prompt (optional): System prompt string or path to .md file
      - params (optional): Dict of OpenAI API parameters (passthrough, including nested)
      - headers (optional): Dict of HTTP headers for OpenAI API requests
      - ip_routing (optional): List of IP-specific overrides

    Args:
        model_id: Model name (custom_name or original)
        client_ip: Client IP address for IP-based routing (optional)

    Returns:
        tuple: (openai_params, adapter_params, headers)
            - openai_params: dict with parameters for OpenAI API (includes model_id)
            - adapter_params: dict with adapter-specific parameters (remove_thinking_tags, etc.)
            - headers: dict with HTTP headers for OpenAI API requests
    """
    # Check for model-specific configuration in config
    models_config = CONFIG.get('models', [])

    # Resolve to original name
    original_name = resolve_model_name(model_id)

    # Find model entry: prefer match by custom_name, then fall back to original name
    model_entry = None
    for model_config in models_config:
        if isinstance(model_config, dict) and model_config.get('custom_name') == model_id:
            model_entry = model_config
            break

    if model_entry is None:
        for model_config in models_config:
            if isinstance(model_config, dict) and model_config.get('name') == original_name:
                model_entry = model_config
                break

    if model_entry is None:
        return {'model_id': original_name}, {}, {}

    # Apply IP-based routing overrides if client_ip provided
    if client_ip:
        model_entry = apply_ip_routing(model_entry, client_ip)

    # Extract adapter params from root level
    adapter_params = {}
    if 'remove_thinking_tags' in model_entry:
        adapter_params['remove_thinking_tags'] = model_entry['remove_thinking_tags']
    if 'system_prompt' in model_entry:
        adapter_params['system_prompt'] = model_entry['system_prompt']
    if 'prompt_caching' in model_entry:
        adapter_params['prompt_caching'] = model_entry['prompt_caching']

    # Extract OpenAI params from 'params' section (passthrough as-is)
    openai_params = dict(model_entry.get('params', {}) or {})
    openai_params['model_id'] = model_entry.get('name', original_name)

    # Extract HTTP headers from 'headers' section
    headers = dict(model_entry.get('headers', {}) or {})

    return openai_params, adapter_params, headers

def create_final_response(model_name, prompt_tokens, completion_tokens, total_duration_ns):
    """
    Helper function for creating final response in Ollama format.
    Updated according to Ollama API documentation.
    """
    return {
        "model": model_name,
        "created_at": datetime.now().isoformat() + "Z",
        "done": True,
        "prompt_eval_count": prompt_tokens,
        "eval_count": completion_tokens,
        "total_duration": total_duration_ns,
        "load_duration": 0, # OpenAI doesn't provide this info
        "prompt_eval_duration": 0, # OpenAI doesn't provide this info
        "eval_duration": int(total_duration_ns * 0.9) if total_duration_ns > 0 else 0 # Approximate value
    }

@app.route('/api/chat', methods=['POST'])
@log_endpoint
def chat():
    """
    Chat endpoint that forwards requests to OpenAI.
    Updated according to Ollama API documentation.
    """
    start_time = time.time()
    try:
        data, error = validate_json_request()
        if error:
            return error

        model_id, error = validate_model_parameter(data)
        if error:
            return error

        # Separate display name and OpenAI name
        display_name = model_id
        original_name = resolve_model_name(model_id)
        client_ip = get_client_ip()

        messages = data.get("messages")
        if not messages:
            return jsonify({"error": "Parameter 'messages' is required"}), 400

        if not isinstance(messages, list) or not messages:
            return jsonify({"error": "Parameter 'messages' must be a non-empty list"}), 400

        stream = data.get("stream", False)

        if stream:
            def generate_stream():
                completion_tokens = 0
                prompt_tokens = -1 # Will be determined later if OpenAI returns it

                _raw_ctx = None
                try:
                    # Get model configuration
                    openai_params, adapter_params, extra_headers = get_model_config(model_id, client_ip=client_ip)

                    # Prepare OpenAI API parameters
                    api_params = {
                        'model': openai_params.get('model_id', original_name),
                        'messages': messages,
                        'stream': True,
                        'stream_options': {"include_usage": True}
                    }

                    # Build extra_body from config params (passthrough without SDK validation)
                    extra_body = {k: v for k, v in openai_params.items() if k != 'model_id'}

                    # Apply system prompt from config (if configured)
                    api_params['messages'] = apply_system_prompt(
                        api_params['messages'], adapter_params, model_id
                    )
                    api_params['messages'] = apply_prompt_caching(
                        api_params['messages'], adapter_params, model_id
                    )

                    # Tracing: merge headers and body metadata
                    merged_headers = build_trace_headers(extra_headers, display_name) or None
                    trace_meta = build_trace_body_metadata(display_name)
                    if trace_meta:
                        extra_body['metadata'] = trace_meta

                    # Create stream (with or without response header extraction)
                    if _tracing_log_headers_enabled():
                        _raw_ctx = client.chat.completions.with_streaming_response.create(
                            **api_params,
                            extra_body=extra_body or None,
                            extra_headers=merged_headers
                        )
                        _raw_response = _raw_ctx.__enter__()
                        _capture_litellm_headers(_raw_response.headers)
                        _log_litellm_headers()
                        response_stream = _raw_response.parse()
                    else:
                        response_stream = client.chat.completions.create(
                            **api_params,
                            extra_body=extra_body or None,
                            extra_headers=merged_headers
                        )

                    # Process stream (with or without thinking tag removal)
                    remove_tags = adapter_params.get('remove_thinking_tags', False)
                    usage = {'prompt_tokens': 0, 'completion_tokens': 0}

                    def make_chat_chunk(dn, content):
                        return {
                            "model": dn,
                            "created_at": datetime.now().isoformat() + "Z",
                            "message": {"role": "assistant", "content": content},
                            "done": False
                        }

                    yield from _process_stream(
                        response_stream, remove_tags, model_id,
                        display_name, make_chat_chunk, usage
                    )

                    prompt_tokens = usage['prompt_tokens']
                    completion_tokens = usage['completion_tokens']

                    # Final chunk with usage info
                    duration_ns = int((time.time() - start_time) * 1e9)
                    final_response = create_final_response(
                        display_name,
                        prompt_tokens if prompt_tokens != -1 else 0,
                        completion_tokens,
                        duration_ns
                    )
                    final_response["message"] = {"role": "assistant", "content": ""}
                    yield json.dumps(final_response) + '\n'

                except Exception as e:
                    yield json.dumps({"error": f"Streaming error: {str(e)}"}) + '\n'
                finally:
                    if _raw_ctx is not None:
                        _raw_ctx.__exit__(None, None, None)

            return Response(stream_with_context(generate_stream()), mimetype='application/x-ndjson')
        else:
            openai_params, adapter_params, extra_headers = get_model_config(model_id, client_ip=client_ip)

            # Prepare OpenAI API parameters
            api_params = {
                'model': openai_params.get('model_id', original_name),
                'messages': messages,
                'stream': False
            }

            # Build extra_body from config params (passthrough without SDK validation)
            extra_body = {k: v for k, v in openai_params.items() if k != 'model_id'}

            # Apply system prompt from config (if configured)
            api_params['messages'] = apply_system_prompt(
                api_params['messages'], adapter_params, model_id
            )
            api_params['messages'] = apply_prompt_caching(
                api_params['messages'], adapter_params, model_id
            )

            # Tracing: merge headers and body metadata
            merged_headers = build_trace_headers(extra_headers, display_name) or None
            trace_meta = build_trace_body_metadata(display_name)
            if trace_meta:
                extra_body['metadata'] = trace_meta

            if _tracing_log_headers_enabled():
                raw = client.chat.completions.with_raw_response.create(
                    **api_params,
                    extra_body=extra_body or None,
                    extra_headers=merged_headers
                )
                response = raw.parse()
                _capture_litellm_headers(raw.headers)
                _log_litellm_headers()
            else:
                response = client.chat.completions.create(
                    **api_params,
                    extra_body=extra_body or None,
                    extra_headers=merged_headers
                )
            duration_ns = int((time.time() - start_time) * 1e9)

            final_response = create_final_response(
                display_name,  # Return display name to client
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                duration_ns
            )
            if not response.choices:
                return jsonify({"error": "No response choices returned from OpenAI"}), 500

            # Remove thinking tags if enabled
            raw_content = response.choices[0].message.content
            cleaned_content = remove_thinking_tags(
                raw_content,
                model_id,
                adapter_params.get('remove_thinking_tags', False)
            )

            final_response["message"] = {
                "role": "assistant",
                "content": cleaned_content
            }
            return jsonify(final_response)

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/generate', methods=['POST'])
@log_endpoint
def generate():
    """
    Ollama /api/generate endpoint - generates completions from a prompt.
    Converts to OpenAI completions API.
    """
    start_time = time.time()
    try:
        data, error = validate_json_request()
        if error:
            return error

        model_id, error = validate_model_parameter(data)
        if error:
            return error

        # Separate display name and OpenAI name
        display_name = model_id
        original_name = resolve_model_name(model_id)
        client_ip = get_client_ip()

        prompt = data.get("prompt")
        if not prompt:
            return jsonify({"error": "Parameter 'prompt' is required"}), 400

        if not isinstance(prompt, str) or not prompt.strip():
            return jsonify({"error": "Parameter 'prompt' must be a non-empty string"}), 400

        stream = data.get("stream", False)

        # Convert prompt to messages format for OpenAI
        messages = [{"role": "user", "content": prompt}]

        # Handle Ollama's 'system' parameter from request
        request_system = data.get("system")
        if request_system and isinstance(request_system, str) and request_system.strip():
            messages.insert(0, {"role": "system", "content": request_system.strip()})

        if stream:
            def generate_stream():
                completion_tokens = 0
                prompt_tokens = 0

                _raw_ctx = None
                try:
                    # Get model configuration
                    openai_params, adapter_params, extra_headers = get_model_config(model_id, client_ip=client_ip)

                    # Prepare OpenAI API parameters
                    api_params = {
                        'model': openai_params.get('model_id', original_name),
                        'messages': messages,
                        'stream': True,
                        'stream_options': {"include_usage": True}
                    }

                    # Build extra_body from config params (passthrough without SDK validation)
                    extra_body = {k: v for k, v in openai_params.items() if k != 'model_id'}

                    # Apply system prompt from config (if configured)
                    api_params['messages'] = apply_system_prompt(
                        api_params['messages'], adapter_params, model_id
                    )
                    api_params['messages'] = apply_prompt_caching(
                        api_params['messages'], adapter_params, model_id
                    )

                    # Tracing: merge headers and body metadata
                    merged_headers = build_trace_headers(extra_headers, display_name) or None
                    trace_meta = build_trace_body_metadata(display_name)
                    if trace_meta:
                        extra_body['metadata'] = trace_meta

                    # Create stream (with or without response header extraction)
                    if _tracing_log_headers_enabled():
                        _raw_ctx = client.chat.completions.with_streaming_response.create(
                            **api_params,
                            extra_body=extra_body or None,
                            extra_headers=merged_headers
                        )
                        _raw_response = _raw_ctx.__enter__()
                        _capture_litellm_headers(_raw_response.headers)
                        _log_litellm_headers()
                        response_stream = _raw_response.parse()
                    else:
                        response_stream = client.chat.completions.create(
                            **api_params,
                            extra_body=extra_body or None,
                            extra_headers=merged_headers
                        )

                    # Process stream (with or without thinking tag removal)
                    remove_tags = adapter_params.get('remove_thinking_tags', False)
                    usage = {'prompt_tokens': 0, 'completion_tokens': 0}

                    def make_generate_chunk(dn, content):
                        return {
                            "model": dn,
                            "created_at": datetime.now().isoformat() + "Z",
                            "response": content,
                            "done": False
                        }

                    yield from _process_stream(
                        response_stream, remove_tags, model_id,
                        display_name, make_generate_chunk, usage
                    )

                    prompt_tokens = usage['prompt_tokens']
                    completion_tokens = usage['completion_tokens']

                    # Final chunk with usage info
                    duration_ns = int((time.time() - start_time) * 1e9)
                    final_response = create_final_response(
                        display_name,
                        prompt_tokens,
                        completion_tokens,
                        duration_ns
                    )
                    final_response["response"] = ""
                    yield json.dumps(final_response) + '\n'

                except Exception as e:
                    yield json.dumps({"error": f"Streaming error: {str(e)}"}) + '\n'
                finally:
                    if _raw_ctx is not None:
                        _raw_ctx.__exit__(None, None, None)

            return Response(stream_with_context(generate_stream()), mimetype='application/x-ndjson')
        else:
            openai_params, adapter_params, extra_headers = get_model_config(model_id, client_ip=client_ip)

            # Prepare OpenAI API parameters
            api_params = {
                'model': openai_params.get('model_id', original_name),
                'messages': messages,
                'stream': False
            }

            # Build extra_body from config params (passthrough without SDK validation)
            extra_body = {k: v for k, v in openai_params.items() if k != 'model_id'}

            # Apply system prompt from config (if configured)
            api_params['messages'] = apply_system_prompt(
                api_params['messages'], adapter_params, model_id
            )
            api_params['messages'] = apply_prompt_caching(
                api_params['messages'], adapter_params, model_id
            )

            # Tracing: merge headers and body metadata
            merged_headers = build_trace_headers(extra_headers, display_name) or None
            trace_meta = build_trace_body_metadata(display_name)
            if trace_meta:
                extra_body['metadata'] = trace_meta

            if _tracing_log_headers_enabled():
                raw = client.chat.completions.with_raw_response.create(
                    **api_params,
                    extra_body=extra_body or None,
                    extra_headers=merged_headers
                )
                response = raw.parse()
                _capture_litellm_headers(raw.headers)
                _log_litellm_headers()
            else:
                response = client.chat.completions.create(
                    **api_params,
                    extra_body=extra_body or None,
                    extra_headers=merged_headers
                )
            duration_ns = int((time.time() - start_time) * 1e9)

            if not response.choices:
                return jsonify({"error": "No response choices returned from OpenAI"}), 500

            final_response = create_final_response(
                display_name,  # Return display name to client
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                duration_ns
            )

            # Remove thinking tags if enabled
            raw_content = response.choices[0].message.content
            cleaned_content = remove_thinking_tags(
                raw_content,
                model_id,
                adapter_params.get('remove_thinking_tags', False)
            )

            final_response["response"] = cleaned_content
            return jsonify(final_response)

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/version', methods=['GET'])
@log_endpoint
def version():
    """Return version information."""
    return jsonify({"version": "0.1.0"})

@app.route('/api/ps', methods=['GET'])
@log_endpoint
def list_running_models():
    """List currently loaded models (mock implementation)."""
    return jsonify({"models": []})

@app.route('/api/embed', methods=['POST'])
@log_endpoint
def embed():
    """Generate embeddings using OpenAI embeddings API."""
    try:
        data, error = validate_json_request()
        if error:
            return error

        model, error = validate_model_parameter(data)
        if error:
            return error

        # Separate display name and OpenAI name
        display_name = model
        original_name = resolve_model_name(model)
        client_ip = get_client_ip()

        # Apply IP routing for embeddings model
        openai_params, _, extra_headers = get_model_config(model, client_ip=client_ip)
        resolved_model = openai_params.get('model_id', original_name)

        input_text = data.get("input")
        if not input_text:
            return jsonify({"error": "Parameter 'input' is required"}), 400

        if not isinstance(input_text, (str, list)):
            return jsonify({"error": "Parameter 'input' must be a string or list of strings"}), 400

        if isinstance(input_text, list) and not input_text:
            return jsonify({"error": "Parameter 'input' cannot be an empty list"}), 400

        if isinstance(input_text, str):
            input_text = [input_text]

        # Tracing: merge headers
        merged_headers = build_trace_headers(extra_headers, display_name) or None

        if _tracing_log_headers_enabled():
            raw = client.embeddings.with_raw_response.create(
                model=resolved_model,
                input=input_text,
                extra_headers=merged_headers
            )
            response = raw.parse()
            _capture_litellm_headers(raw.headers)
            _log_litellm_headers()
        else:
            response = client.embeddings.create(
                model=resolved_model,
                input=input_text,
                extra_headers=merged_headers
            )

        embeddings = [embedding.embedding for embedding in response.data]

        return jsonify({
            "model": display_name,  # Return display name to client
            "embeddings": embeddings,
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": response.usage.prompt_tokens if response.usage else 0
        })

    except Exception as e:
        return jsonify({"error": f"Embedding error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
@log_endpoint
def health_check():
    """Health check endpoint to verify service status."""
    try:
        # Test OpenAI client connection (with timeout to prevent hanging)
        models = client.with_options(timeout=5.0).models.list()
        openai_status = "healthy" if models else "unhealthy"

        # Check if we have cached models
        cached_models_count = len(CACHED_MODELS)

        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat() + "Z",
            "services": {
                "openai": openai_status,
                "cached_models": cached_models_count
            },
            "last_config_reload": _last_config_reload_time,
            "version": "0.1.0"
        }), 200

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat() + "Z",
            "error": str(e)
        }), 503

@app.route('/', methods=['GET'])
@log_endpoint
def root():
    """Root endpoint with basic service information."""
    return jsonify({
        "service": "Ollama to OpenAI Adapter",
        "version": "0.1.0",
        "endpoints": [
            "/api/tags",
            "/api/show",
            "/api/chat",
            "/api/generate",
            "/api/embed",
            "/api/version",
            "/api/ps",
            "/health"
        ]
    })


if __name__ == '__main__':
    get_and_cache_models()

    server_config = CONFIG['server']
    logger.info(f"Starting Ollama -> OpenAI adapter on http://{server_config['host']}:{server_config['port']}")

    # Enable debug mode for auto-reload on code changes
    app.run(
        host=server_config['host'],
        port=server_config['port'],
        debug=True,
        use_reloader=True,
        reloader_type='stat'
    )
