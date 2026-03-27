import json
from flask import g

from ollama_adapter import state

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
    return state.CONFIG.get('tracing', {}).get('enabled', False)


def _tracing_log_headers_enabled():
    """Check if LiteLLM response header extraction is enabled."""
    tracing = state.CONFIG.get('tracing', {})
    return tracing.get('enabled', False) and tracing.get('log_headers', False)


def build_trace_headers(extra_headers, display_name=None):
    """Merge LiteLLM tracing headers into extra_headers.
    Model-specific headers take precedence over tracing headers.
    display_name is sent via x-litellm-spend-logs-metadata for cost tracking."""
    tracing = state.CONFIG.get('tracing', {})
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
    tracing = state.CONFIG.get('tracing', {})
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
        state.logger.info(f"LiteLLM: {' '.join(parts)}")
    state.logger.debug(f"LiteLLM response headers: {headers}")
