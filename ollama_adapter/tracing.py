"""LiteLLM tracing header handling and request metadata."""

import contextlib
import json
import re
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from flask import g

from ollama_adapter import state

LITELLM_RESPONSE_HEADERS = [
    "x-litellm-call-id",
    "x-litellm-model-id",
    "x-litellm-model-api-base",
    "x-litellm-response-cost",
    "x-litellm-response-duration-ms",
    "x-litellm-overhead-duration-ms",
    "x-litellm-version",
]


def tracing_enabled() -> bool:
    """Check if the tracing master switch is on."""
    return bool(state.CONFIG.get("tracing", {}).get("enabled", False))


def tracing_log_headers_enabled() -> bool:
    """Check if LiteLLM response header extraction is enabled."""
    tracing = state.CONFIG.get("tracing", {})
    return bool(tracing.get("enabled", False)) and bool(tracing.get("log_headers", False))


def _slugify(name: str) -> str:
    """Convert a model name to a URL-friendly slug.

    Example: "GPT-4o Mini" -> "gpt-4o-mini"
    """
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9-]", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug)
    return slug.strip("-")


def _build_grouped_trace_id(grouping: str) -> str:
    """Build a deterministic trace ID based on grouping interval.

    Format: {prefix}_{YYYY}_{MM}_{DD}_{HH}  (hourly)
            {prefix}_{YYYY}_{MM}_{DD}        (daily)
    """
    tracing = state.CONFIG.get("tracing", {})
    prefix = tracing.get("trace_id_prefix", "oa")
    tz_name = tracing.get("timezone", "UTC")
    now = datetime.now(tz=ZoneInfo(tz_name))
    fmt = "%Y_%m_%d_%H" if grouping == "hourly" else "%Y_%m_%d"
    return f"{prefix}_{now.strftime(fmt)}"


def _build_tags(tracing: dict[str, Any], grouping: str, display_name: str | None) -> str:
    """Combine static tags from config with the model slug when grouping is active."""
    parts: list[str] = []
    tags: str = tracing.get("tags", "")
    if tags:
        parts.append(tags)
    if grouping in ("hourly", "daily") and display_name:
        parts.append(_slugify(display_name))
    return ",".join(parts)


def build_trace_headers(extra_headers: dict[str, str] | None, display_name: str | None = None) -> dict[str, str] | None:
    """Merge LiteLLM tracing headers into extra_headers.

    Model-specific headers take precedence over tracing headers.
    display_name is sent via x-litellm-spend-logs-metadata for cost tracking.
    When trace_grouping is "hourly" or "daily", generates a deterministic trace ID.
    """
    tracing = state.CONFIG.get("tracing", {})
    if not tracing.get("enabled", False) or not tracing.get("send_trace_headers", False):
        return extra_headers

    trace_headers: dict[str, str] = {}
    trace_id: str | None = getattr(g, "trace_id", None)
    request_id: str | None = getattr(g, "request_id", None)
    trace_id_incoming: bool = getattr(g, "trace_id_incoming", False)

    grouping = tracing.get("trace_grouping", "")
    if grouping in ("hourly", "daily") and not trace_id_incoming:
        trace_id = _build_grouped_trace_id(grouping)

    if trace_id:
        trace_headers["x-litellm-trace-id"] = trace_id
    if request_id:
        trace_headers["x-litellm-call-id"] = request_id

    combined_tags = _build_tags(tracing, grouping, display_name)
    if combined_tags:
        trace_headers["x-litellm-tags"] = combined_tags

    if display_name:
        trace_headers["x-litellm-spend-logs-metadata"] = json.dumps({"adapter_model": display_name})

    if not trace_headers:
        return extra_headers

    merged = dict(trace_headers)
    if extra_headers:
        merged.update(extra_headers)
    return merged


def build_trace_body_metadata(display_name: str | None = None) -> dict[str, str] | None:
    """Return metadata dict for the request body.

    Includes trace_name for Langfuse and adapter_model for LiteLLM UI.
    """
    tracing = state.CONFIG.get("tracing", {})
    if not tracing.get("enabled", False) or not tracing.get("send_trace_headers", False):
        return None
    if not display_name:
        return None
    prefix = tracing.get("trace_name_prefix", "")
    trace_name = f"{prefix}{display_name}" if prefix else display_name
    return {"trace_name": trace_name, "adapter_model": display_name}


def capture_litellm_headers(headers: Any) -> None:
    """Extract LiteLLM response headers into g.litellm_response_headers."""
    captured: dict[str, str] = {}
    for h in LITELLM_RESPONSE_HEADERS:
        value = headers.get(h)
        if value is not None:
            captured[h] = value
    with contextlib.suppress(RuntimeError):
        g.litellm_response_headers = captured


def log_litellm_headers() -> None:
    """Log captured LiteLLM headers: compact at INFO, full at DEBUG."""
    try:
        headers: dict[str, str] = getattr(g, "litellm_response_headers", {})
    except RuntimeError:
        return
    if not headers:
        return

    parts: list[str] = []
    if "x-litellm-model-id" in headers:
        parts.append(f"model_id={headers['x-litellm-model-id']}")
    if "x-litellm-response-cost" in headers:
        parts.append(f"cost=${headers['x-litellm-response-cost']}")
    if "x-litellm-response-duration-ms" in headers:
        parts.append(f"llm_duration={headers['x-litellm-response-duration-ms']}ms")
    if "x-litellm-overhead-duration-ms" in headers:
        parts.append(f"overhead={headers['x-litellm-overhead-duration-ms']}ms")
    if "x-litellm-call-id" in headers:
        parts.append(f"litellm_call_id={headers['x-litellm-call-id']}")
    if parts:
        state.logger.info("LiteLLM: %s", " ".join(parts))
    state.logger.debug("LiteLLM response headers: %s", headers)
