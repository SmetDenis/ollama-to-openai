"""Protocol-agnostic upstream chat-completion pipeline shared by the Ollama and OpenAI routes.

Resolves the model configuration, merges client parameters, prepares the final
`messages` (system prompt + prompt caching), assembles the upstream request with
trace headers, and opens the upstream call. Contains no Flask response formatting.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from ollama_adapter import state
from ollama_adapter.debug_prompt import build_config_view, build_debug_content, build_outgoing_view
from ollama_adapter.models import apply_prompt_caching, apply_system_prompt, get_model_config
from ollama_adapter.tracing import (
    build_trace_body_metadata,
    build_trace_headers,
    capture_litellm_headers,
    log_litellm_headers,
    tracing_log_headers_enabled,
)

# Keys the adapter sets itself; a copy in `extra_body` would silently override them
# (the OpenAI SDK merges `extra_body` over named arguments).
RESERVED_UPSTREAM_KEYS = frozenset({"model", "messages", "stream"})

# Client-supplied keys that could redirect, re-authenticate or re-route the request inside
# a LiteLLM proxy (client-side credentials, provider selection, header injection, nested
# LiteLLM params). Dropped from client parameters; config `params` may still set them.
# Not exhaustive: LiteLLM providers define additional provider-specific auth params.
CLIENT_PARAM_DENYLIST = frozenset(
    {
        "api_key",
        "api_base",
        "base_url",
        "api_version",
        "custom_llm_provider",
        "extra_headers",
        "headers",
        "litellm_params",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "aws_region_name",
        "aws_profile_name",
        "vertex_project",
        "vertex_location",
        "vertex_credentials",
        "azure_ad_token",
    }
)


@dataclass(frozen=True)
class ResolvedModel:
    """Model configuration after custom-name resolution and IP routing."""

    openai_params: dict[str, Any]  # includes the internal "model_id" key
    adapter_params: dict[str, Any]
    headers: dict[str, str]

    @property
    def upstream_model(self) -> str:
        """Return the upstream model ID actually sent to the provider."""
        return str(self.openai_params.get("model_id", ""))


@dataclass(frozen=True)
class MergedParams:
    """Client and config parameters merged into the upstream `extra_body`."""

    extra_body: dict[str, Any]
    client_wants_usage: bool


@dataclass(frozen=True)
class UpstreamRequest:
    """Everything needed to call `chat.completions.create` upstream."""

    display_name: str
    model: str
    messages: list[dict[str, Any]]
    stream: bool
    extra_body: dict[str, Any]
    extra_headers: dict[str, str] | None
    stream_options: dict[str, Any] | None = None

    def create_kwargs(self) -> dict[str, Any]:
        """Build keyword arguments for the OpenAI SDK `create` call."""
        kwargs: dict[str, Any] = {"model": self.model, "messages": self.messages, "stream": self.stream}
        if self.stream_options is not None:
            kwargs["stream_options"] = self.stream_options
        kwargs["extra_body"] = self.extra_body or None
        kwargs["extra_headers"] = self.extra_headers
        return kwargs


def resolve_model(model_id: str, client_ip: str | None) -> ResolvedModel:
    """Resolve the model configuration for `model_id`, applying IP routing for `client_ip`."""
    openai_params, adapter_params, headers = get_model_config(model_id, client_ip=client_ip)
    return ResolvedModel(openai_params=openai_params, adapter_params=adapter_params, headers=headers)


def build_extra_body(openai_params: dict[str, Any], display_name: str) -> dict[str, Any]:
    """Build the Ollama-path `extra_body`: config params plus trace metadata if configured."""
    extra_body = {k: v for k, v in openai_params.items() if k != "model_id"}
    trace_meta = build_trace_body_metadata(display_name)
    if trace_meta:
        extra_body["metadata"] = trace_meta
    return extra_body


def merge_client_params(
    client_params: Mapping[str, Any],
    openai_params: dict[str, Any],
    display_name: str,
    *,
    streaming: bool,
) -> MergedParams:
    """Merge client request parameters with config `params` into an upstream `extra_body`.

    Config params override client params (shallow). `metadata` is dict-merged with
    precedence client < config < trace metadata. When streaming, `include_usage` is
    always requested upstream (the adapter logs usage); `client_wants_usage` records
    whether the client itself asked for the usage chunk.
    """
    client: dict[str, Any] = {}
    for key, value in client_params.items():
        if key in RESERVED_UPSTREAM_KEYS:
            continue
        if key in CLIENT_PARAM_DENYLIST:
            state.logger.debug("Dropping client parameter '%s' (not allowed from clients)", key)
            continue
        client[key] = value

    config = {k: v for k, v in openai_params.items() if k != "model_id" and k not in RESERVED_UPSTREAM_KEYS}
    body: dict[str, Any] = {**client, **config}

    trace_meta = build_trace_body_metadata(display_name)
    config_meta = config.get("metadata")
    if "metadata" in config and not isinstance(config_meta, dict):
        if trace_meta:
            body["metadata"] = trace_meta
    else:
        parts = [m for m in (client.get("metadata"), config_meta, trace_meta) if isinstance(m, dict)]
        if parts:
            body["metadata"] = {k: v for part in parts for k, v in part.items()}

    client_stream_options = client.get("stream_options")
    client_wants_usage = (
        streaming and isinstance(client_stream_options, dict) and bool(client_stream_options.get("include_usage"))
    )
    if streaming:
        stream_options: dict[str, Any] = {}
        if isinstance(client_stream_options, dict):
            stream_options.update(client_stream_options)
        config_stream_options = config.get("stream_options")
        if isinstance(config_stream_options, dict):
            stream_options.update(config_stream_options)
        stream_options["include_usage"] = True
        body["stream_options"] = stream_options
    else:
        body.pop("stream_options", None)

    return MergedParams(extra_body=body, client_wants_usage=client_wants_usage)


def prepare_messages(
    messages: list[dict[str, Any]],
    adapter_params: dict[str, Any],
    model_id: str,
    *,
    warn_on_replace: bool = False,
) -> list[dict[str, Any]]:
    """Apply the configured system prompt and prompt caching. Raises `PromptRenderError`."""
    prepared = apply_system_prompt(messages, adapter_params, model_id, warn_on_replace=warn_on_replace)
    return apply_prompt_caching(prepared, adapter_params, model_id)


def build_upstream_request(  # noqa: PLR0913 — keyword-only request fields
    *,
    display_name: str,
    resolved: ResolvedModel,
    messages: list[dict[str, Any]],
    stream: bool,
    extra_body: dict[str, Any],
    stream_options: dict[str, Any] | None = None,
) -> UpstreamRequest:
    """Assemble the upstream request, merging trace headers into the configured headers."""
    return UpstreamRequest(
        display_name=display_name,
        model=resolved.upstream_model,
        messages=messages,
        stream=stream,
        extra_body=extra_body,
        extra_headers=build_trace_headers(resolved.headers, display_name) or None,
        stream_options=stream_options,
    )


@contextmanager
def open_chat_completion(req: UpstreamRequest) -> Iterator[Any]:
    """Call `chat.completions.create` upstream and yield the response (or stream).

    When LiteLLM header logging is on, the raw/streaming response variants are used so
    response headers can be captured. The underlying HTTP response is always closed on exit.
    """
    client = state.client
    assert client is not None  # noqa: S101
    kwargs = req.create_kwargs()

    if tracing_log_headers_enabled():
        if req.stream:
            with client.chat.completions.with_streaming_response.create(**kwargs) as raw_stream:
                capture_litellm_headers(raw_stream.headers)
                log_litellm_headers()
                yield raw_stream.parse()
            return
        raw = client.chat.completions.with_raw_response.create(**kwargs)
        response = raw.parse()
        capture_litellm_headers(raw.headers)
        log_litellm_headers()
        yield response
        return

    response = client.chat.completions.create(**kwargs)
    try:
        yield response
    finally:
        close = getattr(response, "close", None)
        if req.stream and callable(close):
            close()


def build_debug_text(
    *,
    model_id: str,
    display_name: str,
    messages: list[dict[str, Any]],
    resolved: ResolvedModel,
    extra_body: dict[str, Any],
) -> str:
    """Render the debug output: combined config, outgoing request, compiled messages."""
    merged_headers = build_trace_headers(resolved.headers, display_name)
    config_view = build_config_view(model_id, resolved.openai_params, resolved.adapter_params, resolved.headers)
    outgoing_view = build_outgoing_view(resolved.openai_params, extra_body, merged_headers)
    return build_debug_content(
        messages, resolved.adapter_params, model_id, config_view=config_view, outgoing_view=outgoing_view
    )
