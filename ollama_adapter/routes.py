"""Flask API routes implementing the Ollama-compatible interface."""

import json
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from flask import Blueprint, Response, jsonify, request, stream_with_context

from ollama_adapter import state
from ollama_adapter.debug_prompt import build_debug_content, is_debug_trigger, last_user_text
from ollama_adapter.error_formatter import format_error_text
from ollama_adapter.error_formatter import is_enabled as error_handling_enabled
from ollama_adapter.logging_utils import (
    get_client_ip,
    log_endpoint,
    validate_json_request,
    validate_model_parameter,
)
from ollama_adapter.models import (
    apply_prompt_caching,
    apply_system_prompt,
    create_final_response,
    get_and_cache_models,
    get_model_config,
    resolve_model_name,
)
from ollama_adapter.prompt_renderer import PromptRenderError
from ollama_adapter.thinking import StreamContext, process_stream, remove_thinking_tags
from ollama_adapter.tracing import (
    build_trace_body_metadata,
    build_trace_headers,
    capture_litellm_headers,
    log_litellm_headers,
    tracing_log_headers_enabled,
)

bp = Blueprint("api", __name__)


@dataclass(frozen=True)
class _CompletionContext:
    """Shared parameters for streaming and non-streaming completion calls."""

    model_id: str
    display_name: str
    original_name: str
    messages: list[dict[str, Any]]
    start_time: float


_PROMPT_ERROR_PREFIX = "[PROMPT ERROR]"


def _build_prompt_error_payload(ctx: _CompletionContext, response_key: str, error_message: str) -> dict[str, Any]:
    """Build an Ollama-format response carrying a prompt render error as assistant content."""
    payload = create_final_response(ctx.display_name, 0, 0, 0)
    error_text = f"{_PROMPT_ERROR_PREFIX} {error_message}"
    if response_key == "message":
        payload["message"] = {"role": "assistant", "content": error_text}
    else:
        payload["response"] = error_text
    return payload


def _build_runtime_error_payload(display_name: str, response_key: str, error_text: str) -> dict[str, Any]:
    """Build an Ollama-format response carrying a runtime error as assistant content."""
    payload = create_final_response(display_name, 0, 0, 0)
    if response_key == "message":
        payload["message"] = {"role": "assistant", "content": error_text}
    else:
        payload["response"] = error_text
    return payload


def _yield_runtime_error_chunks(
    ctx: _CompletionContext,
    response_key: str,
    make_chunk: Callable[[str, str], dict[str, Any]],
    error_text: str,
) -> Generator[str]:
    """Yield two ndjson chunks: the runtime-error message and the final done marker."""
    yield json.dumps(make_chunk(ctx.display_name, error_text)) + "\n"
    final_payload = _build_runtime_error_payload(ctx.display_name, response_key, "")
    final_payload["done"] = True
    yield json.dumps(final_payload) + "\n"


def _build_api_params(ctx: _CompletionContext, openai_params: dict[str, Any], *, streaming: bool) -> dict[str, Any]:
    """Assemble the OpenAI Chat Completions API parameters."""
    api_params: dict[str, Any] = {
        "model": openai_params.get("model_id", ctx.original_name),
        "messages": ctx.messages,
        "stream": streaming,
    }
    if streaming:
        api_params["stream_options"] = {"include_usage": True}
    return api_params


def _build_extra_body(openai_params: dict[str, Any], ctx: _CompletionContext) -> dict[str, Any]:
    """Build the OpenAI `extra_body` dict, including trace metadata if configured."""
    extra_body = {k: v for k, v in openai_params.items() if k != "model_id"}
    trace_meta = build_trace_body_metadata(ctx.display_name)
    if trace_meta:
        extra_body["metadata"] = trace_meta
    return extra_body


def _yield_prompt_error_chunks(
    ctx: _CompletionContext,
    response_key: str,
    make_chunk: Callable[[str, str], dict[str, Any]],
    exc: PromptRenderError,
) -> Generator[str]:
    """Yield two ndjson chunks: the error message and the final done marker."""
    error_text = f"{_PROMPT_ERROR_PREFIX} {exc}"
    yield json.dumps(make_chunk(ctx.display_name, error_text)) + "\n"
    final_error = _build_prompt_error_payload(ctx, response_key, str(exc))
    if response_key == "message":
        final_error["message"] = {"role": "assistant", "content": ""}
    else:
        final_error["response"] = ""
    yield json.dumps(final_error) + "\n"


def _call_openai_streaming(
    ctx: _CompletionContext,
    make_chunk: Callable[[str, str], dict[str, Any]],
    response_key: str,
) -> Response:
    """Execute a streaming completion call shared by chat() and generate()."""
    client_ip = get_client_ip()

    def generate_stream() -> Generator[str]:
        assert state.client is not None  # noqa: S101
        raw_ctx = None
        try:
            openai_params, adapter_params, extra_headers = get_model_config(ctx.model_id, client_ip=client_ip)
            api_params = _build_api_params(ctx, openai_params, streaming=True)
            extra_body = _build_extra_body(openai_params, ctx)

            try:
                api_params["messages"] = apply_system_prompt(api_params["messages"], adapter_params, ctx.model_id)
            except PromptRenderError as exc:
                state.logger.error("Prompt render failed (model=%s): %s", ctx.model_id, exc, exc_info=True)
                yield from _yield_prompt_error_chunks(ctx, response_key, make_chunk, exc)
                return
            api_params["messages"] = apply_prompt_caching(api_params["messages"], adapter_params, ctx.model_id)

            merged_headers = build_trace_headers(extra_headers, ctx.display_name) or None

            if tracing_log_headers_enabled():
                raw_ctx = state.client.chat.completions.with_streaming_response.create(
                    **api_params, extra_body=extra_body or None, extra_headers=merged_headers
                )
                raw_response = raw_ctx.__enter__()
                capture_litellm_headers(raw_response.headers)
                log_litellm_headers()
                response_stream = raw_response.parse()
            else:
                response_stream = state.client.chat.completions.create(
                    **api_params, extra_body=extra_body or None, extra_headers=merged_headers
                )

            stream_ctx = StreamContext(
                model_id=ctx.model_id,
                display_name=ctx.display_name,
                make_chunk=make_chunk,
                remove_tags=adapter_params.get("remove_thinking_tags", False),
            )
            usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

            yield from process_stream(response_stream, stream_ctx, usage)

            duration_ns = int((time.time() - ctx.start_time) * 1e9)
            final_response = create_final_response(
                ctx.display_name, usage["prompt_tokens"], usage["completion_tokens"], duration_ns
            )

            if response_key == "message":
                final_response["message"] = {"role": "assistant", "content": ""}
            else:
                final_response["response"] = ""

            yield json.dumps(final_response) + "\n"

        except Exception as e:  # noqa: BLE001
            state.logger.exception("Streaming error (model=%s)", ctx.model_id)
            if error_handling_enabled():
                yield from _yield_runtime_error_chunks(ctx, response_key, make_chunk, format_error_text(e))
            else:
                yield json.dumps({"error": f"Streaming error: {e!s}"}) + "\n"
        finally:
            if raw_ctx is not None:
                raw_ctx.__exit__(None, None, None)

    return Response(stream_with_context(generate_stream()), mimetype="application/x-ndjson")


def _call_openai_non_streaming(ctx: _CompletionContext, response_key: str) -> Response | tuple[Response, int]:
    """Execute a non-streaming completion call shared by chat() and generate()."""
    assert state.client is not None  # noqa: S101
    client_ip = get_client_ip()
    openai_params, adapter_params, extra_headers = get_model_config(ctx.model_id, client_ip=client_ip)

    api_params = _build_api_params(ctx, openai_params, streaming=False)
    extra_body = _build_extra_body(openai_params, ctx)

    try:
        api_params["messages"] = apply_system_prompt(api_params["messages"], adapter_params, ctx.model_id)
    except PromptRenderError as exc:
        state.logger.error("Prompt render failed (model=%s): %s", ctx.model_id, exc, exc_info=True)
        return jsonify(_build_prompt_error_payload(ctx, response_key, str(exc)))
    api_params["messages"] = apply_prompt_caching(api_params["messages"], adapter_params, ctx.model_id)

    merged_headers = build_trace_headers(extra_headers, ctx.display_name) or None

    if tracing_log_headers_enabled():
        raw = state.client.chat.completions.with_raw_response.create(
            **api_params, extra_body=extra_body or None, extra_headers=merged_headers
        )
        response = raw.parse()
        capture_litellm_headers(raw.headers)
        log_litellm_headers()
    else:
        response = state.client.chat.completions.create(
            **api_params, extra_body=extra_body or None, extra_headers=merged_headers
        )

    duration_ns = int((time.time() - ctx.start_time) * 1e9)

    if not response.choices:
        if error_handling_enabled():
            error_text = format_error_text(RuntimeError("No response choices returned from upstream"))
            return jsonify(_build_runtime_error_payload(ctx.display_name, response_key, error_text))
        return jsonify({"error": "No response choices returned from OpenAI"}), 500

    final_response = create_final_response(
        ctx.display_name, response.usage.prompt_tokens, response.usage.completion_tokens, duration_ns
    )

    raw_content = response.choices[0].message.content
    cleaned_content = remove_thinking_tags(
        raw_content, ctx.model_id, remove_enabled=adapter_params.get("remove_thinking_tags", False)
    )

    if response_key == "message":
        final_response["message"] = {"role": "assistant", "content": cleaned_content}
    else:
        final_response["response"] = cleaned_content

    return jsonify(final_response)


def _runtime_error_response(display_name: str, response_key: str, exc: Exception) -> Response:
    """Build a Flask `Response` that delivers a runtime error as Ollama-format content."""
    payload = _build_runtime_error_payload(display_name, response_key, format_error_text(exc))
    return jsonify(payload)


def _build_debug_payload(ctx: _CompletionContext, response_key: str, content: str) -> dict[str, Any]:
    """Build an Ollama-format response carrying the compiled debug prompt as assistant content."""
    payload = create_final_response(ctx.display_name, 0, 0, 0)
    if response_key == "message":
        payload["message"] = {"role": "assistant", "content": content}
    else:
        payload["response"] = content
    return payload


def _yield_debug_chunks(
    ctx: _CompletionContext,
    response_key: str,
    make_chunk: Callable[[str, str], dict[str, Any]],
    content: str,
) -> Generator[str]:
    """Yield the debug content chunk followed by the final done marker (ndjson)."""
    yield json.dumps(make_chunk(ctx.display_name, content)) + "\n"
    final_payload = _build_debug_payload(ctx, response_key, "")
    final_payload["done"] = True
    yield json.dumps(final_payload) + "\n"


def _debug_response(
    ctx: _CompletionContext,
    response_key: str,
    make_chunk: Callable[[str, str], dict[str, Any]],
    *,
    streaming: bool,
) -> Response:
    """Return the compiled-prompt debug output without calling the upstream model."""
    client_ip = get_client_ip()
    _, adapter_params, _ = get_model_config(ctx.model_id, client_ip=client_ip)
    content = build_debug_content(ctx.messages, adapter_params, ctx.model_id)
    if streaming:
        return Response(
            stream_with_context(_yield_debug_chunks(ctx, response_key, make_chunk, content)),
            mimetype="application/x-ndjson",
        )
    return jsonify(_build_debug_payload(ctx, response_key, content))


@bp.route("/api/tags", methods=["GET", "POST"])
@log_endpoint
def handle_tags() -> Response | tuple[Response, int]:
    """List or filter available models."""
    models = get_and_cache_models()
    if not models:
        return jsonify({"error": "Failed to get model list from OpenAI"}), 500

    if request.method == "GET":
        return jsonify({"models": models})

    result = validate_json_request()
    if isinstance(result, tuple):
        return result
    data = result

    model_result = validate_model_parameter(data)
    if isinstance(model_result, tuple):
        return model_result
    model_identifier = model_result

    original_name = resolve_model_name(model_identifier)
    found_model = next(
        (
            m
            for m in models
            if m["name"] == model_identifier or m["model"] == model_identifier or m["digest"] == original_name
        ),
        None,
    )
    if found_model:
        return jsonify({"models": [found_model]})
    return jsonify({"error": "Model not found"}), 404


@bp.route("/api/show", methods=["POST"])
@log_endpoint
def show_model() -> Response | tuple[Response, int]:
    """Emulate /api/show endpoint with model metadata."""
    result = validate_json_request()
    if isinstance(result, tuple):
        return result
    data = result

    model_result = validate_model_parameter(data)
    if isinstance(model_result, tuple):
        return model_result
    model_id = model_result

    original_name = resolve_model_name(model_id)

    response_data = {
        "modelfile": (
            f'# Modelfile generated by "ollama show"\n'
            f"# To build a new Modelfile based on this one, replace the FROM line with:\n"
            f"# FROM {original_name}\n"
            f"\n"
            f"FROM /Users/matt/.ollama/models/blobs/"
            f"sha256:200765e1283640ffbd013184bf496e261032fa75b99498a9613be4e94d63ad52\n"
            f'TEMPLATE """{{{{ .System }}}}\n'
            f"USER: {{{{ .Prompt }}}}\n"
            f'ASSISTANT: """\n'
            f"PARAMETER num_ctx 100000\n"
            f'PARAMETER stop "</s>"\n'
            f'PARAMETER stop "USER:"\n'
            f'PARAMETER stop "ASSISTANT:"'
        ),
        "parameters": (
            "num_keep                       24\n"
            'stop                           "<|start_header_id|>"\n'
            'stop                           "<|end_header_id|>"\n'
            'stop                           "<|eot_id|>"'
        ),
        "template": (
            "{{ if .System }}<|start_header_id|>system<|end_header_id|>\n"
            "\n"
            "{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>\n"
            "\n"
            "{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>\n"
            "\n"
            "{{ .Response }}<|eot_id|>"
        ),
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
        "capabilities": ["completion", "vision"],
    }
    return jsonify(response_data)


@bp.route("/api/chat", methods=["POST"])
@log_endpoint
def chat() -> Response | tuple[Response, int]:  # noqa: PLR0911
    """Forward chat requests to OpenAI."""
    start_time = time.time()
    display_name_for_error = "unknown"
    try:
        result = validate_json_request()
        if isinstance(result, tuple):
            return result
        data = result

        model_result = validate_model_parameter(data)
        if isinstance(model_result, tuple):
            return model_result
        model_id = model_result
        display_name_for_error = model_id

        original_name = resolve_model_name(model_id)

        messages = data.get("messages")
        if not isinstance(messages, list) or not messages:
            return jsonify({"error": "Parameter 'messages' must be a non-empty list"}), 400

        ctx = _CompletionContext(
            model_id=model_id,
            display_name=model_id,
            original_name=original_name,
            messages=messages,
            start_time=start_time,
        )

        def make_chat_chunk(dn: str, content: str) -> dict[str, Any]:
            return {
                "model": dn,
                "created_at": datetime.now(tz=UTC).isoformat(),
                "message": {"role": "assistant", "content": content},
                "done": False,
            }

        streaming = bool(data.get("stream", False))

        if (text := last_user_text(messages)) is not None and is_debug_trigger(text):
            return _debug_response(ctx, "message", make_chat_chunk, streaming=streaming)

        if streaming:
            return _call_openai_streaming(ctx, make_chat_chunk, "message")
        return _call_openai_non_streaming(ctx, "message")

    except Exception as e:  # noqa: BLE001
        state.logger.exception("Chat endpoint error")
        if error_handling_enabled():
            return _runtime_error_response(display_name_for_error, "message", e)
        return jsonify({"error": f"Internal server error: {e!s}"}), 500


@bp.route("/api/generate", methods=["POST"])
@log_endpoint
def generate() -> Response | tuple[Response, int]:  # noqa: PLR0911
    """Generate completions from a prompt (Ollama /api/generate)."""
    start_time = time.time()
    display_name_for_error = "unknown"
    try:
        result = validate_json_request()
        if isinstance(result, tuple):
            return result
        data = result

        model_result = validate_model_parameter(data)
        if isinstance(model_result, tuple):
            return model_result
        model_id = model_result
        display_name_for_error = model_id

        original_name = resolve_model_name(model_id)

        prompt = data.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return jsonify({"error": "Parameter 'prompt' must be a non-empty string"}), 400

        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]

        request_system = data.get("system")
        if request_system and isinstance(request_system, str) and request_system.strip():
            messages.insert(0, {"role": "system", "content": request_system.strip()})

        ctx = _CompletionContext(
            model_id=model_id,
            display_name=model_id,
            original_name=original_name,
            messages=messages,
            start_time=start_time,
        )

        def make_generate_chunk(dn: str, content: str) -> dict[str, Any]:
            return {
                "model": dn,
                "created_at": datetime.now(tz=UTC).isoformat(),
                "response": content,
                "done": False,
            }

        streaming = bool(data.get("stream", False))

        if is_debug_trigger(prompt):
            return _debug_response(ctx, "response", make_generate_chunk, streaming=streaming)

        if streaming:
            return _call_openai_streaming(ctx, make_generate_chunk, "response")
        return _call_openai_non_streaming(ctx, "response")

    except Exception as e:  # noqa: BLE001
        state.logger.exception("Generate endpoint error")
        if error_handling_enabled():
            return _runtime_error_response(display_name_for_error, "response", e)
        return jsonify({"error": f"Internal server error: {e!s}"}), 500


@bp.route("/api/version", methods=["GET"])
@log_endpoint
def version() -> Response:
    """Return version information."""
    return jsonify({"version": "0.1.0"})


@bp.route("/api/ps", methods=["GET"])
@log_endpoint
def list_running_models() -> Response:
    """List currently loaded models (mock implementation)."""
    return jsonify({"models": []})


@bp.route("/api/embed", methods=["POST"])
@log_endpoint
def embed() -> Response | tuple[Response, int]:
    """Generate embeddings using OpenAI embeddings API."""
    try:
        result = validate_json_request()
        if isinstance(result, tuple):
            return result
        data = result

        model_result = validate_model_parameter(data)
        if isinstance(model_result, tuple):
            return model_result
        model = model_result

        display_name = model
        original_name = resolve_model_name(model)
        client_ip = get_client_ip()

        openai_params, _, extra_headers = get_model_config(model, client_ip=client_ip)
        resolved_model = openai_params.get("model_id", original_name)

        input_text = data.get("input")
        if not input_text or not isinstance(input_text, str | list):
            return jsonify({"error": "Parameter 'input' must be a non-empty string or list of strings"}), 400

        if isinstance(input_text, str):
            input_text = [input_text]

        assert state.client is not None  # noqa: S101
        merged_headers = build_trace_headers(extra_headers, display_name) or None

        if tracing_log_headers_enabled():
            raw = state.client.embeddings.with_raw_response.create(
                model=resolved_model, input=input_text, extra_headers=merged_headers
            )
            response = raw.parse()
            capture_litellm_headers(raw.headers)
            log_litellm_headers()
        else:
            response = state.client.embeddings.create(
                model=resolved_model, input=input_text, extra_headers=merged_headers
            )

        embeddings = [embedding.embedding for embedding in response.data]

        return jsonify(
            {
                "model": display_name,
                "embeddings": embeddings,
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": response.usage.prompt_tokens if response.usage else 0,
            }
        )

    except Exception as e:  # noqa: BLE001
        state.logger.exception("Embed endpoint error")
        return jsonify({"error": f"Embedding error: {e!s}"}), 500


_HEALTH_CHECK_TIMEOUT = 5.0


@bp.route("/health", methods=["GET"])
@log_endpoint
def health_check() -> tuple[Response, int]:
    """Verify service status and OpenAI connectivity."""
    try:
        assert state.client is not None  # noqa: S101
        models = state.client.with_options(timeout=_HEALTH_CHECK_TIMEOUT).models.list()
        openai_status = "healthy" if models else "unhealthy"

        return jsonify(
            {
                "status": "healthy",
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "services": {"openai": openai_status, "cached_models": len(state.CACHED_MODELS)},
                "last_config_reload": state.last_config_reload_time,
                "version": "0.1.0",
            }
        ), 200

    except Exception:  # noqa: BLE001
        state.logger.exception("Health check failed")
        return jsonify(
            {
                "status": "unhealthy",
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "error": "OpenAI API unreachable",
            }
        ), 503


@bp.route("/", methods=["GET"])
@log_endpoint
def root() -> Response:
    """Return basic service information."""
    return jsonify(
        {
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
                "/health",
            ],
        }
    )
