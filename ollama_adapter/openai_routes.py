"""Flask blueprint implementing the OpenAI-compatible `/v1` API.

`/v1` is a faithful passthrough on top of the shared upstream pipeline (`completion.py`):
client parameters are honoured (config `params` win), upstream fields are preserved,
errors use real HTTP status codes, and streaming uses server-sent events.
"""

import hmac
import itertools
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack
from datetime import datetime
from functools import wraps
from typing import Any

from flask import Blueprint, Response, jsonify, request, stream_with_context

from ollama_adapter import state
from ollama_adapter.completion import (
    UpstreamRequest,
    build_debug_text,
    build_upstream_request,
    merge_client_params,
    open_chat_completion,
    prepare_messages,
    resolve_model,
)
from ollama_adapter.debug_prompt import is_debug_trigger, last_user_text
from ollama_adapter.logging_utils import get_client_ip, log_endpoint
from ollama_adapter.models import get_and_cache_models, is_model_allowed, resolve_model_name
from ollama_adapter.openai_errors import OpenAIHTTPError, error_response, render_error, to_openai_error
from ollama_adapter.openai_translate import (
    SSE_DONE,
    UsageSink,
    chat_chunk_to_text_completion,
    chat_completion_payload,
    chat_to_text_completion,
    debug_chat_chunks,
    debug_chat_completion,
    iter_chat_chunks,
    sse,
)

bp = Blueprint("openai_api", __name__)

_HTTP_BAD_REQUEST = 400
_HTTP_UNAUTHORIZED = 401
_HTTP_NOT_FOUND = 404
_OWNED_BY = "ollama-openai"
_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
_END = object()

# Legacy completions parameters that cannot be emulated through chat completions.
_LEGACY_ONLY_PARAMS = ("prompt", "echo", "suffix", "logprobs", "best_of")

ChunkIterFactory = Callable[[Iterable[Any], UsageSink], Iterator[dict[str, Any]]]
ChunkTransform = Callable[[dict[str, Any]], dict[str, Any] | None]


_V1_PREFIX = "/v1"


def openai_api_enabled(section: dict[str, Any] | None = None) -> bool:
    """Whether the `/v1` endpoints are enabled (default: true).

    Pass an already-read `openai_api` section to evaluate one consistent config snapshot.
    """
    if section is None:
        section = state.CONFIG.get("openai_api") or {}
    return bool(section.get("enabled", True))


def is_v1_path(path: str) -> bool:
    """Return True for `/v1` and every path below it."""
    return path == _V1_PREFIX or path.startswith(_V1_PREFIX + "/")


def _is_authorized(keys: list[str]) -> bool:
    """Check the Bearer token against every configured key in constant time."""
    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        return False
    matched = False
    for key in keys:
        matched |= hmac.compare_digest(token.encode(), key.encode())
    return matched


@bp.before_app_request
def _guard() -> tuple[Response, int] | None:
    """Reject `/v1` requests when the API is disabled or the API key is missing/invalid.

    Registered app-wide (not blueprint-scoped) so it also runs for unmatched `/v1` URLs and
    wrong methods, before Flask raises 404/405. Reads the `openai_api` section exactly once
    so a concurrent hot-reload cannot mix the old `enabled` flag with new `api_keys`.
    """
    if not is_v1_path(request.path):
        return None

    section = state.CONFIG.get("openai_api") or {}
    if not openai_api_enabled(section):
        return render_error(
            OpenAIHTTPError(
                _HTTP_NOT_FOUND, f"Unknown request URL: {request.method} {request.path}", code="unknown_url"
            )
        )

    keys = section.get("api_keys") or []
    if keys and not _is_authorized(keys):
        state.logger.warning("Rejected %s: missing or invalid API key (client_ip=%s)", request.path, get_client_ip())
        return render_error(
            OpenAIHTTPError(
                _HTTP_UNAUTHORIZED,
                "Incorrect API key provided.",
                code="invalid_api_key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        )
    return None


def _openai_errors(f: Callable[..., Any]) -> Callable[..., Any]:
    """Translate any exception escaping a view into an OpenAI-format HTTP error response."""

    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return f(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            return error_response(exc)

    return wrapper


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def _bad_request(message: str, *, param: str | None = None, code: str | None = None) -> OpenAIHTTPError:
    return OpenAIHTTPError(_HTTP_BAD_REQUEST, message, param=param, code=code)


def _json_body() -> dict[str, Any]:
    data = request.get_json(force=True, silent=True)
    if not isinstance(data, dict) or not data:
        msg = "Request body must be a non-empty JSON object"
        raise _bad_request(msg)
    return data


def _require_model(body: dict[str, Any]) -> str:
    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        msg = "'model' is required and must be a non-empty string"
        raise _bad_request(msg, param="model")
    return model.strip()


def _parse_stream(body: dict[str, Any]) -> bool:
    stream = body.get("stream")
    if stream is None:
        return False
    if not isinstance(stream, bool):
        msg = "'stream' must be a boolean"
        raise _bad_request(msg, param="stream")
    return stream


def _parse_prompt(prompt: Any) -> str:
    if isinstance(prompt, list) and len(prompt) == 1:
        prompt = prompt[0]
    if not isinstance(prompt, str) or not prompt.strip():
        msg = "'prompt' must be a non-empty string (batched prompts and token arrays are not supported)"
        raise _bad_request(msg, param="prompt")
    return prompt


def _reject_legacy_only_params(body: dict[str, Any]) -> None:
    unsupported = {
        "echo": bool(body.get("echo")),
        "suffix": isinstance(body.get("suffix"), str) and bool(body["suffix"]),
        # `false` means "not requested"; any other value (including 0) asks for logprobs.
        "logprobs": body.get("logprobs") is not None and body.get("logprobs") is not False,
        "best_of": body.get("best_of") not in (None, 1),
    }
    for param, is_set in unsupported.items():
        if is_set:
            msg = f"'{param}' is not supported by this server's /v1/completions endpoint"
            raise _bad_request(msg, param=param, code="unsupported_parameter")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _model_object(entry: dict[str, Any]) -> dict[str, Any]:
    try:
        created = int(datetime.fromisoformat(entry["modified_at"]).timestamp())
    except (KeyError, TypeError, ValueError):
        created = 0
    return {"id": entry["name"], "object": "model", "created": created, "owned_by": _OWNED_BY}


def _cached_models() -> list[dict[str, Any]]:
    """Return the model list; upstream failures propagate and map to their real HTTP status."""
    return get_and_cache_models(raise_errors=True)


@bp.route("/v1/models", methods=["GET"])
@log_endpoint
@_openai_errors
def list_models() -> Response:
    """List available models in OpenAI format."""
    return jsonify({"object": "list", "data": [_model_object(m) for m in _cached_models()]})


@bp.route("/v1/models/<path:model_id>", methods=["GET"])
@log_endpoint
@_openai_errors
def retrieve_model(model_id: str) -> Response:
    """Return a single model by client-facing or upstream name."""
    original_name = resolve_model_name(model_id)
    found = next((m for m in _cached_models() if model_id == m["name"] or original_name == m["digest"]), None)
    if found is None:
        msg = f"The model '{model_id}' does not exist"
        raise OpenAIHTTPError(_HTTP_NOT_FOUND, msg, param="model", code="model_not_found")
    return jsonify(_model_object(found))


# ---------------------------------------------------------------------------
# Completions
# ---------------------------------------------------------------------------


def _log_usage(display_name: str, usage: Any) -> None:
    if usage:
        state.logger.info("Usage (model=%s): %s", display_name, usage)


def _debug_response(display_name: str, content: str, *, streaming: bool, include_usage: bool, legacy: bool) -> Response:
    """Return the compiled-prompt debug output without calling the upstream model."""
    if not streaming:
        payload = debug_chat_completion(display_name, content)
        return jsonify(chat_to_text_completion(payload) if legacy else payload)
    events = []
    for chunk in debug_chat_chunks(display_name, content, include_usage=include_usage):
        out = chat_chunk_to_text_completion(chunk) if legacy else chunk
        if out is not None:
            events.append(sse(out))
    events.append(SSE_DONE)
    return Response("".join(events), mimetype="text/event-stream", headers=_SSE_HEADERS)


def _sse_response(req: UpstreamRequest, make_chunks: ChunkIterFactory, transform: ChunkTransform | None) -> Response:
    """Stream upstream chunks to the client as server-sent events.

    The upstream call is opened and its first chunk read *before* the response starts, so
    early failures (HTTP errors, an immediate SSE error event) still get a real HTTP status.
    Later failures are sent as an SSE `error` event without the `[DONE]` terminator.
    """
    stack = ExitStack()
    try:
        raw_iter = iter(stack.enter_context(open_chat_completion(req)))
        first = next(raw_iter, _END)
    except BaseException:
        stack.close()
        raise
    source: Iterable[Any] = raw_iter if first is _END else itertools.chain([first], raw_iter)
    sink = UsageSink()

    def generate() -> Iterator[str]:
        try:
            for chunk in make_chunks(source, sink):
                out = transform(chunk) if transform else chunk
                if out is not None:
                    yield sse(out)
            yield SSE_DONE
            _log_usage(req.display_name, sink.usage)
        except Exception as exc:  # noqa: BLE001
            err = to_openai_error(exc)
            state.logger.error("Streaming error (model=%s): %s", req.display_name, err.message, exc_info=exc)
            yield sse(err.body())
        finally:
            stack.close()

    response = Response(stream_with_context(generate()), mimetype="text/event-stream", headers=_SSE_HEADERS)
    response.call_on_close(stack.close)
    return response


def _complete(  # noqa: PLR0913 — keyword-only mode flags
    client_params: dict[str, Any],
    model_id: str,
    messages: list[dict[str, Any]],
    *,
    streaming: bool,
    legacy: bool,
    debug: bool,
) -> Response:
    """Run a chat completion upstream and render it as chat (or legacy text) completion."""
    if not is_model_allowed(model_id):
        msg = f"The model '{model_id}' does not exist"
        raise OpenAIHTTPError(_HTTP_NOT_FOUND, msg, param="model", code="model_not_found")
    resolved = resolve_model(model_id, get_client_ip())
    merged = merge_client_params(client_params, resolved.openai_params, model_id, streaming=streaming)

    if debug:
        content = build_debug_text(
            model_id=model_id,
            display_name=model_id,
            messages=messages,
            resolved=resolved,
            extra_body=merged.extra_body,
        )
        return _debug_response(
            model_id, content, streaming=streaming, include_usage=merged.client_wants_usage, legacy=legacy
        )

    req = build_upstream_request(
        display_name=model_id,
        resolved=resolved,
        messages=prepare_messages(messages, resolved.adapter_params, model_id, warn_on_replace=True),
        stream=streaming,
        extra_body=merged.extra_body,
    )
    remove_tags = bool(resolved.adapter_params.get("remove_thinking_tags", False))

    if not streaming:
        with open_chat_completion(req) as upstream_response:
            payload = chat_completion_payload(
                upstream_response, display_name=model_id, model_id=model_id, remove_tags=remove_tags
            )
        _log_usage(model_id, payload.get("usage"))
        return jsonify(chat_to_text_completion(payload) if legacy else payload)

    def make_chunks(stream: Iterable[Any], sink: UsageSink) -> Iterator[dict[str, Any]]:
        return iter_chat_chunks(
            stream,
            display_name=model_id,
            model_id=model_id,
            remove_tags=remove_tags,
            emit_usage=merged.client_wants_usage,
            sink=sink,
        )

    return _sse_response(req, make_chunks, chat_chunk_to_text_completion if legacy else None)


@bp.route("/v1/chat/completions", methods=["POST"])
@log_endpoint
@_openai_errors
def chat_completions() -> Response:
    """OpenAI Chat Completions (streaming and non-streaming)."""
    body = _json_body()
    model_id = _require_model(body)
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        msg = "'messages' must be a non-empty array"
        raise _bad_request(msg, param="messages")
    streaming = _parse_stream(body)

    text = last_user_text(messages, include_parts=True)
    debug = text is not None and is_debug_trigger(text)
    return _complete(body, model_id, messages, streaming=streaming, legacy=False, debug=debug)


@bp.route("/v1/completions", methods=["POST"])
@log_endpoint
@_openai_errors
def completions() -> Response:
    """Legacy OpenAI Completions, implemented on top of chat completions."""
    body = _json_body()
    model_id = _require_model(body)
    streaming = _parse_stream(body)
    prompt = _parse_prompt(body.get("prompt"))
    _reject_legacy_only_params(body)

    client_params = {k: v for k, v in body.items() if k not in _LEGACY_ONLY_PARAMS}
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    return _complete(
        client_params, model_id, messages, streaming=streaming, legacy=True, debug=is_debug_trigger(prompt)
    )
