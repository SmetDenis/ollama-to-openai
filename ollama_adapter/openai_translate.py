"""Pure payload/chunk transforms and SSE framing for the OpenAI-compatible `/v1` endpoints.

Upstream SDK objects are converted back to plain dicts with `to_dict(mode="json")`, which
round-trips every field the provider sent (including `tool_calls`, `reasoning_content`,
`logprobs`, provider-specific extras and explicit nulls). Only `model` is rewritten to the
client-facing name, and assistant text is optionally stripped of a leading thinking block.
"""

import copy
import json
import time
import uuid
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, cast

from ollama_adapter.thinking import ThinkingTagFilter, remove_thinking_tags

SSE_DONE = "data: [DONE]\n\n"
_CHUNK_OBJECT = "chat.completion.chunk"
_CHUNK_TEMPLATE_KEYS = ("id", "object", "created", "model", "system_fingerprint", "service_tier")
_TOOL_FINISH_REASONS = frozenset({"tool_calls", "function_call"})


@dataclass
class UsageSink:
    """Receives the last `usage` object seen in a stream (for adapter-side logging)."""

    usage: dict[str, Any] | None = None


def as_dict(obj: Any) -> dict[str, Any]:
    """Convert an OpenAI SDK model (or mapping) into an independent plain dict."""
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return cast("dict[str, Any]", to_dict(mode="json", warnings=False))
    if isinstance(obj, Mapping):
        return copy.deepcopy(dict(obj))
    msg = f"Cannot convert {type(obj).__name__} to a dict"
    raise TypeError(msg)


def sse(data: Mapping[str, Any]) -> str:
    """Frame a JSON payload as one server-sent event."""
    return f"data: {json.dumps(data, separators=(',', ':'), ensure_ascii=False)}\n\n"


def chat_completion_payload(resp: Any, *, display_name: str, model_id: str, remove_tags: bool) -> dict[str, Any]:
    """Return the client-facing `chat.completion` payload for an upstream response."""
    payload = as_dict(resp)
    payload["model"] = display_name
    if remove_tags:
        for choice in payload.get("choices") or []:
            message = choice.get("message") if isinstance(choice, dict) else None
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                message["content"] = remove_thinking_tags(message["content"], model_id, remove_enabled=True)
    return payload


def _synthetic_chunk(template: Mapping[str, Any], index: int, text: str) -> dict[str, Any]:
    chunk = {k: template[k] for k in _CHUNK_TEMPLATE_KEYS if k in template}
    chunk.setdefault("object", _CHUNK_OBJECT)
    chunk["choices"] = [{"index": index, "delta": {"content": text}, "logprobs": None, "finish_reason": None}]
    return chunk


def _choice_carries_nothing(choice: Mapping[str, Any]) -> bool:
    delta = choice.get("delta") or {}
    other_delta = any(v is not None for k, v in delta.items() if k != "content")
    return (
        not delta.get("content")
        and not other_delta
        and choice.get("finish_reason") is None
        and choice.get("logprobs") is None
    )


class _ChunkTagStripper:
    """Applies one `ThinkingTagFilter` per choice index across a chunk stream."""

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self._filters: dict[int, ThinkingTagFilter] = {}
        self._finished: set[int] = set()

    def strip(self, chunk: dict[str, Any]) -> bool:
        """Filter the chunk's choices in place; return False when nothing is left to send."""
        kept: list[Any] = []
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                kept.append(choice)
                continue
            index = choice.get("index", 0)
            if index in self._finished:
                kept.append(choice)
                continue
            tag_filter = self._filters.setdefault(index, ThinkingTagFilter(self._model_id))
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                delta = {}
                choice["delta"] = delta
            content = delta.get("content")
            filtered = False
            if isinstance(content, str) and content:
                delta["content"] = tag_filter.feed(content) or None
                filtered = delta["content"] is None
            if choice.get("finish_reason") is not None:
                self._finished.add(index)
                if tail := tag_filter.flush():
                    delta["content"] = (delta.get("content") or "") + tail
                    filtered = False
            if filtered and _choice_carries_nothing(choice):
                continue
            kept.append(choice)
        chunk["choices"] = kept
        return bool(kept)

    def flush(self, template: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
        """Emit buffered text of choices that never sent a `finish_reason`."""
        for index, tag_filter in self._filters.items():
            if index in self._finished:
                continue
            self._finished.add(index)
            if tail := tag_filter.flush():
                yield _synthetic_chunk(template, index, tail)


def iter_chat_chunks(  # noqa: PLR0913 — keyword-only stream options
    stream: Iterable[Any],
    *,
    display_name: str,
    model_id: str,
    remove_tags: bool,
    emit_usage: bool,
    sink: UsageSink,
) -> Iterator[dict[str, Any]]:
    """Translate upstream `chat.completion.chunk` objects into client-facing chunk dicts.

    Rewrites `model`, records usage into `sink`, forwards the usage chunk only when
    `emit_usage` is set, and (with `remove_tags`) strips a leading thinking block per
    choice. Usage-only chunks are held back until the upstream stream ends, so buffered
    thinking text is emitted before them; other choice-less chunks pass through and never
    end tag detection early.
    """
    stripper = _ChunkTagStripper(model_id) if remove_tags else None
    template: dict[str, Any] = {"object": _CHUNK_OBJECT, "model": display_name}
    pending_usage_chunks: list[dict[str, Any]] = []

    for raw in stream:
        chunk = as_dict(raw)
        chunk["model"] = display_name
        template = chunk
        usage = chunk.get("usage")
        if usage:
            sink.usage = usage
        if not emit_usage:
            chunk.pop("usage", None)

        if not chunk.get("choices"):
            if usage:
                if emit_usage:
                    pending_usage_chunks.append(chunk)
                continue
            yield chunk
            continue

        if stripper and not stripper.strip(chunk) and not chunk.get("usage"):
            continue
        yield chunk

    if stripper:
        yield from stripper.flush(template)
    yield from pending_usage_chunks


def _map_finish_reason(reason: Any) -> Any:
    return "stop" if reason in _TOOL_FINISH_REASONS else reason


def chat_to_text_completion(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Repackage a `chat.completion` payload as a legacy `text_completion` payload."""
    result = _text_completion_header(payload)
    result["choices"] = [
        {
            "text": ((choice.get("message") or {}).get("content") or ""),
            "index": choice.get("index", 0),
            "logprobs": None,
            "finish_reason": _map_finish_reason(choice.get("finish_reason")),
        }
        for choice in payload.get("choices") or []
        if isinstance(choice, dict)
    ]
    result["usage"] = payload.get("usage")
    return result


def chat_chunk_to_text_completion(chunk: Mapping[str, Any]) -> dict[str, Any] | None:
    """Repackage a chat chunk as a legacy `text_completion` chunk; None when nothing remains."""
    choices = []
    for choice in chunk.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        text = (choice.get("delta") or {}).get("content")
        finish_reason = choice.get("finish_reason")
        if not text and finish_reason is None:
            continue
        choices.append(
            {
                "text": text or "",
                "index": choice.get("index", 0),
                "logprobs": None,
                "finish_reason": _map_finish_reason(finish_reason),
            }
        )
    if not choices and not chunk.get("usage"):
        return None
    result = _text_completion_header(chunk)
    result["choices"] = choices
    if "usage" in chunk:
        result["usage"] = chunk["usage"]
    return result


def _text_completion_header(source: Mapping[str, Any]) -> dict[str, Any]:
    header: dict[str, Any] = {
        "id": source.get("id"),
        "object": "text_completion",
        "created": source.get("created"),
        "model": source.get("model"),
    }
    if source.get("system_fingerprint") is not None:
        header["system_fingerprint"] = source["system_fingerprint"]
    return header


def _debug_id() -> str:
    return f"chatcmpl-debug-{uuid.uuid4().hex[:24]}"


_ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def debug_chat_completion(display_name: str, content: str) -> dict[str, Any]:
    """Build a `chat.completion` payload carrying the debug output as assistant content."""
    return {
        "id": _debug_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": display_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": dict(_ZERO_USAGE),
    }


def debug_chat_chunks(display_name: str, content: str, *, include_usage: bool) -> list[dict[str, Any]]:
    """Build the streamed chunk sequence carrying the debug output."""
    base = {"id": _debug_id(), "object": _CHUNK_OBJECT, "created": int(time.time()), "model": display_name}
    delta_choice = {
        "index": 0,
        "delta": {"role": "assistant", "content": content},
        "logprobs": None,
        "finish_reason": None,
    }
    finish_choice = {"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}
    chunks = [{**base, "choices": [delta_choice]}, {**base, "choices": [finish_choice]}]
    if include_usage:
        for chunk in chunks:
            chunk["usage"] = None
        chunks.append({**base, "choices": [], "usage": dict(_ZERO_USAGE)})
    return chunks
