"""Tests for ollama_adapter.openai_translate."""

import json

import pytest

from ollama_adapter.openai_translate import (
    SSE_DONE,
    UsageSink,
    as_dict,
    chat_chunk_to_text_completion,
    chat_completion_payload,
    chat_to_text_completion,
    debug_chat_chunks,
    debug_chat_completion,
    iter_chat_chunks,
    sse,
)

from .conftest import make_sdk_chunk, make_sdk_completion

BASE = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "up/model"}
USAGE = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


def chunk(delta=None, *, finish=None, index=0, usage=None, choices=True, **extra):
    data = {**BASE, **extra}
    data["choices"] = (
        [{"index": index, "delta": delta or {}, "logprobs": None, "finish_reason": finish}] if choices else []
    )
    data["usage"] = usage
    return make_sdk_chunk(data)


def run(chunks, *, remove_tags=False, emit_usage=False):
    sink = UsageSink()
    out = list(
        iter_chat_chunks(
            iter(chunks), display_name="Client", model_id="m", remove_tags=remove_tags, emit_usage=emit_usage, sink=sink
        )
    )
    return out, sink


def text_of(out, index=0):
    return "".join((c["delta"].get("content") or "") for o in out for c in o["choices"] if c["index"] == index)


class TestPassthrough:
    def test_sdk_roundtrip_keeps_extras_and_nulls(self):
        data = {
            **BASE,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": None,
                        "reasoning_content": "hmm",
                        "tool_calls": [
                            {"index": 0, "id": "t1", "type": "function", "function": {"name": "f", "arguments": "{"}}
                        ],
                    },
                    "logprobs": None,
                    "finish_reason": None,
                    "provider_extra": {"a": 1},
                }
            ],
            "usage": None,
        }
        assert as_dict(make_sdk_chunk(data)) == data

    def test_model_rewritten_and_tool_calls_preserved(self):
        tool_delta = {
            "tool_calls": [{"index": 0, "id": "t", "type": "function", "function": {"name": "f", "arguments": ""}}]
        }
        out, _ = run([chunk(tool_delta), chunk(finish="tool_calls")])
        assert [o["model"] for o in out] == ["Client", "Client"]
        assert out[0]["choices"][0]["delta"]["tool_calls"] == tool_delta["tool_calls"]

    def test_usage_emitted_only_when_requested(self):
        chunks = [chunk({"content": "hi"}), chunk(choices=False, usage=USAGE)]
        out, sink = run(chunks)
        assert len(out) == 1
        assert "usage" not in out[0]
        assert sink.usage == USAGE

        out, _ = run(chunks, emit_usage=True)
        assert out[-1]["usage"] == USAGE
        assert out[0]["usage"] is None

    def test_chat_completion_payload(self):
        resp = make_sdk_completion(
            {
                "id": "x",
                "object": "chat.completion",
                "created": 1,
                "model": "up",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "<think>r</think>Hi", "reasoning_content": "r"},
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                ],
                "usage": USAGE,
            }
        )
        payload = chat_completion_payload(resp, display_name="Client", model_id="m", remove_tags=True)
        assert payload["model"] == "Client"
        assert payload["choices"][0]["message"] == {"role": "assistant", "content": "Hi", "reasoning_content": "r"}
        untouched = chat_completion_payload(resp, display_name="Client", model_id="m", remove_tags=False)
        assert untouched["choices"][0]["message"]["content"] == "<think>r</think>Hi"


class TestThinkingStripping:
    def test_buffered_chunks_dropped_and_text_kept(self):
        chunks = [
            chunk({"role": "assistant", "content": ""}),
            chunk({"content": "<think>"}),
            chunk({"content": "reason"}),
            chunk({"content": "</think>Answer"}),
            chunk(finish="stop"),
        ]
        out, _ = run(chunks, remove_tags=True)
        assert text_of(out) == "Answer"
        assert out[0]["choices"][0]["delta"]["role"] == "assistant"
        assert out[-1]["choices"][0]["finish_reason"] == "stop"
        assert len(out) == 3

    def test_tool_call_chunk_kept_while_content_buffered(self):
        tool = {"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}
        out, _ = run([chunk({"content": "<think>x", **tool})], remove_tags=True)
        assert out[0]["choices"][0]["delta"]["tool_calls"] == tool["tool_calls"]

    def test_unclosed_tail_goes_into_finish_chunk(self):
        out, _ = run([chunk({"content": "<thi"}), chunk(finish="length")], remove_tags=True)
        assert out[-1]["choices"][0] == {
            "index": 0,
            "delta": {"content": "<thi"},
            "logprobs": None,
            "finish_reason": "length",
        }

    def test_tail_flushed_before_usage_chunk(self):
        chunks = [chunk({"content": "<thi"}), chunk(choices=False, usage=USAGE)]
        out, _ = run(chunks, remove_tags=True, emit_usage=True)
        assert out[0]["choices"][0]["delta"]["content"] == "<thi"
        assert out[1]["usage"] == USAGE

    def test_tail_flushed_at_end_of_stream(self):
        out, _ = run([chunk({"content": "<think>never closed"})], remove_tags=True)
        assert text_of(out) == "never closed"

    def test_choiceless_chunk_mid_thought_does_not_end_detection(self):
        chunks = [
            chunk({"content": "<think>secret reasoning"}),
            chunk(choices=False, prompt_filter_results=[{"index": 0}]),
            chunk({"content": "</think>Answer"}),
            chunk(finish="stop"),
        ]
        out, _ = run(chunks, remove_tags=True)
        assert text_of(out) == "Answer"
        assert any("prompt_filter_results" in o for o in out)

    def test_usage_chunk_held_until_stream_end(self):
        chunks = [
            chunk({"content": "<think>x"}),
            chunk(choices=False, usage=USAGE),
            chunk({"content": "</think>late"}),
            chunk(finish="stop"),
        ]
        out, sink = run(chunks, remove_tags=True, emit_usage=True)
        assert text_of(out) == "late"
        assert out[-1]["usage"] == USAGE
        assert sink.usage == USAGE

    def test_filtered_content_becomes_null_on_kept_choice(self):
        out, _ = run([chunk({"role": "assistant", "content": "<think>x"})], remove_tags=True)
        assert out[0]["choices"][0]["delta"] == {"role": "assistant", "content": None}

    def test_n2_interleaved(self):
        chunks = [
            chunk({"content": "<think>a"}, index=0),
            chunk({"content": "B1"}, index=1),
            chunk({"content": "</think>A"}, index=0),
            chunk({"content": "<think>not"}, index=1),
            chunk(finish="stop", index=0),
            chunk(finish="stop", index=1),
        ]
        out, _ = run(chunks, remove_tags=True)
        assert text_of(out, 0) == "A"
        assert text_of(out, 1) == "B1<think>not"


class TestLegacy:
    def test_chat_to_text_completion(self):
        payload = {
            "id": "x",
            "created": 1,
            "model": "M",
            "system_fingerprint": "fp",
            "choices": [{"index": 0, "message": {"content": None}, "finish_reason": "tool_calls"}],
            "usage": USAGE,
        }
        assert chat_to_text_completion(payload) == {
            "id": "x",
            "object": "text_completion",
            "created": 1,
            "model": "M",
            "system_fingerprint": "fp",
            "choices": [{"text": "", "index": 0, "logprobs": None, "finish_reason": "stop"}],
            "usage": USAGE,
        }

    @pytest.mark.parametrize(
        ("choices", "usage", "expected_choices"),
        [
            ([{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}], None, None),
            (
                [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
                None,
                [{"text": "hi", "index": 0, "logprobs": None, "finish_reason": None}],
            ),
            (
                [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                None,
                [{"text": "", "index": 0, "logprobs": None, "finish_reason": "stop"}],
            ),
            ([], USAGE, []),
        ],
    )
    def test_chunk_mapping(self, choices, usage, expected_choices):
        result = chat_chunk_to_text_completion({**BASE, "choices": choices, "usage": usage})
        if expected_choices is None:
            assert result is None
        else:
            assert result["object"] == "text_completion"
            assert result["choices"] == expected_choices


class TestFramingAndDebug:
    def test_sse_framing(self):
        assert sse({"a": "é"}) == 'data: {"a":"é"}\n\n'
        assert SSE_DONE == "data: [DONE]\n\n"

    def test_debug_completion(self):
        payload = debug_chat_completion("M", "text")
        assert payload["object"] == "chat.completion"
        assert payload["choices"][0]["message"] == {"role": "assistant", "content": "text"}
        assert payload["id"].startswith("chatcmpl-debug-")

    def test_debug_chunks(self):
        chunks = debug_chat_chunks("M", "text", include_usage=True)
        assert [c["choices"] and c["choices"][0]["finish_reason"] for c in chunks] == [None, "stop", []]
        assert chunks[-1]["usage"]["total_tokens"] == 0
        assert json.loads(sse(chunks[0])[len("data: ") :])["choices"][0]["delta"]["content"] == "text"
        assert all("usage" not in c for c in debug_chat_chunks("M", "t", include_usage=False))
