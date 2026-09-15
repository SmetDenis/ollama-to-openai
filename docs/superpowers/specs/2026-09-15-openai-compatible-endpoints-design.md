# Design: OpenAI-compatible `/v1` endpoints

**Date:** 2026-09-15
**Status:** Approved design, implemented
**Area:** `ollama_adapter/completion.py` (new), `ollama_adapter/openai_errors.py` (new), `ollama_adapter/openai_translate.py` (new), `ollama_adapter/openai_routes.py` (new), `ollama_adapter/thinking.py`, `ollama_adapter/models.py`, `ollama_adapter/config.py`, `ollama_adapter/debug_prompt.py`, `ollama_adapter/routes.py`, `ollama_adapter/app.py`

## Overview

The adapter spoke only the Ollama protocol (`/api/*`). It now also exposes an
OpenAI-compatible API under `/v1` on the same port — the same place real Ollama
serves its OpenAI compatibility layer. Target clients: IDE agents (Cursor,
Continue, Cline — tools/function calling, streamed `tool_calls`), chat UIs (Open
WebUI etc. — image content parts) and scripts on the official OpenAI SDKs
(exact response and error shapes).

Unlike `/api`, `/v1` is a **faithful passthrough**: `tool_calls`,
`reasoning_content`, `logprobs`, `n > 1` and unknown provider fields survive;
client parameters are honoured; errors use real HTTP status codes. All adapter
features still apply: IP routing, `custom_name`, Jinja system prompts, prompt
caching, tracing, `remove_thinking_tags`, the `debug` keyword.

## Goals

- `POST /v1/chat/completions` (streaming SSE and non-streaming).
- `POST /v1/completions` (legacy) implemented on top of chat completions.
- `GET /v1/models`, `GET /v1/models/<id>` (ids may contain `/`).
- Optional Bearer-key auth for `/v1` only; a switch to disable `/v1` entirely.
- `/api/*` behaviour unchanged (existing tests untouched).

## Non-Goals / rejected options

| Option | Decision | Why |
|---|---|---|
| `/v1/responses` (Responses API, Codex CLI) | Rejected | Not needed by the user |
| `/v1/embeddings` | Deferred (follow-up) | Not requested for v1; trivial to add later on top of `/api/embed` logic |
| Assistant-text errors (`[LLM ERROR]`) on `/v1` | Rejected | Breaks SDK retries and agents, which would treat the error as model output. `error_handling` stays Ollama-only |
| Forwarding raw upstream SSE bytes | Rejected | The thinking filter and `model` rewrite need parsed chunks; SDK objects round-trip losslessly via `to_dict(mode="json")` (verified) |
| Legacy completions: batch prompts, token arrays, `echo`, `suffix` (FIM), `logprobs`, `best_of > 1` | Rejected with HTTP 400 `unsupported_parameter` | Cannot be emulated through chat completions. Continue tab-autocomplete (FIM) is therefore not supported |
| Direct upstream `completions.create` passthrough | Rejected | Many LiteLLM-routed models lack legacy completions; system prompt/debug/caching would not apply |
| Storing `created` in the model cache | Rejected | Would change `/api/tags` output; `created` is recovered losslessly from `modified_at` |
| Client-params-win merge | Rejected | Config `params` act as an admin-controlled limiter (e.g. `max_tokens`) |
| Unlisted models passthrough when `models` is non-empty | Replaced after review | Now HTTP 404 on both APIs (see "Review outcome") |
| Remapping upstream 401/403 to 502 | Rejected by user | Passthrough as-is; accepted risk of the client confusing it with its own key |
| Configurable client-param denylist | Deferred (follow-up) | A hardcoded list covers the known LiteLLM credential/routing keys |
| Separate cached content part for the config prompt in `prepend` mode | Deferred (follow-up) | Today the whole merged system block is cached; a changing client system prompt means a cache miss |

## Design

### Module layout

- `completion.py` — shared upstream pipeline used by both protocols: model
  resolution, client-param merge, message preparation (system prompt + caching),
  request assembly with trace headers, the upstream call context manager
  (plain / raw / streaming response with LiteLLM header capture), debug text.
  No Flask response formatting.
- `openai_errors.py` — exception → HTTP status + OpenAI error body.
- `openai_translate.py` — pure payload/chunk transforms and SSE framing.
- `openai_routes.py` — the `/v1` blueprint (auth hook, views, SSE streaming).
- `thinking.py` — the tag-removal state machine is exposed as a reusable
  `ThinkingTagFilter.feed(text) -> str / flush() -> str`; the Ollama stream
  processor is a thin wrapper around it.

### Parameter merge (`/v1`)

1. Client body minus `model`, `messages`, `stream` and a hardcoded denylist
   (`api_key`, `api_base`, `base_url`, `api_version`; dropped with a debug log).
2. Config `params` (minus internal `model_id`) override client keys (shallow).
3. `metadata`: shallow dict merge, precedence client < config < trace metadata.
4. Streaming: `stream_options` = client ∪ config, `include_usage` forced `true`
   (the adapter logs usage); the usage chunk is forwarded to the client only if
   the client itself asked for `include_usage`. Non-streaming: `stream_options`
   is dropped.
5. Everything goes to the SDK as `extra_body` (the SDK merges `extra_body` over
   named kwargs, so reserved keys must never be present there).

### `system_prompt_mode`

New per-model field (also overridable in `ip_routing`, validated at load):
`replace` (default, previous behaviour) | `prepend` | `append`. Applies to both
protocols. The first message with role `system` **or `developer`** is the
target; merge modes keep its role and other keys. String content is joined with
`\n\n`; list-of-parts content gets a text part inserted before/after. With no
client system message the config prompt is inserted at index 0 as `system`.
Prompt caching marks the last text part when system content is a list.

### Errors (`/v1`)

| Source | HTTP | `type` / `code` |
|---|---|---|
| Request validation | 400 | `invalid_request_error` / `null` or `unsupported_parameter` |
| Missing/wrong Bearer key | 401 | `invalid_request_error` / `invalid_api_key` |
| `/v1` disabled (any path and method), unknown `/v1` path | 404 | `invalid_request_error` / `unknown_url` |
| Wrong method | 405 | `invalid_request_error` / `method_not_allowed` (+ `Allow` header) |
| Model not found (`/v1/models/<id>`) | 404 | `invalid_request_error` / `model_not_found` |
| `PromptRenderError` | 500 | `prompt_render_error` (header `x-should-retry: false`) |
| `APITimeoutError` | 504 | `timeout_error` / `upstream_timeout` |
| `APIConnectionError` | 502 | `api_connection_error` / `upstream_connection_error` |
| `APIStatusError` (incl. upstream 401/403) | upstream status | upstream error body, missing keys filled; `retry-after` forwarded |
| Other `APIError` | 502 | upstream body if dict, else `api_error` |
| `/v1/models` upstream failure | same mapping as above (real status) | an empty filtered list is a normal `200 {"data": []}` |
| Anything else | 500 | `server_error` |

Streaming: the upstream call is opened and the first raw chunk is read before
the HTTP response starts, so early failures still get a real status. Usage-only
chunks are held back until the upstream stream ends (after flushing buffered
thinking text); other choice-less chunks pass through without ending tag detection. Failures
after streaming started emit `data: {"error": {...}}` and end the stream without
`[DONE]` (the OpenAI SDK raises `APIError` on such an event).

### Debug keyword on `/v1`

Chat: the last user message — string content, or the `text` parts of a content
list joined by newlines. Completions: `prompt`. The response is a regular
`chat.completion` / `text_completion` (or SSE chunks) whose content is the same
fenced debug block as on `/api`, including the merged client params in the
`outgoing request` section.

### Auth

```yaml
openai_api:
  enabled: true     # false -> every /v1 route returns 404
  api_keys: []      # empty/missing -> /v1 is unauthenticated
```

Checked in an app-wide `before_request` hook limited to `/v1` paths (registered
after the config hot-reload hook), so it also covers unmatched `/v1` URLs and wrong
methods — with keys configured those answer 401 before routing can reveal 404/405.
The `openai_api` section is read once per request (no hot-reload race between
`enabled` and `api_keys`). Constant-time comparison against every key. `/api/*` is
never gated (Ollama clients do not send keys). The section is validated even when
absent, so pre-existing configs log "openai_api is enabled without api_keys".

## Risks

- Upstream 401/403 passthrough may look like the client's own key is wrong.
- Retries stack: the adapter's SDK retries upstream, client SDKs retry our 5xx.
- `debug` exposes the compiled system prompt to anyone who can call `/v1` —
  set `api_keys` when exposed beyond localhost.
- `remove_thinking_tags` may misalign `logprobs` with visible content;
  `reasoning_content` is deliberately untouched.
- Behind nginx SSE needs `X-Accel-Buffering: no` (sent by the adapter).
- No response headers are sent until the upstream's first chunk arrives (price of
  real status codes for early failures). Reasoning models with a slow first chunk
  can hit short proxy/client header timeouts.
- Pre-existing: unknown non-`/v1` routes answer HTTP 500 (kept for `/api` parity).

## Fixed along the way: streaming thinking-tag loss

Extracting `ThinkingTagFilter` exposed a pre-existing bug in the Ollama streaming
state machine: text that shared an upstream chunk with a tag boundary was held in
a buffer that the end-of-stream flush never emitted, and the very first chunk was
never checked for a tag. Reproduced on the old code:
`["<think>", "x</think>Hi"]` → `""` and `["<think>x</think>Hello", "!"]` → `""`.
Root cause fix: after every state transition the remaining text is re-dispatched
immediately, and the flush includes every buffer. Also fixed: whitespace after the
close tag is dropped even when it arrives in later chunks (parity with the
non-streaming regex), a close tag split right after a 1000-char buffer flush is
still detected, and non-closing `</...>` sequences are scanned iteratively (no
recursion depth limit).

## Review outcome (Codex gpt-5.6-sol, Fable 5.1)

Fixed after review: `/v1` disabled now 404s for every method/path; `openai_api`
section read once in the guard; `/v1/models` keeps upstream status codes and
returns an empty list as 200; missing `openai_api` section still warns about
unauthenticated `/v1`; a choice-less chunk mid-stream no longer ends thinking-tag
detection (usage chunks deferred to stream end); `logprobs: false` accepted on
legacy completions; `/v1/models` responses truncated in logs; 405 carries `Allow`.

Product decisions taken after review:

| Question | Decision | Why / consequence |
|---|---|---|
| Unlisted models (non-empty `models`) | HTTP 404 on **both** APIs: `/api/chat`, `/api/generate`, `/api/embed`, `/api/show` (Ollama-style `{"error": "model \"x\" not found"}`) and `/v1/chat/completions`, `/v1/completions` (`model_not_found`) | Closes the bypass of per-model `params` limits. **Breaking for `/api`:** clients calling unlisted models (including embedding models) stop working until the model is listed. Empty `models` keeps passthrough. Rejected: `/v1`-only 404; leaving passthrough |
| Absent `openai_api` section | Enabled without auth + startup/reload warning | Convenience over secure-by-default; rejected: disabled until the section exists |
| Default `system_prompt_mode` | `replace` everywhere; WARNING on `/v1` when it discards a non-empty client system/developer message | One behaviour for both APIs; rejected: `prepend` default on `/v1` only (same model would behave differently per API) |
| LiteLLM control keys from clients | Extend the hardcoded denylist (`custom_llm_provider`, `extra_headers`, `headers`, `litellm_params`, common provider auth params) | LiteLLM docs list `api_key`/`api_base`/`base_url` plus provider-specific auth params (e.g. `vertex_project`) as client-side credentials; list is not exhaustive. Rejected: allowlist of OpenAI params (would silently drop new provider params) |

Still open:
- Limiter key aliasing: config `max_tokens` does not cap a client
  `max_completion_tokens` (both are sent). Documented workaround: set both in `params`.
- Whether LiteLLM keeps `cache_control` when translating `developer` → `system`.

Accepted as-is / low priority: raw request bodies logged when `log_requests: true`;
no `MAX_CONTENT_LENGTH`; per-choice thinking buffers with `n > 1`; `build_extra_body`
(Ollama path) does not strip reserved keys from config `params`.

## Follow-ups

- `/v1/embeddings` (reuse the `/api/embed` logic, OpenAI response shape).
- Cache-split for `prepend`/`append`: send the config prompt as its own cached
  content part so a changing client system prompt does not invalidate the cache.
- Configurable client-parameter denylist (currently hardcoded, not exhaustive).
- `max_tokens` / `max_completion_tokens` aliasing in `merge_client_params`.
- Optional: gate `/api/*` with `openai_api.api_keys`-style keys (needs a separate
  switch; Ollama clients such as Raycast do not send keys).
- Unknown non-`/v1` routes still return HTTP 500 instead of 404 (pre-existing).
