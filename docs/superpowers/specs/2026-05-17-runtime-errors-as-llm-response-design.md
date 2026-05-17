# Spec: Runtime errors as LLM-style responses

Date: 2026-05-17

## Problem

When the upstream LLM provider (LiteLLM proxy, OpenAI, Anthropic, Google) fails — rate limit, auth error, timeout, 5xx, connection drop, "no choices" — the proxy returns:

* `/api/chat` (non-streaming): `HTTP 500 {"error": "..."}`
* `/api/chat` (streaming): a single NDJSON line `{"error": "..."}` with **no `done: true`**
* `/api/generate`: same as `/api/chat`

Raycast uses the Ollama native REST API and expects an assistant message (or a streamed sequence terminated by `done: true`). When it sees `HTTP 500` or a half-stream, the user gets either nothing or a generic "request failed" pop-up — they never see *why*.

The proxy already has a working pattern for prompt-render failures (`PromptRenderError → "[PROMPT ERROR] …"` assistant message, `HTTP 200`). Extend the same pattern to every runtime error so the user always sees a readable explanation inside the Raycast chat.

## Scope

Endpoints that **MUST** translate errors into LLM-style responses:

* `POST /api/chat` (streaming + non-streaming)
* `POST /api/generate` (streaming + non-streaming)

Endpoints that **MUST NOT** change:

* `GET/POST /api/tags` — list endpoint, HTTP-status semantics matter
* `POST /api/show` — metadata
* `POST /api/embed` — returns a vector, no place for a textual error
* `GET /health` — must keep 503 for monitoring
* `GET /api/version`, `GET /api/ps`, `GET /` — trivial

## Design

### Output format

Mirror the existing `[PROMPT ERROR]` contract:

* Non-streaming response — `HTTP 200`, JSON body equal to `create_final_response(…)` plus either
  `message = {"role": "assistant", "content": "<error-text>"}` (chat) or `response = "<error-text>"` (generate).
* Streaming response — `HTTP 200`, `application/x-ndjson`, two chunks:
  1. `make_chunk(display_name, "<error-text>")` — the visible chunk
  2. `create_final_response(…)` with `done: true` and empty `content`/`response`

The error text is composed as `f"{prefix} {category}: {message}"`, e.g.
`"[LLM ERROR] Rate limit: 429 You exceeded your current quota"`.

### Categorisation

Translate exceptions into short, opaque category names so Raycast users can recognise the class of failure without reading vendor jargon. Categories map by `isinstance` checks against the OpenAI SDK exception tree (with a fallback for generic `Exception`):

| Exception class | Category | Notes |
| --- | --- | --- |
| `openai.RateLimitError` | `Rate limit` | 429 |
| `openai.AuthenticationError` | `Auth` | 401 — LiteLLM key wrong |
| `openai.PermissionDeniedError` | `Permission denied` | 403 |
| `openai.NotFoundError` | `Not found` | 404 — model missing |
| `openai.BadRequestError` | `Bad request` | 400 — payload rejected |
| `openai.UnprocessableEntityError` | `Unprocessable` | 422 |
| `openai.ConflictError` | `Conflict` | 409 |
| `openai.APITimeoutError` | `Timeout` | network |
| `openai.APIConnectionError` | `Connection` | network |
| `openai.InternalServerError` | `Upstream 5xx` | 5xx from LiteLLM/provider |
| `openai.APIError` (other) | `API` | base class catch-all |
| anything else | `Unexpected` | bug in proxy code |

`message` is the cleaned `str(exc)` (or the class name when the exception carries no message).

### Configuration

New optional top-level section `error_handling` in `config.yml`:

```yaml
error_handling:
  enabled: true            # if false, restore the legacy HTTP 500 / inline error behaviour
  show_details: true       # include the exception message (false → only category)
  include_type: true       # include the category in the rendered text (false → only details)
  prefix: "[LLM ERROR]"    # visible prefix in the assistant message
```

Defaults (when section absent or `enabled` unset): `enabled=true`, `show_details=true`, `include_type=true`, `prefix="[LLM ERROR]"`.

Validation in `config.py` rejects non-bool flags and non-string `prefix`.

### Code changes

1. New module `ollama_adapter/error_formatter.py`
   * `categorize_error(exc) -> ErrorPresentation`
   * `format_error_text(exc) -> str`
   * `is_enabled() -> bool`
   * Reads `state.CONFIG["error_handling"]`, falls back to defaults.

2. `ollama_adapter/routes.py`
   * Add helpers `_build_runtime_error_payload`, `_yield_runtime_error_chunks` (mirror the existing `_build_prompt_error_payload`, `_yield_prompt_error_chunks`).
   * In `_call_openai_streaming` replace the `except Exception` body — yield ndjson chunks instead of `{"error": ...}`.
   * In `_call_openai_non_streaming` replace the `no choices` early return with the new payload.
   * In `chat()` and `generate()` outer-except — convert exception to a `HTTP 200` Ollama response when `error_handling.enabled`.
   * Keep `HTTP 500` paths reachable when `error_handling.enabled == false` (backward compat).

3. `ollama_adapter/config.py`
   * Add `_validate_error_handling(...)` and call it from `load_config(...)`.

4. `ollama_adapter/app.py` — **no change**. The blanket `@app.errorhandler(Exception)` stays at HTTP 500. Our per-endpoint handler runs first; the catch-all only fires for failures unrelated to LLM content.

### Logging

`state.logger.exception(...)` calls remain unchanged — full stack trace still ends up in container logs. Only the client-facing surface changes.

## Tests

### Unit tests in `ollama_adapter` repo

* `tests/test_error_formatter.py` (new):
  * Categorisation table for every supported `openai.*` exception class.
  * `format_error_text` with each combination of `show_details` / `include_type`.
  * `is_enabled` for missing / `enabled=true` / `enabled=false` configs.

* `tests/test_routes.py` (update):
  * `test_api_error_returns_assistant_message` — replaces `test_api_error_500`. Asserts `HTTP 200`, prefix, `done: true`.
  * `test_no_choices_returns_assistant_message` — replaces `test_no_choices_500`.
  * `test_streaming_error_emits_done_chunk` — replaces `test_streaming_error_logged`. Asserts last chunk has `done: true` and at least one chunk contains the prefix.
  * `test_legacy_disabled_returns_500` — new. With `error_handling: {enabled: false}` the proxy still produces the old `HTTP 500 {"error": ...}` body.
  * Same trio for `/api/generate`.

### Integration test in deployment repo

* `tests/test_chat_errors.py` (new in `/Volumes/docker/ollama-openai/tests/`):
  * Skipped by default unless `OLLAMA_TEST_ERROR_MODEL` env var is set; the value is a model name that is **known** to fail through the deployed proxy (e.g. an entry that points at a nonexistent backend).
  * Sends a streaming `/api/chat` request and asserts the NDJSON sequence ends with `done: true` and that some chunk contains the configured prefix.

## Risks / non-goals

* The proxy still emits 500s for non-LLM endpoints (`/api/tags`, `/health`, …). That is intentional: monitoring depends on it.
* If the LiteLLM proxy returns malformed JSON inside an exception message (rare), the assistant content may look ugly. Cosmetic; not a blocker.
* Raycast users will see `[LLM ERROR] …` instead of "request failed". This is the explicit goal.
* `error_handling.enabled: false` is supported for one release as an escape hatch.
