# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ollama-to-OpenAI adapter — a Python service (Flask) that translates Ollama API requests into OpenAI API calls. Allows Ollama clients to transparently use OpenAI models (and compatible providers via LiteLLM). It also exposes an OpenAI-compatible API under `/v1` on the same port (IDE agents, chat UIs, OpenAI SDK scripts) with the same adapter features. Design decisions and rejected options: `docs/superpowers/specs/2026-09-15-openai-compatible-endpoints-design.md`.

## Architecture

Modular structure in the `ollama_adapter/` package. Dependencies: Flask, OpenAI SDK, PyYAML, Jinja2.

### Module Structure

```
ollama_adapter/
  __init__.py          # Empty
  __main__.py          # Entrypoint: create_app + app.run
  state.py             # Global state: CONFIG, client, CACHED_MODELS, jinja_env
  config.py            # load_config(), init_state(), hot-reload, builds jinja_env
  logging_utils.py     # TraceContextFilter, validation, log_request/response, @log_endpoint
  tracing.py           # LiteLLM headers, build_trace_*, capture/log headers
  thinking.py          # remove_thinking_tags(), ThinkingTagFilter (streaming feed/flush), process_stream()
  prompt_renderer.py   # init_jinja_env(), render_file(), render_inline(), PromptRenderError
  error_formatter.py   # categorize_error(), format_error_text() — runtime errors → LLM-style text
  debug_prompt.py      # debug keyword detection + compiled-prompt output (markers)
  models.py            # Models, prompts (system_prompt_mode), IP-routing, model config
  completion.py        # Shared upstream pipeline: model resolution, param merge, prompts, headers, upstream call
  routes.py            # Ollama Flask Blueprint (/api/*, /health, /)
  openai_routes.py     # OpenAI-compatible Blueprint (/v1/*): auth guard, views, SSE streaming
  openai_translate.py  # /v1 pure transforms: SDK objects -> dicts, chunk filtering, text_completion, SSE framing
  openai_errors.py     # /v1 exception -> HTTP status + OpenAI error body
  app.py               # create_app() factory, before_request hooks, /v1-aware error handler
```

### Import Graph

`state.py` is a leaf node (imports nothing from the package). All modules import `state`. `prompt_renderer.py` is a leaf (imports only `jinja2`, no project modules). `models.py` imports `prompt_renderer`. `debug_prompt.py` is a near-leaf: imports `state`, `models`, and `prompt_renderer`. `completion.py` imports `models`, `tracing`, `debug_prompt`. `openai_errors.py` imports `prompt_renderer`; `openai_translate.py` imports `thinking`. `routes.py` imports `completion`, `logging_utils`, `tracing`, `thinking`, `models`, `prompt_renderer`, `debug_prompt`, `error_formatter`. `openai_routes.py` imports `completion`, `openai_errors`, `openai_translate`, `logging_utils`, `models`, `debug_prompt`. `config.py` imports `logging_utils` (TraceContextFilter), `prompt_renderer` (init_jinja_env), and `models` (`SYSTEM_PROMPT_MODES`; `get_and_cache_models` on reload). `app.py` imports both blueprints and `openai_errors`. No circular dependencies.

### Request Flow

1. `@app.before_request` (`app.py`) — checks `config.yml` mtime; on change, reloads config, recreates the OpenAI client, and refreshes the model cache
2. `@log_endpoint` decorator (`logging_utils.py`) — logs request/response and measures duration
3. Endpoint handler — `routes.py` (Ollama) or `openai_routes.py` (`/v1`, preceded by the app-wide `_guard` hook registered via `bp.before_app_request` and limited to `/v1` paths: `openai_api.enabled` and Bearer `api_keys`, section read once; runs for unmatched `/v1` URLs too) — validates input, then goes through `completion.py`: `resolve_model` → (`/v1` only: `merge_client_params`) → `prepare_messages` → `build_upstream_request` → `open_chat_completion`; the response is formatted as Ollama ndjson/JSON or OpenAI JSON/SSE
4. If tracing is enabled — `@app.before_request` generates `request_id`/`trace_id`, which are injected into logs via `TraceContextFilter`

### Key Functions

- **`get_model_config(model_id, client_ip)`** (`models.py`) — core configuration function. Returns the tuple `(openai_params, adapter_params, headers)`
- **`resolve_model_name(client_name)`** (`models.py`) — resolves custom_name to the original OpenAI model ID
- **`is_model_allowed(model_id)`** (`models.py`) — allowlist check against a non-empty `models` list (custom or original name); used by `_validate_allowed_model()` (`routes.py`) and `_complete()` (`openai_routes.py`) before any other work, including `debug`
- **`get_display_name(original_name)`** (`models.py`) — reverse mapping for client responses
- **`apply_ip_routing(model_entry, client_ip)`** (`models.py`) — applies IP-specific overrides; shallow merge for dict fields
- **`get_and_cache_models(*, force_refresh, raise_errors)`** (`models.py`) — fetches models from the OpenAI API, caches in `state.CACHED_MODELS`; failures return `[]` (Ollama semantics) unless `raise_errors=True` (`/v1/models` — keeps the upstream status)
- **`apply_system_prompt(messages, adapter_params, model_id, *, warn_on_replace)`** (`models.py`) — renders and injects the system prompt; propagates `PromptRenderError` to caller. `/v1` passes `warn_on_replace=True`: a WARNING is logged when `replace` mode discards a non-empty client `system`/`developer` message
- **`_resolve_system_prompt(adapter_params, model_id)`** (`models.py`) — picks between `system_prompt_inline` and `system_prompt_file`, renders via `state.jinja_env`; raises `PromptRenderError`
- **`_collect_prompt_vars(adapter_params)`** (`models.py`) — merges built-in date/time vars with global `prompts.vars` and model `prompt_vars` (user vars override built-ins; model overrides global)
- **`_build_datetime_vars()`** (`models.py`) — computes per-request built-in date/time template vars (`now` object, flat parts, human/ISO presets) in `prompts.timezone` (default UTC)
- **`init_jinja_env(base_dir)`** / **`render_file(env, path, vars)`** / **`render_inline(env, text, vars)`** (`prompt_renderer.py`) — Jinja2 sandboxed renderer; all errors wrap into `PromptRenderError`
- **`apply_prompt_caching(messages, adapter_params, model_id)`** (`models.py`) — adds `cache_control` markers for Anthropic/Gemini
- **`categorize_error(exc)`** / **`format_error_text(exc)`** / **`is_enabled()`** (`error_formatter.py`) — translate runtime errors into user-facing assistant content
- **`remove_thinking_tags(content, model_id, remove_enabled)`** (`thinking.py`) — strips `<think>`/`<thinking>` tags
- **`ThinkingTagFilter.feed(text)` / `.flush()`** (`thinking.py`) — protocol-agnostic streaming tag removal (state machine; re-dispatches text sharing a chunk with a tag boundary, iterative close-tag scan); used by `process_stream()` (Ollama ndjson) and `iter_chat_chunks()` (`/v1`, one filter per choice index)
- **`_call_openai_streaming()`** / **`_call_openai_non_streaming()`** (`routes.py`) — Ollama helpers for `chat()` and `generate()` on top of `completion.py`
- **`resolve_model()`** / **`merge_client_params()`** / **`prepare_messages()`** / **`build_upstream_request()`** / **`open_chat_completion()`** / **`build_debug_text()`** (`completion.py`) — shared upstream pipeline. `merge_client_params`: client body minus `model/messages/stream` and `CLIENT_PARAM_DENYLIST` (client-side credentials `api_key`/`api_base`/`base_url`/`api_version`, LiteLLM control keys `custom_llm_provider`/`extra_headers`/`headers`/`litellm_params`, common provider auth params — not exhaustive), config `params` override (shallow), `metadata` merged client < config < trace, streaming forces `stream_options.include_usage` and records `client_wants_usage`. Everything goes via `extra_body` (the SDK merges it over named kwargs, so reserved keys must never be there). `open_chat_completion` is a context manager choosing plain / `with_raw_response` / `with_streaming_response` and always closing the upstream response
- **`iter_chat_chunks()`** / **`chat_completion_payload()`** / **`chat_to_text_completion()`** / **`chat_chunk_to_text_completion()`** (`openai_translate.py`) — `/v1` passthrough via `to_dict(mode="json")` (keeps `tool_calls`, `reasoning_content`, extras, nulls); rewrites only `model`; usage chunk forwarded only if requested and held back until the stream ends; buffered thinking text is emitted before the finish chunk; choice-less non-usage chunks pass through without ending tag detection
- **`to_openai_error(exc)`** / **`error_response(exc)`** (`openai_errors.py`) — `/v1` error mapping (see spec table): `APIStatusError` passthrough with `retry-after`, timeout 504, connection 502, `PromptRenderError` 500 + `x-should-retry: false`, Flask 404/405 → `unknown_url`/`method_not_allowed`
- **`_sse_response()`** (`openai_routes.py`) — opens upstream and reads the first chunk before the response starts (early failures keep a real HTTP status); later failures become an SSE `error` event without `[DONE]`; upstream closed via `ExitStack` in the generator and `call_on_close`
- **`is_debug_trigger(text)`** / **`build_debug_content(messages, adapter_params, model_id, *, config_view, outgoing_view)`** (`debug_prompt.py`) — detect the `debug` keyword (after stripping markup) and render the fully compiled `messages` array as a fenced block with visible include markers, optionally preceded by `model config` / `outgoing request` JSON sections
- **`build_config_view(model_id, openai_params, adapter_params, headers)`** / **`build_outgoing_view(openai_params, extra_body, merged_headers)`** (`debug_prompt.py`) — shape the final combined model config (post IP-routing) and the actual upstream request for the debug sections
- **`_mask_secrets(obj)`** / **`_is_sensitive_key(key)`** (`debug_prompt.py`) — recursively mask values under secret-bearing keys (auth/api-key/token/secret/password/cookie, whole-word match so `max_tokens` is safe) before rendering debug JSON
- **`render_file_debug(env, path, vars)`** / **`render_inline_debug(env, text, vars)`** (`prompt_renderer.py`) — debug renderers that wrap each `{% include %}` boundary in visible markers via `_DebugMarkerLoader`
- **`place_system_message(messages, content, *, mode)`** / **`merge_system_content()`** (`models.py`) — shared system-message placement used by both the normal and debug paths: targets the first `system` **or `developer`** message; `replace` swaps content (role kept), `prepend`/`append` merge (strings joined by a blank line, content-part lists get a text part; other keys kept); without such a message a `system` message is inserted at index 0
- **`_select_prompt_source(adapter_params, model_id)`** (`models.py`) — picks `("file", path)` / `("inline", text)` / `(None, None)`; single choke-point that logs the both-set warning

### Prompts Directory

`prompts/` (configurable via `prompts.base_dir`) — Jinja2 template files referenced by `system_prompt_file` in the model config. All `system_prompt_file` paths and `{% include "..." %}` directives are resolved **relative to this directory**; absolute paths and `..` are rejected by the sandbox. Mounted read-only in Docker. Files are re-rendered on every request — editable without restart.

### Prompt Templating

Templates are rendered via `jinja2.sandbox.SandboxedEnvironment` with `StrictUndefined`. Errors (missing file, undefined variable, syntax error, sandbox violation, include cycle) raise `PromptRenderError`, which `routes.py` translates into an HTTP 200 Ollama response carrying an `assistant` message that starts with `[PROMPT ERROR] ...`. The request is **not** forwarded to OpenAI on render failure.

How the rendered prompt combines with a client system message is controlled by `system_prompt_mode` (`replace` default / `prepend` / `append`, validated in `config.py`, overridable in `ip_routing`). `apply_prompt_caching` marks string content as one cached text part, or the last text part of a content-part list (skipped if a part already has `cache_control`).

Every rendered prompt also receives built-in date/time variables computed per request in `prompts.timezone` (IANA name, default `UTC`) by `_build_datetime_vars()` (`models.py`): a `now` datetime object (`{{ now.strftime('%H:%M') }}`, `{{ now.year }}`, ...), flat parts `year`/`month`/`day`/`hour`/`minute`/`weekday` (month and weekday as hardcoded English names — locale/platform independent), and presets `date_human`/`time_human`/`datetime_human`/`date_iso`/`datetime_iso`. They are the lowest-priority layer, so any same-named `prompts.vars`/`prompt_vars` entry overrides them. Conditionals like `{% if weekday == "Friday" %}...{% endif %}` work via these variables.

### Runtime Error Handling

`/api/chat` and `/api/generate` (both streaming and non-streaming) translate runtime failures from the upstream LLM provider into a successful Ollama-format response with an `assistant` message that starts with the configured prefix (default `[LLM ERROR]`). Supported categories: `Rate limit`, `Auth`, `Permission denied`, `Not found`, `Unprocessable`, `Bad request`, `Conflict`, `Timeout`, `Connection`, `Upstream 5xx`, `API`, `Unexpected`. Logging of the full stack trace via `state.logger.exception(...)` is preserved. Other endpoints (`/api/tags`, `/api/embed`, `/api/show`, `/health`) keep their HTTP-status semantics. Configurable via the optional `error_handling` section (`enabled`, `show_details`, `include_type`, `prefix`); setting `enabled: false` restores legacy HTTP 500 / inline error body.

The OpenAI-compatible `/v1` endpoints never use `error_handling`: they answer with real HTTP status codes and the OpenAI error body `{"error": {message, type, param, code}}` (`openai_errors.py`), including `PromptRenderError` (HTTP 500, type `prompt_render_error`).

Variable priority (low → high): built-in date/time → `prompts.vars` → `model.prompt_vars` → `ip_routing[matched].prompt_vars`. Merge is shallow over top-level keys; `prompt_vars` participates in `apply_ip_routing` alongside `params`/`headers`.

### Debug Prompt Output

If a client's input reduces to the keyword `debug` (markup stripped via `<[^>]*>`, then trimmed and lowercased — this unwraps the Raycast `<user_input>...</user_input>` envelope), `/api/chat` and `/api/generate` short-circuit: instead of calling the upstream model they return an HTTP 200 Ollama response whose assistant content is the fully compiled request, wrapped in a single fenced code block.

The block opens with two JSON sections, then the compiled `messages` array:
- `═══ model config ═══` — the final **combined** model configuration after IP routing and all merges (what was actually used): `requested_model` (name sent by the client), `resolved_model` (the upstream model actually used), `params` (OpenAI params minus the internal `model_id`), `headers`, and `adapter` (remove_thinking_tags / prompt_caching / system_prompt source / prompt_vars).
- `═══ outgoing request ═══` — what actually goes upstream, assembled by the **same** helpers as a real call (`_build_extra_body`, `build_trace_headers`): `model`, `extra_body` (params plus trace `metadata` when tracing is on), and `extra_headers` (configured headers merged with trace headers). `messages` are omitted here — they appear verbatim below.

Both JSON sections pass through `_mask_secrets`: values under secret-bearing keys (auth/api-key/token/secret/password/cookie, matched as whole words so `max_tokens` is not masked) are shown as `abcd…wxyz` (or `****` when short). The masking is display-only and never appears in the real request.

Then each message is a `═══ message[i] role=... ═══` section; the system prompt shows visible include markers (`── BEGIN file: ... ──`, `── include: ... ──`, `── end include: ... ──`, `── END file: ... ──`). Variables are already substituted and IP routing already applied. Render errors are shown inline as a `── PROMPT RENDER ERROR ──` marker. The feature has no config flag (always on) and the keyword `debug` is hardcoded. Tag stripping affects detection only — messages are displayed verbatim (tags included). Detection: `chat` checks the last user message (string content only); `generate` checks the `prompt`. On `/v1`: `/v1/chat/completions` checks the last user message including the `text` parts of a content-part list (`last_user_text(include_parts=True)`), `/v1/completions` checks `prompt`; the output is returned as a `chat.completion` / `text_completion` (or SSE chunks) and the `outgoing request` section includes the merged client params. Long `data:` URLs (inline images) are abbreviated in the messages dump.

## Commands

```bash
# Run locally
python3 -m ollama_adapter
# or via venv
./.venv/bin/python3 -m ollama_adapter

# Docker Compose (port 11345 -> 11434)
docker-compose up -d

# Docker standalone
docker build -t ollama-to-openai .
docker run -p 11434:11434 -v ./config.yml:/app/config.yml:ro ollama-to-openai

# Recreate virtual environment
uv sync
```

## Configuration

`config.yml` file (see `config-example.yml` for a full example with comments). Key sections:

- **`server`**: `host`, `port`
- **`openai`**: `api_key` (required), `base_url` (optional — for LiteLLM, Azure, etc.)
- **`clients`**: named IP address groups for `ip_routing`
- **`logging`**: `log_level`, `log_requests`
- **`tracing`**: LiteLLM proxy integration — request_id/trace_id, headers, tags
- **`prompts`**: `base_dir` (default `./prompts`), `timezone` (IANA name for built-in date/time vars, default `UTC`), and `vars` (global Jinja2 variables)
- **`error_handling`** (optional): `enabled`, `show_details`, `include_type`, `prefix` — controls whether runtime errors in `/api/chat` and `/api/generate` are translated into LLM-style responses (default: enabled, prefix `[LLM ERROR]`)
- **`openai_api`** (optional): `enabled` (default `true`; `false` → every `/v1` path and method 404), `api_keys` (Bearer tokens for `/v1` only; empty/missing → no auth, logged as a warning at load even when the section is absent; constant-time compare)
- **`models`**: model list with a two-level structure:
  - Root level: `name` (required), `custom_name`, `remove_thinking_tags`, `prompt_caching`, `system_prompt_inline`, `system_prompt_file` (mutually exclusive; file wins on conflict; legacy `system_prompt` deprecated), `system_prompt_mode` (`replace`/`prepend`/`append`), `prompt_vars` (overrides global `prompts.vars`)
  - `params`: dict of OpenAI API parameters — passed through without validation
  - `headers`: dict of custom HTTP headers
  - `ip_routing`: list of IP-specific overrides (inheritance + shallow merge; `params`, `headers`, `prompt_vars` are dict-merged)

If `models` is empty — all available OpenAI models are exposed and any model name is forwarded. If populated — only the listed ones (by `name` or `custom_name`, `is_model_allowed()`): unlisted models get HTTP 404 on every model-taking endpoint — `/api/chat`, `/api/generate`, `/api/embed`, `/api/show` (`{"error": "model \"x\" not found"}`) and `/v1/chat/completions`, `/v1/completions` (`model_not_found`). Embedding models must be listed too.

Config hot-reload: on every request the file mtime is checked. On change — config is reloaded, the OpenAI client is recreated, and the model cache is refreshed. On load failure — the current config is preserved.

## API Endpoints

| Endpoint        | Method   | Description                                |
|-----------------|----------|--------------------------------------------|
| `/api/chat`     | POST     | Chat completions (streaming/non-streaming) |
| `/api/generate` | POST     | Text generation (streaming/non-streaming)  |
| `/api/embed`    | POST     | Embeddings                                 |
| `/api/tags`     | GET/POST | List models                                |
| `/api/show`     | POST     | Model information                          |
| `/api/version`  | GET      | Service version                            |
| `/api/ps`       | GET      | Running models (mock)                      |
| `/health`       | GET      | Health check with OpenAI verification      |
| `/`             | GET      | Service information                        |
| `/v1/chat/completions` | POST | OpenAI chat completions (SSE streaming/non-streaming) |
| `/v1/completions` | POST   | Legacy text completions via chat (`echo`/`suffix`/`logprobs`/`best_of>1` → 400) |
| `/v1/models`    | GET      | OpenAI model list                          |
| `/v1/models/<path:id>` | GET | Retrieve one model                      |

Not implemented (by decision): `/v1/responses`, `/v1/embeddings` (possible follow-up).

## Testing

**Important**: Do NOT start the server automatically after changes. The user tests manually.

Automated tests: `tests/test_*.py` via `make check`. `/v1` tests build real SDK objects with `make_sdk_chunk`/`make_sdk_completion`/`make_status_error` and parse SSE with `collect_sse` (`tests/conftest.py`).

Manual tests: `tests/manual-check.http` — HTTP requests for all endpoints (including `/v1`), including error cases. Port 11345 (docker-compose) or 11434 (locally).

## Development Notes

- **Package Manager**: UV with lock file (`uv.lock`)
- **Python**: 3.13+ (`.python-version`)
- **Flask debug mode**: enabled — auto-reload on code changes
- **Version**: 0.1.0
- **CI/CD**: GitHub Actions (`.github/workflows/ci.yml`) — linters + Docker image build to GHCR
- **Linters and tests**: always run via `make` (e.g. `make check`), never call ruff/mypy/pytest directly
- **Final check**: on completing any task, always run `make pre-commit` — it formats code, applies lint fixes, and runs the full check (format → lint-fix → check)
