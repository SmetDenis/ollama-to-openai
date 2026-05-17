# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ollama-to-OpenAI adapter — a Python service (Flask) that translates Ollama API requests into OpenAI API calls. Allows Ollama clients to transparently use OpenAI models (and compatible providers via LiteLLM).

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
  thinking.py          # remove_thinking_tags(), _process_stream() — 5-state machine
  prompt_renderer.py   # init_jinja_env(), render_file(), render_inline(), PromptRenderError
  error_formatter.py   # categorize_error(), format_error_text() — runtime errors → LLM-style text
  models.py            # Models, prompts, IP-routing, model config
  routes.py            # Flask Blueprint, all endpoints, shared completion helper
  app.py               # create_app() factory, before_request hooks
```

### Import Graph

`state.py` is a leaf node (imports nothing from the package). All modules import `state`. `prompt_renderer.py` is a leaf (imports only `jinja2`, no project modules). `routes.py` imports `logging_utils`, `tracing`, `thinking`, `models`, `prompt_renderer`. `config.py` imports `logging_utils` (TraceContextFilter), `prompt_renderer` (init_jinja_env), and `models` (get_and_cache_models on reload). `models.py` imports `prompt_renderer`. No circular dependencies.

### Request Flow

1. `@app.before_request` (`app.py`) — checks `config.yml` mtime; on change, reloads config, recreates the OpenAI client, and refreshes the model cache
2. `@log_endpoint` decorator (`logging_utils.py`) — logs request/response and measures duration
3. Endpoint handler (`routes.py`) — validates input, retrieves model config, calls the OpenAI API, formats the response in Ollama format
4. If tracing is enabled — `@app.before_request` generates `request_id`/`trace_id`, which are injected into logs via `TraceContextFilter`

### Key Functions

- **`get_model_config(model_id, client_ip)`** (`models.py`) — core configuration function. Returns the tuple `(openai_params, adapter_params, headers)`
- **`resolve_model_name(client_name)`** (`models.py`) — resolves custom_name to the original OpenAI model ID
- **`get_display_name(original_name)`** (`models.py`) — reverse mapping for client responses
- **`apply_ip_routing(model_entry, client_ip)`** (`models.py`) — applies IP-specific overrides; shallow merge for dict fields
- **`get_and_cache_models()`** (`models.py`) — fetches models from the OpenAI API, caches in `state.CACHED_MODELS`
- **`apply_system_prompt(messages, adapter_params, model_id)`** (`models.py`) — renders and injects the system prompt; propagates `PromptRenderError` to caller
- **`_resolve_system_prompt(adapter_params, model_id)`** (`models.py`) — picks between `system_prompt_inline` and `system_prompt_file`, renders via `state.jinja_env`; raises `PromptRenderError`
- **`_collect_prompt_vars(adapter_params)`** (`models.py`) — merges global `prompts.vars` with model `prompt_vars` (model overrides global)
- **`init_jinja_env(base_dir)`** / **`render_file(env, path, vars)`** / **`render_inline(env, text, vars)`** (`prompt_renderer.py`) — Jinja2 sandboxed renderer; all errors wrap into `PromptRenderError`
- **`apply_prompt_caching(messages, adapter_params, model_id)`** (`models.py`) — adds `cache_control` markers for Anthropic/Gemini
- **`categorize_error(exc)`** / **`format_error_text(exc)`** / **`is_enabled()`** (`error_formatter.py`) — translate runtime errors into user-facing assistant content
- **`remove_thinking_tags(content, model_id, remove_enabled)`** (`thinking.py`) — strips `<think>`/`<thinking>` tags
- **`_process_stream()`** (`thinking.py`) — 5-state machine for streaming tag removal
- **`_call_openai_streaming()`** / **`_call_openai_non_streaming()`** (`routes.py`) — shared helpers for `chat()` and `generate()`

### Prompts Directory

`prompts/` (configurable via `prompts.base_dir`) — Jinja2 template files referenced by `system_prompt_file` in the model config. All `system_prompt_file` paths and `{% include "..." %}` directives are resolved **relative to this directory**; absolute paths and `..` are rejected by the sandbox. Mounted read-only in Docker. Files are re-rendered on every request — editable without restart.

### Prompt Templating

Templates are rendered via `jinja2.sandbox.SandboxedEnvironment` with `StrictUndefined`. Errors (missing file, undefined variable, syntax error, sandbox violation, include cycle) raise `PromptRenderError`, which `routes.py` translates into an HTTP 200 Ollama response carrying an `assistant` message that starts with `[PROMPT ERROR] ...`. The request is **not** forwarded to OpenAI on render failure.

### Runtime Error Handling

`/api/chat` and `/api/generate` (both streaming and non-streaming) translate runtime failures from the upstream LLM provider into a successful Ollama-format response with an `assistant` message that starts with the configured prefix (default `[LLM ERROR]`). Supported categories: `Rate limit`, `Auth`, `Permission denied`, `Not found`, `Unprocessable`, `Bad request`, `Conflict`, `Timeout`, `Connection`, `Upstream 5xx`, `API`, `Unexpected`. Logging of the full stack trace via `state.logger.exception(...)` is preserved. Other endpoints (`/api/tags`, `/api/embed`, `/api/show`, `/health`) keep their HTTP-status semantics. Configurable via the optional `error_handling` section (`enabled`, `show_details`, `include_type`, `prefix`); setting `enabled: false` restores legacy HTTP 500 / inline error body.

Variable priority (low → high): `prompts.vars` → `model.prompt_vars` → `ip_routing[matched].prompt_vars`. Merge is shallow over top-level keys; `prompt_vars` participates in `apply_ip_routing` alongside `params`/`headers`.

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
- **`prompts`**: `base_dir` (default `./prompts`) and `vars` (global Jinja2 variables)
- **`error_handling`** (optional): `enabled`, `show_details`, `include_type`, `prefix` — controls whether runtime errors in `/api/chat` and `/api/generate` are translated into LLM-style responses (default: enabled, prefix `[LLM ERROR]`)
- **`models`**: model list with a two-level structure:
  - Root level: `name` (required), `custom_name`, `remove_thinking_tags`, `prompt_caching`, `system_prompt_inline`, `system_prompt_file` (mutually exclusive; file wins on conflict; legacy `system_prompt` deprecated), `prompt_vars` (overrides global `prompts.vars`)
  - `params`: dict of OpenAI API parameters — passed through without validation
  - `headers`: dict of custom HTTP headers
  - `ip_routing`: list of IP-specific overrides (inheritance + shallow merge; `params`, `headers`, `prompt_vars` are dict-merged)

If `models` is empty — all available OpenAI models are exposed. If populated — only the listed ones.

Config hot-reload: on every request the file mtime is checked. On change — config is reloaded, the OpenAI client is recreated, and the model cache is refreshed. On load failure — the current config is preserved.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/chat` | POST | Chat completions (streaming/non-streaming) |
| `/api/generate` | POST | Text generation (streaming/non-streaming) |
| `/api/embed` | POST | Embeddings |
| `/api/tags` | GET/POST | List models |
| `/api/show` | POST | Model information |
| `/api/version` | GET | Service version |
| `/api/ps` | GET | Running models (mock) |
| `/health` | GET | Health check with OpenAI verification |
| `/` | GET | Service information |

## Testing

**Important**: Do NOT start the server automatically after changes. The user tests manually.

Manual tests: `tests/manual-check.http` — HTTP requests for all endpoints, including error cases. Port 11345 (docker-compose) or 11434 (locally).

## Development Notes

- **Package Manager**: UV with lock file (`uv.lock`)
- **Python**: 3.13+ (`.python-version`)
- **Flask debug mode**: enabled — auto-reload on code changes
- **Version**: 0.1.0
- **CI/CD**: GitHub Actions (`.github/workflows/ci.yml`) — linters + Docker image build to GHCR
- **Linters and tests**: always run via `make` (e.g. `make check`), never call ruff/mypy/pytest directly
- **Final check**: on completing any task, always run `make pre-commit` — it formats code, applies lint fixes, and runs the full check (format → lint-fix → check)
