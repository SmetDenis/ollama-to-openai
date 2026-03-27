# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ollama-to-OpenAI adapter — a Python service (Flask) that translates Ollama API requests into OpenAI API calls. Allows Ollama clients to transparently use OpenAI models (and compatible providers via LiteLLM).

## Architecture

Modular structure in the `ollama_adapter/` package. Dependencies: Flask, OpenAI SDK, PyYAML.

### Module Structure

```
ollama_adapter/
  __init__.py          # Empty
  __main__.py          # Entrypoint: create_app + app.run
  state.py             # Global state: CONFIG, client, CACHED_MODELS
  config.py            # load_config(), init_state(), hot-reload
  logging_utils.py     # TraceContextFilter, validation, log_request/response, @log_endpoint
  tracing.py           # LiteLLM headers, build_trace_*, capture/log headers
  thinking.py          # remove_thinking_tags(), _process_stream() — 5-state machine
  models.py            # Models, prompts, IP-routing, model config
  routes.py            # Flask Blueprint, all endpoints, shared completion helper
  app.py               # create_app() factory, before_request hooks
```

### Import Graph

`state.py` is a leaf node (imports nothing from the package). All modules import `state`. `routes.py` imports `logging_utils`, `tracing`, `thinking`, `models`. `config.py` imports `logging_utils` (TraceContextFilter) and `models` (get_and_cache_models on reload). No circular dependencies.

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
- **`apply_system_prompt(messages, adapter_params, model_id)`** (`models.py`) — injects the system prompt from config
- **`resolve_system_prompt(value)`** (`models.py`) — if the value ends with `.md`, reads the file on every request (hot-reload)
- **`apply_prompt_caching(messages, adapter_params, model_id)`** (`models.py`) — adds `cache_control` markers for Anthropic/Gemini
- **`remove_thinking_tags(content, model_id, remove_enabled)`** (`thinking.py`) — strips `<think>`/`<thinking>` tags
- **`_process_stream()`** (`thinking.py`) — 5-state machine for streaming tag removal
- **`_call_openai_streaming()`** / **`_call_openai_non_streaming()`** (`routes.py`) — shared helpers for `chat()` and `generate()`

### Prompts Directory

`prompts/` — system prompt files (`.md`) referenced by `system_prompt` in the model config. Mounted read-only in Docker. Files are re-read on every request — editable without restart.

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
- **`models`**: model list with a two-level structure:
  - Root level: `name` (required), `custom_name`, `remove_thinking_tags`, `prompt_caching`, `system_prompt`
  - `params`: dict of OpenAI API parameters — passed through without validation
  - `headers`: dict of custom HTTP headers
  - `ip_routing`: list of IP-specific overrides (inheritance + shallow merge)

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
