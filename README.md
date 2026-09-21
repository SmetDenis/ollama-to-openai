# Ollama to OpenAI Adapter

A Python service that translates Ollama API requests to OpenAI API calls, enabling Ollama clients to use OpenAI models (and compatible providers via LiteLLM) seamlessly. It also exposes an OpenAI-compatible API under `/v1` on the same port, so IDE agents, chat UIs and OpenAI SDK scripts get the same adapter features.

## Features

- Complete Ollama API compatibility (chat, generate, embed, tags, show)
- OpenAI-compatible API under `/v1` (chat completions, legacy completions, models) — faithful passthrough of tools, reasoning content and provider fields, real HTTP error codes, optional API keys
- Streaming and non-streaming responses
- Model name mapping (`custom_name`) with bidirectional resolution
- Per-model configuration: parameters, headers, system prompts
- IP-based routing — different clients get different models/settings
- System prompt injection from config (inline or template files, hot-reloaded per request); replace, prepend or append to the client's own system message
- Jinja2 prompt templating: `{% include "..." %}` between files and `{{ var }}` substitution
- Built-in date/time placeholders in prompts — `now` object, flat parts (`year`, `month`, `day`, `hour`, `minute`, `weekday`), and presets (`date_human`, `time_human`, `datetime_human`, `date_iso`, `datetime_iso`), computed per request in a configurable `prompts.timezone`; enables conditionals like `{% if weekday == "Friday" %}…{% endif %}`
- Prompt caching support (Anthropic/Gemini via LiteLLM)
- `<think>`/`<thinking>` tag removal (streaming and non-streaming)
- Client label cleanup — a leading `Text:` / `Текст:` line (Raycast) is stripped from the last user message before the `debug` check and before the upstream call; configurable via `input_cleanup`
- Runtime error translation — upstream failures (rate limit, auth, timeout, 5xx) become assistant messages with a `[LLM ERROR]` prefix so clients like Raycast always see a readable explanation instead of an HTTP 500
- Config hot-reload — changes to `config.yml` apply without restart
- Request/response logging with optional LiteLLM tracing integration
- Health monitoring

## Requirements

- Python 3.13+
- OpenAI API key (or compatible provider)

## Installation

1. Clone the repository
2. Copy `config-example.yml` to `config.yml`
3. Configure your API key and models in `config.yml`
4. Install dependencies:

```bash
uv sync
```

## Configuration

Edit `config.yml` (see `config-example.yml` for a full reference with comments).

```yaml
server:
    host: "0.0.0.0"
    port: 11434

openai:
    api_key: "your-api-key"
    # base_url: "https://your-litellm-proxy/v1"  # Optional custom endpoint

logging:
    log_level: "INFO"
    log_requests: true

models:
    -   name: openai/gpt-4o-mini
        custom_name: "GPT-4o Mini"

    -   name: openai/gpt-4o
        custom_name: "GPT-4o"
        remove_thinking_tags: true
        system_prompt_file: "assistant.md"   # relative to prompts.base_dir
        params:
            temperature: 0.7
            max_tokens: 2000
```

### Model Name Mapping

Use `custom_name` to expose models under friendly names:

```yaml
models:
    -   name: us.anthropic.claude-sonnet-4-5-20250929-v1:0
        custom_name: "Sonnet 4.5"
```

- `custom_name` is optional — if not specified, the original model name is used
- `custom_name` must be unique across all models
- Clients can use either the custom name or the original name in requests
- Responses return the custom name to clients
- When `models` is non-empty it is an allowlist: any other model name gets HTTP 404 on `/api/chat`, `/api/generate`, `/api/embed`, `/api/show` and `/v1/chat/completions`, `/v1/completions` (this also keeps clients from bypassing per-model `params` limits). **List embedding models too** if you use `/api/embed`. With an empty `models` list every upstream model is allowed.

### IP-Based Routing

Route different clients to different backend models based on IP address:

```yaml
clients:
    office:
        - "192.168.1.100"
        - "192.168.1.101"
    home: "10.0.0.5"

models:
    -   name: openai/gpt-4o
        custom_name: "Assistant"
        params:
            temperature: 0.7
        ip_routing:
            -   ip: "office"
                name: openai/gpt-4o-mini
                params:
                    temperature: 0.3
            -   ip: "home"
                params:
                    temperature: 0.9
```

Fields not specified in `ip_routing` entries inherit from the parent model. Dict fields (`params`, `headers`) are shallow-merged.

### System Prompts

A model can declare a system prompt either inline or from a file. The two are mutually exclusive — if both are set, the file wins and a warning is logged.

```yaml
models:
    -   name: openai/gpt-4o
        system_prompt_inline: "You are a helpful assistant."

    -   name: openai/gpt-4o-mini
        system_prompt_file: "assistant.md"   # any extension; resolved under prompts.base_dir
```

- `system_prompt_file` paths are **relative to `prompts.base_dir`** (default `./prompts`). Absolute paths and `..` are rejected to prevent reading files outside the prompts tree.
- Templates are re-rendered on every request — file edits and config changes take effect without a restart.
- `system_prompt_mode` controls what happens when the client sends its own system message (the first `system` or `developer` message):
  - `replace` (default) — the config prompt replaces it
  - `prepend` / `append` — the config prompt is joined before/after it (blank line for strings, an extra text part for content-part lists). Use this for IDE agents whose system message carries tool instructions.
  Allowed in `ip_routing` overrides too. Without a client system message the config prompt is always inserted first. On `/v1` a warning is logged whenever `replace` discards a non-empty client system/developer message.
- The legacy `system_prompt` field is no longer recognized — it is ignored and a deprecation warning is logged. Migrate to `system_prompt_inline` or `system_prompt_file`.

### Prompt Templates

Prompts are Jinja2 templates rendered inside a `SandboxedEnvironment`. You get `{% include %}`, `{% if %}`, `{% for %}`, filters, macros, and variable substitution.

```yaml
prompts:
    base_dir: "./prompts"
    vars:
        company_name: "Acme Corp"
        default_role: "junior"

models:
    -   name: openai/gpt-4o
        system_prompt_file: "role/main.md"
        prompt_vars:
            role: "senior"            # overrides default_role for this model
```

Inside `prompts/role/main.md`:

```jinja
You work for {{ company_name }} as a {{ role }} engineer.

{% include "snippets/safety.md" %}
{% include "snippets/style-guide.md" %}
```

**Variable priority (low → high):** built-in date/time → `prompts.vars` → `model.prompt_vars` → `ip_routing[matched].prompt_vars`. Merge is shallow (top-level keys).

**Built-in date/time variables.** Every rendered prompt automatically gets the current date/time, computed per request in `prompts.timezone` (IANA name, default `UTC`; e.g. `Europe/Moscow`). Same-named entries in `prompts.vars`/`prompt_vars` override them.

| Variable         | Type          | Example                                                          |
|------------------|---------------|------------------------------------------------------------------|
| `now`            | `datetime`    | `{{ now.strftime('%H:%M') }}` → `09:05`, `{{ now.month }}` → `6` |
| `year`           | int           | `2026`                                                           |
| `month`          | str (English) | `June`                                                           |
| `day`            | int           | `15`                                                             |
| `hour`           | int           | `9`                                                              |
| `minute`         | int           | `5`                                                              |
| `weekday`        | str (English) | `Monday`                                                         |
| `date_human`     | str           | `Monday, June 15, 2026`                                          |
| `time_human`     | str           | `09:05`                                                          |
| `datetime_human` | str           | `Monday, June 15, 2026, 09:05`                                   |
| `date_iso`       | str           | `2026-06-15`                                                     |
| `datetime_iso`   | str           | `2026-06-15 09:05`                                               |

```yaml
prompts:
    timezone: "Europe/Moscow"
```

Conditionals work with these too:

```jinja
You are an assistant. Today is {{ date_human }}.
{% if weekday == "Friday" %}
Remind the user to submit their weekly report.
{% endif %}
```

**Errors are surfaced to the client.** Missing files, undefined variables, syntax errors, sandbox violations, and include cycles all return HTTP 200 with an `assistant` message starting with `[PROMPT ERROR] ...` — the request is not forwarded to OpenAI.

**Security notes:**

- `FileSystemLoader` blocks `..` and absolute paths; templates cannot escape `prompts.base_dir`.
- `SandboxedEnvironment` blocks access to unsafe attributes (e.g. `__class__`, `__mro__`).
- Symlinks inside `prompts/` are followed — keep that in mind when constructing the directory.

**Breaking change from earlier versions:** `system_prompt_file` paths were resolved against CWD; they are now resolved against `prompts.base_dir`. Rewrite `system_prompt_file: prompts/main.md` as `system_prompt_file: main.md` (and put the file inside `prompts/`).

### OpenAI-compatible API (`/v1`)

Point any OpenAI client at `http://<host>:11434/v1`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="sk-local-1")
client.chat.completions.create(model="GPT-4o Mini", messages=[{"role": "user", "content": "Hello"}])
```

```yaml
openai_api:
    enabled: true          # false -> every /v1 route returns 404
    api_keys:              # empty/missing -> /v1 accepts any request
        - "sk-local-1"
```

- **Same adapter features:** `custom_name`, IP routing, system prompts (`system_prompt_mode`), prompt caching, tracing, `remove_thinking_tags` (streaming too), and the `debug` keyword.
- **Parameters:** everything the client sends (`tools`, `temperature`, `response_format`, images, ...) is forwarded; model `params` from the config override client values. Client-side credentials and LiteLLM control keys sent by a client are dropped (`api_key`, `api_base`, `base_url`, `api_version`, `custom_llm_provider`, `extra_headers`, `headers`, `litellm_params` and common provider auth params such as `vertex_project` or `aws_access_key_id`; the list is not exhaustive). A config `max_tokens` does not cap a client-sent `max_completion_tokens` — set both in `params` if you need a hard limit.
- **Responses:** upstream fields are passed through unchanged (`tool_calls`, `reasoning_content`, `logprobs`, ...); only `model` is replaced with the name the client requested. Streaming uses SSE; the usage chunk is sent only when the client asks for `stream_options.include_usage`.
- **Errors:** real HTTP status codes with the OpenAI error body `{"error": {"message", "type", "param", "code"}}`. Upstream errors keep their status and body (including 401/403 from the upstream provider). `error_handling` (the `[LLM ERROR]` text) applies only to the Ollama endpoints.
- **`/v1/completions`** runs through chat completions: `prompt` must be a single string; `echo`, `suffix` (fill-in-the-middle), `logprobs` and `best_of > 1` return HTTP 400.
- `GET /v1/models` keeps upstream error statuses (e.g. 429 with `retry-after`); an empty filtered model list is returned as `{"object": "list", "data": []}`.
- Authentication applies to `/v1` only — Ollama endpoints stay open. Without `api_keys` (including configs that have no `openai_api` section at all) a warning is logged at startup and on every config reload. Set `api_keys` whenever the port is reachable beyond localhost (the `debug` keyword reveals compiled system prompts).
- Not implemented: `/v1/responses`, `/v1/embeddings`.

### Client Input Cleanup

Some clients prepend a label line to what they send. Raycast sends:

```text
Text:
<what the user actually typed>
```

The adapter strips that label from the **last user message** before the `debug` keyword check
and before the upstream request is built — so `Text:\ndebug` still returns the compiled prompt,
and the model never sees the label.

```yaml
input_cleanup:
    enabled: true          # false -> nothing is stripped
    strip_prefixes:        # REPLACES the defaults entirely
        - "Text:"
        - "Текст:"
        - "Prompt:"
```

Defaults when the section is absent: enabled, with `["Text:", "Текст:"]`.

- The label must be the **whole first line**: prefix, optional spaces/tabs, then a line break (case-insensitive). A mid-text `Text: ...` is left alone, and so is `Text: debug` on a single line.
- A message consisting of nothing but the label is left untouched (it would otherwise become an empty request).
- Only the last user message is touched — earlier turns keep whatever the client sent. For a content-part message (`/v1` with images) only the first text part is cleaned.
- Applies to `/api/chat`, `/api/generate`, `/v1/chat/completions` and `/v1/completions`.

### Tracing (LiteLLM Integration)

```yaml
tracing:
    enabled: true
    log_headers: true
    send_trace_headers: true
    trace_id_prefix: "oa"
    tags: "ollama-adapter,production"
```

When enabled, each request gets `request_id`/`trace_id` visible in logs. LiteLLM response headers (cost, duration, model-id) are extracted and logged.

## Running

### Local

```bash
python3 -m ollama_adapter
```

### Docker

```bash
docker build -t ollama-to-openai .
docker run -p 11434:11434 -v ./config.yml:/app/config.yml:ro ollama-to-openai
```

### Docker Compose

```bash
docker-compose up -d
```

The service starts on `http://localhost:11434` by default (or `http://localhost:11345` with docker-compose).

## API Endpoints

| Endpoint        | Method   | Description                                |
|-----------------|----------|--------------------------------------------|
| `/api/chat`     | POST     | Chat completions (streaming/non-streaming) |
| `/api/generate` | POST     | Text generation (streaming/non-streaming)  |
| `/api/embed`    | POST     | Generate embeddings                        |
| `/api/tags`     | GET/POST | List available models                      |
| `/api/show`     | POST     | Model information                          |
| `/api/version`  | GET      | Service version                            |
| `/api/ps`       | GET      | List running models                        |
| `/health`       | GET      | Health check with OpenAI connectivity      |
| `/`             | GET      | Service info                               |

OpenAI-compatible endpoints (see [OpenAI-compatible API](#openai-compatible-api-v1)):

| Endpoint               | Method | Description                                         |
|------------------------|--------|-----------------------------------------------------|
| `/v1/chat/completions` | POST   | Chat completions (SSE streaming/non-streaming)      |
| `/v1/completions`      | POST   | Legacy text completions (via chat completions)      |
| `/v1/models`           | GET    | List models                                         |
| `/v1/models/{id}`      | GET    | Retrieve a model (ids may contain `/`)              |

## Usage Examples

```bash
# List models
curl http://localhost:11434/api/tags

# Chat completion
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GPT-4o Mini",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'

# Streaming chat
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GPT-4o Mini",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'

# Text generation
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GPT-4o Mini",
    "prompt": "Explain quantum computing",
    "stream": false
  }'

# OpenAI-compatible chat (streaming)
curl -N -X POST http://localhost:11434/v1/chat/completions \
  -H "Authorization: Bearer sk-local-1" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GPT-4o Mini",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'

# Embeddings
curl -X POST http://localhost:11434/api/embed \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/text-embedding-3-small",
    "input": "Hello world"
  }'
```

## Project Structure

```
ollama_adapter/
  __init__.py          # Package init
  __main__.py          # Entrypoint (python -m ollama_adapter)
  state.py             # Global state: CONFIG, client, CACHED_MODELS, jinja_env
  config.py            # Config loading, validation, hot-reload
  logging_utils.py     # Request validation, logging, @log_endpoint decorator
  tracing.py           # LiteLLM tracing integration
  thinking.py          # <think>/<thinking> tag removal (regex + ThinkingTagFilter for streams)
  prompt_renderer.py   # Jinja2 sandboxed environment + PromptRenderError
  error_formatter.py   # Runtime errors -> "[LLM ERROR]" assistant text (Ollama endpoints)
  input_cleanup.py     # Strips client label prefixes ("Text:") from the last user message
  debug_prompt.py      # "debug" keyword: compiled prompt output
  models.py            # Model resolution, caching, IP routing, system prompts
  completion.py        # Shared upstream pipeline (param merge, prompts, headers, upstream call)
  routes.py            # Flask Blueprint with the Ollama API endpoints
  openai_routes.py     # Flask Blueprint with the OpenAI-compatible /v1 endpoints
  openai_translate.py  # /v1 payload/chunk transforms and SSE framing
  openai_errors.py     # /v1 exception -> HTTP status + OpenAI error body
  app.py               # Flask app factory
```

## Testing

Automated tests: `make check` (format, lint, mypy, pytest). Manual test cases are available in `tests/manual-check.http`.
