# Ollama to OpenAI Adapter

A Python service that translates Ollama API requests to OpenAI API calls, enabling Ollama clients to use OpenAI models (and compatible providers via LiteLLM) seamlessly.

## Features

- Complete Ollama API compatibility (chat, generate, embed, tags, show)
- Streaming and non-streaming responses
- Model name mapping (`custom_name`) with bidirectional resolution
- Per-model configuration: parameters, headers, system prompts
- IP-based routing — different clients get different models/settings
- System prompt injection from config (inline or template files, hot-reloaded per request)
- Jinja2 prompt templating: `{% include "..." %}` between files and `{{ var }}` substitution
- Prompt caching support (Anthropic/Gemini via LiteLLM)
- `<think>`/`<thinking>` tag removal (streaming and non-streaming)
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
  - name: openai/gpt-4o-mini
    custom_name: "GPT-4o Mini"

  - name: openai/gpt-4o
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
  - name: us.anthropic.claude-sonnet-4-5-20250929-v1:0
    custom_name: "Sonnet 4.5"
```

- `custom_name` is optional — if not specified, the original model name is used
- `custom_name` must be unique across all models
- Clients can use either the custom name or the original name in requests
- Responses return the custom name to clients

### IP-Based Routing

Route different clients to different backend models based on IP address:

```yaml
clients:
  office:
    - "192.168.1.100"
    - "192.168.1.101"
  home: "10.0.0.5"

models:
  - name: openai/gpt-4o
    custom_name: "Assistant"
    params:
      temperature: 0.7
    ip_routing:
      - ip: "office"
        name: openai/gpt-4o-mini
        params:
          temperature: 0.3
      - ip: "home"
        params:
          temperature: 0.9
```

Fields not specified in `ip_routing` entries inherit from the parent model. Dict fields (`params`, `headers`) are shallow-merged.

### System Prompts

A model can declare a system prompt either inline or from a file. The two are mutually exclusive — if both are set, the file wins and a warning is logged.

```yaml
models:
  - name: openai/gpt-4o
    system_prompt_inline: "You are a helpful assistant."

  - name: openai/gpt-4o-mini
    system_prompt_file: "assistant.md"   # any extension; resolved under prompts.base_dir
```

- `system_prompt_file` paths are **relative to `prompts.base_dir`** (default `./prompts`). Absolute paths and `..` are rejected to prevent reading files outside the prompts tree.
- Templates are re-rendered on every request — file edits and config changes take effect without a restart.
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
  - name: openai/gpt-4o
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

**Variable priority (low → high):** `prompts.vars` → `model.prompt_vars` → `ip_routing[matched].prompt_vars`. Merge is shallow (top-level keys).

**Errors are surfaced to the client.** Missing files, undefined variables, syntax errors, sandbox violations, and include cycles all return HTTP 200 with an `assistant` message starting with `[PROMPT ERROR] ...` — the request is not forwarded to OpenAI.

**Security notes:**
- `FileSystemLoader` blocks `..` and absolute paths; templates cannot escape `prompts.base_dir`.
- `SandboxedEnvironment` blocks access to unsafe attributes (e.g. `__class__`, `__mro__`).
- Symlinks inside `prompts/` are followed — keep that in mind when constructing the directory.

**Breaking change from earlier versions:** `system_prompt_file` paths were resolved against CWD; they are now resolved against `prompts.base_dir`. Rewrite `system_prompt_file: prompts/main.md` as `system_prompt_file: main.md` (and put the file inside `prompts/`).

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
  thinking.py          # <think>/<thinking> tag removal (regex + streaming state machine)
  prompt_renderer.py   # Jinja2 sandboxed environment + PromptRenderError
  models.py            # Model resolution, caching, IP routing, system prompts
  routes.py            # Flask Blueprint with all API endpoints
  app.py               # Flask app factory
```

## Testing

Manual test cases are available in `tests/manual-check.http`.
