# Ollama to OpenAI Adapter

A Python service that translates Ollama API requests to OpenAI API calls, enabling Ollama clients to use OpenAI models seamlessly.

## Features

- Complete Ollama API compatibility
- OpenAI model mapping and caching
- Streaming and non-streaming responses
- Configurable model parameters
- Request/response logging
- Health monitoring

## Requirements

- Python 3.13+
- OpenAI API key

## Installation

1. Clone the repository
2. Copy `config-example.yml` to `config.yml`
3. Configure your OpenAI API key in `config.yml`

## Configuration

Edit `config.yml`:

```yaml
server:
  host: "0.0.0.0"
  port: 11434

openai:
  api_key: "your-openai-api-key"
  # base_url: "https://api.openai.com/v1"  # Optional custom endpoint

logging:
  log_level: "INFO"
  log_requests: true

# Model configurations (list format)
models:
  - name: "openai/gpt-4o-mini"
  - name: "openai/gpt-4o"
    temperature: 0.7
    max_tokens: 2000
```

### Model Name Mapping

You can use custom display names for models while maintaining the correct OpenAI model IDs internally:

```yaml
models:
  - name: openai/gpt-4o
    custom_name: "GPT-4o"
    temperature: 0.7

  - name: us.anthropic.claude-sonnet-4-5-20250929-v1:0
    custom_name: "Sonnet 4.5"
```

**Features:**
- `custom_name` is optional - if not specified, the original OpenAI model name is used
- `custom_name` must be unique across all models
- Clients can use either the custom name OR the original name in requests
- The adapter automatically maps custom names to OpenAI model IDs for API calls
- Responses return the custom name to clients

**Example request:**
```bash
curl http://localhost:11434/api/chat -d '{
  "model": "Sonnet 4.5",
  "messages": [{"role": "user", "content": "Hello"}]
}'
```

## Running

### Local Python

```bash
python3 ollama_to_openai_adapter.py
```

### Docker

```bash
# Build and run with Docker
docker build -t ollama-to-openai .
docker run -p 11434:11434 -v ./config.yml:/app/config.yml:ro ollama-to-openai
```

### Docker Compose

```bash
# Run with docker-compose
docker-compose up -d
```

The service will start on `http://localhost:11434` by default (or `http://localhost:11345` with docker-compose).

## API Endpoints

- `GET /api/tags` - List available models
- `POST /api/chat` - Chat completions
- `POST /api/generate` - Text generation
- `POST /api/embed` - Generate embeddings
- `POST /api/show` - Model information
- `GET /api/version` - Service version
- `GET /api/ps` - List running models
- `GET /health` - Health check

## Usage Example

```bash
# List models
curl http://localhost:11434/api/tags

# Chat completion
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

## Testing

Manual test cases are available in `tests/manual-check.http`.
