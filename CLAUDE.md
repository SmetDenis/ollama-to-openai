# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Ollama-to-OpenAI adapter service written in Python. It acts as a proxy that translates Ollama API requests to OpenAI API calls, allowing Ollama clients to use OpenAI models seamlessly.

The service supports Docker containerization with docker-compose for easy deployment.

## Architecture

The application consists of a single Flask web server (`ollama_to_openai_adapter.py`) that:

1. **Configuration Management**: Loads settings from `config.yml` including server host/port, OpenAI API credentials, models list with filtering, logging configuration, and comprehensive model-specific parameter support
2. **Model Caching**: Fetches and caches OpenAI models, mapping them to Ollama-compatible format with synthetic metadata
3. **API Translation**: Implements complete Ollama endpoints (`/api/tags`, `/api/show`, `/api/chat`, `/api/generate`, `/api/embed`, `/api/version`, `/api/ps`, `/health`, `/`) that proxy to OpenAI
4. **Streaming Support**: Handles both streaming and non-streaming chat completions and text generation with proper NDJSON format
5. **Logging System**: Comprehensive request/response logging with configurable levels, request tracking, and performance metrics
6. **Input Validation**: Robust validation for all API endpoints with proper error handling and descriptive error messages
7. **Health Monitoring**: Health check endpoint for service monitoring with OpenAI connectivity status
8. **Parameter Passthrough**: All OpenAI API parameters can be configured per model without validation, providing full flexibility

## Commands

### Running the Application
```bash
# Local Python execution
python3 ollama_to_openai_adapter.py

# Or with virtual environment
./.venv/bin/python3 ollama_to_openai_adapter.py

# Docker build and run
docker build -t ollama-to-openai .
docker run -p 11434:11434 -v ./config.yml:/app/config.yml:ro ollama-to-openai

# Docker Compose (service runs on port 11345)
docker-compose up -d
```


### Setup and Installation
```bash
# Virtual environment is managed by UV package manager
# Dependencies are defined in pyproject.toml
# No manual installation needed if .venv exists

# Configuration setup
cp config-example.yml config.yml
# Edit config.yml with your OpenAI API key and settings

# To recreate environment (if needed):
# uv sync
```


## Configuration

The `config.yml` file contains:
- `server.host` and `server.port`: Web server binding configuration
- `openai.api_key`: OpenAI API key (required)
- `openai.base_url`: Custom OpenAI endpoint URL (optional, for Azure OpenAI, local LLMs, etc.)
- `logging.log_level`: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.log_requests`: Enable/disable detailed request/response logging
- `models`: List of model configurations with full parameter support:
  - `name`: Model identifier (required)
  - All OpenAI API parameters supported (temperature, max_tokens, top_p, frequency_penalty, presence_penalty, response_format, stop, logit_bias, etc.)
  - Parameters are passed through to OpenAI API without validation

## Key Implementation Details

- **Model Mapping**: OpenAI models are mapped to Ollama format with synthetic metadata (parent_model, format, family, parameter_size, quantization_level)
- **Error Handling**: Configuration validation on startup, comprehensive input validation with descriptive error messages for all endpoints
- **Response Format**: All responses follow Ollama API specifications with proper timing (total_duration, eval_duration, load_duration) and token usage data
- **Streaming**: Uses Flask's `stream_with_context` for real-time NDJSON streaming in chat and generation endpoints
- **Parameter Flexibility**: All OpenAI API parameters can be configured per model in config.yml and are passed through without validation
- **Debug Mode**: Flask runs in debug mode with auto-reload and stat-based reloader for development
- **Complete API Coverage**: Full Ollama API compatibility including chat, generate, embed, tags, show, version, ps, health endpoints
- **Logging**: Request/response logging with performance tracking, configurable detail levels, and error tracking
- **Health Monitoring**: Health endpoint checks OpenAI connectivity and reports cached model count

## Testing

**Important**: The user will handle testing manually. Do not automatically run the server after making changes. The user will start the server themselves to test any modifications.

### Manual Testing

Use the HTTP test cases in `tests/manual-check.http` (178 lines) which includes comprehensive test scenarios:
- Health checks and service information
- Model listing and details (GET and POST formats)
- Chat completions (streaming and non-streaming, multi-turn conversations)
- Text generation with prompts (streaming and non-streaming)
- Process listing (mock endpoint)
- Error handling and validation tests (invalid models, missing fields, malformed JSON)

**Note**: Test cases use port 11345 (docker-compose port), adjust to 11434 for local development.


## Development Notes

- **Package Management**: Uses UV package manager with dependencies in `pyproject.toml`
- **Environment**: Virtual environment managed in `.venv/` directory
- **Version**: 0.1.0 (consistent across project files)
- **Python**: Requires Python 3.13+
