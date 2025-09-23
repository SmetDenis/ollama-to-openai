# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Ollama-to-OpenAI adapter service written in Python. It acts as a proxy that translates Ollama API requests to OpenAI API calls, allowing Ollama clients to use OpenAI models seamlessly.

## Architecture

The application consists of a single Flask web server (`ollama_to_openai_adapter.py`) that:

1. **Configuration Management**: Loads settings from `config.yml` including server host/port, OpenAI API credentials, allowed models list, logging configuration, and model-specific settings
2. **Model Caching**: Fetches and caches OpenAI models, mapping them to Ollama-compatible format without `:latest` suffix
3. **API Translation**: Implements comprehensive Ollama endpoints (`/api/tags`, `/api/show`, `/api/chat`, `/api/generate`, `/api/embed`, `/api/version`, `/api/ps`, `/health`, `/`) that proxy to OpenAI
4. **Streaming Support**: Handles both streaming and non-streaming chat completions and text generation
5. **Logging System**: Comprehensive request/response logging with configurable levels and request tracking
6. **Input Validation**: Robust validation for all API endpoints with proper error handling
7. **Health Monitoring**: Health check endpoint for service monitoring

## Commands

## Project Structure

```
.
├── ollama_to_openai_adapter.py  # Main Flask application
├── config.yml                  # Active configuration file
├── config-example.yml          # Configuration template
├── pyproject.toml              # Project metadata and dependencies
├── uv.lock                     # UV package manager lock file
├── .venv/                      # Virtual environment (UV-managed)
├── tests/
│   └── manual-check.http       # HTTP test cases for manual testing
├── README.md                   # Basic project description
├── LICENSE                     # Project license
└── CLAUDE.md                   # This file
```

### Setup and Installation
```bash
# Virtual environment is managed by UV package manager
# Dependencies are defined in pyproject.toml
# No manual installation needed if .venv exists

# To recreate environment (if needed):
# uv sync
```

### Running the Application
```bash
# Run the adapter server
python3 ollama_to_openai_adapter.py

# Or with full path to venv python
./.venv/bin/python3 ollama_to_openai_adapter.py
```

### Development
```bash
# The project uses UV package manager
# Dependencies are defined in pyproject.toml:
# - flask>=3.1.2
# - openai>=1.108.2
# - pyyaml>=6.0.2

# Virtual environment is in .venv/ directory
# Python executable: .venv/bin/python (symlinked to system Python 3.13)
```

## Configuration

The `config.yml` file contains:
- `server.host` and `server.port`: Web server binding configuration
- `openai.api_key`: OpenAI API key (required)
- `openai.base_url`: Custom OpenAI endpoint URL (optional)
- `openai.allowed_models`: List of specific models to expose (optional, defaults to all)
- `logging.log_level`: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.log_requests`: Enable/disable request/response logging
- `models.<model_name>.temperature`: Per-model temperature override

## Key Implementation Details

- **Model Mapping**: OpenAI models are mapped to Ollama format without `:latest` suffix, with synthetic metadata matching Ollama specifications
- **Error Handling**: Configuration validation on startup with descriptive error messages, comprehensive input validation for all endpoints
- **Response Format**: All responses follow Ollama API specifications with timing and token usage data
- **Streaming**: Uses Flask's `stream_with_context` for real-time chat streaming and text generation
- **Temperature Configuration**: Default temperature 0.3, configurable per model in config.yml
- **Debug Mode**: Flask runs in debug mode with auto-reload for development
- **Multiple Endpoints**: Complete Ollama API compatibility including `/api/generate`, `/api/embed`, `/health`

## Testing

**Important**: The user will handle testing manually. Do not automatically run the server after making changes. The user will start the server themselves to test any modifications.

### Manual Testing

Use the HTTP test cases in `tests/manual-check.http` which includes comprehensive test scenarios:
- Health checks and service information
- Model listing and details
- Chat completions (streaming and non-streaming)
- Text generation with prompts
- Embeddings generation
- Error handling and validation tests

## Documentation Resources

### API Documentation Access
Use Context7 to access up-to-date documentation for libraries:

**OpenAI Python Library**:
- Library ID: `/openai/openai-python`
- Key features: Chat completions, streaming, function calling, assistants, structured outputs with Pydantic
- Installation: `pip install openai`

**Ollama API Documentation**:
- The adapter implements Ollama API compatibility based on official Ollama REST API specification
- All endpoints return responses in Ollama format with proper timing and usage statistics

### Usage
```python
# Access documentation via Context7
mcp__context7__resolve_library_id(libraryName="openai python")
mcp__context7__get_library_docs(context7CompatibleLibraryID="/openai/openai-python")
```

## Dependencies

Core dependencies managed in virtual environment (`.venv/`):
- Flask 3.1.2 (web server)
- openai 1.108.2 (OpenAI client)
- PyYAML 6.0.2 (configuration parsing)

**Package Management**: Project uses UV package manager with dependencies defined in `pyproject.toml` and locked in `uv.lock`
