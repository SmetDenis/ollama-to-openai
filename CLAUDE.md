# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Ollama-to-OpenAI adapter service written in Python. It acts as a proxy that translates Ollama API requests to OpenAI API calls, allowing Ollama clients to use OpenAI models seamlessly.

## Architecture

The application consists of a single Flask web server (`ollama_to_openai_adapter.py`) that:

1. **Configuration Management**: Loads settings from `config.yaml` including server host/port, OpenAI API credentials, and allowed models list
2. **Model Caching**: Fetches and caches OpenAI models, mapping them to Ollama-compatible format with `:latest` tags
3. **API Translation**: Implements key Ollama endpoints (`/api/tags`, `/api/show`, `/api/chat`) that proxy to OpenAI
4. **Streaming Support**: Handles both streaming and non-streaming chat completions

## Commands

### Setup and Installation
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (if needed)
pip install -r requirements.txt  # Note: no requirements.txt exists, dependencies are in venv
```

### Running the Application
```bash
# Run the adapter server
python3 ollama_to_openai_adapter.py

# Or with activated venv
./venv/bin/python3 ollama_to_openai_adapter.py
```

### Development
```bash
# Check installed packages
./venv/bin/pip list

# Install new packages
./venv/bin/pip install package_name
```

## Configuration

The `config.yaml` file contains:
- `server.host` and `server.port`: Web server binding configuration
- `openai.api_key`: OpenAI API key (required)
- `openai.base_url`: Custom OpenAI endpoint URL (optional)
- `openai.allowed_models`: List of specific models to expose (optional, defaults to all)

## Key Implementation Details

- **Model Mapping**: OpenAI models are mapped to Ollama format with `:latest` suffix and synthetic metadata
- **Error Handling**: Configuration validation on startup with descriptive error messages
- **Response Format**: All responses follow Ollama API specifications with timing and token usage data
- **Streaming**: Uses Flask's `stream_with_context` for real-time chat streaming

## Testing

**Important**: The user will handle testing manually. Do not automatically run the server after making changes. The user will start the server themselves to test any modifications.

## Documentation Resources

### API Documentation Access
Use Context7 to access up-to-date documentation for libraries:

**OpenAI Python Library**:
- Library ID: `/openai/openai-python`
- Key features: Chat completions, streaming, function calling, assistants, structured outputs with Pydantic
- Installation: `pip install openai`

**Ollama API Documentation**:
- Local reference: `ollama-api-docs.md` (located in project root)
- Contains complete Ollama REST API specification including endpoints, request/response formats, and examples

### Usage
```python
# Access documentation via Context7
mcp__context7__resolve_library_id(libraryName="openai python")
mcp__context7__get_library_docs(context7CompatibleLibraryID="/openai/openai-python")
```

## Dependencies

Core dependencies managed in virtual environment (`venv/`):
- Flask 3.1.2 (web server)
- openai 1.108.2 (OpenAI client)
- PyYAML 6.0.2 (configuration parsing)
