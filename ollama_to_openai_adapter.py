import sys
import yaml
import json
import time
import re
import logging
from functools import wraps
from datetime import datetime
from flask import Flask, request, jsonify, Response, stream_with_context
from openai import OpenAI

# Will configure logging after loading config
logger = logging.getLogger(__name__)

# --- Input validation helpers ---

def validate_json_request():
    """Validate that request contains valid JSON."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body cannot be empty"}), 400

    return data, None

def validate_model_parameter(data):
    """Validate model parameter exists and is valid."""
    model = data.get("model") or data.get("name")
    if not model:
        return None, jsonify({"error": "Parameter 'model' or 'name' is required"}), 400

    if not isinstance(model, str) or not model.strip():
        return None, jsonify({"error": "Parameter 'model' must be a non-empty string"}), 400

    return model.strip(), None

# --- Logging helpers ---

def log_request(endpoint, method, data=None):
    """Log incoming request details."""
    if not CONFIG.get('logging', {}).get('log_requests', True):
        return

    log_data = {
        "endpoint": endpoint,
        "method": method,
        "client_ip": request.remote_addr,
        "user_agent": request.headers.get('User-Agent', 'Unknown')
    }

    if data:
        # Don't log sensitive data in full, just structure
        if isinstance(data, dict):
            safe_data = {k: f"<{type(v).__name__}>" if k in ['messages', 'prompt'] and len(str(v)) > 100
                        else v for k, v in data.items()}
            log_data["request_data"] = safe_data
        else:
            log_data["request_data"] = f"<{type(data).__name__}>"

    logger.info(f"Request: {log_data}")

def log_response(endpoint, status_code, response_data=None, error=None):
    """Log response details."""
    if not CONFIG.get('logging', {}).get('log_requests', True):
        return

    log_data = {
        "endpoint": endpoint,
        "status_code": status_code
    }

    if error:
        log_data["error"] = str(error)
        logger.error(f"Response: {log_data}")
    elif response_data:
        # Truncate large responses for readability
        if isinstance(response_data, dict):
            # For streaming responses or large data, show structure only
            if any(key in response_data for key in ['models', 'embeddings']) and len(str(response_data)) > 500:
                log_data["response"] = {k: f"<{len(v)} items>" if isinstance(v, list) else f"<{type(v).__name__}>"
                                      for k, v in response_data.items()}
            else:
                log_data["response"] = response_data
        elif isinstance(response_data, str) and len(response_data) > 200:
            log_data["response"] = response_data[:200] + "..."
        else:
            log_data["response"] = response_data

        logger.info(f"Response: {log_data}")
    else:
        logger.info(f"Response: {log_data}")


def log_endpoint(f):
    """Decorator to log requests and responses for endpoints."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get the actual endpoint path
        endpoint_path = request.path

        # Log request with data if it's a JSON request
        if request.is_json:
            try:
                data = request.get_json()
                log_request(endpoint_path, request.method, data)
            except:
                log_request(endpoint_path, request.method)
        else:
            log_request(endpoint_path, request.method)

        start_time = time.time()

        try:
            # Call the original function
            result = f(*args, **kwargs)
            duration = time.time() - start_time

            # Handle different response types
            if isinstance(result, tuple):
                response, status_code = result
                if hasattr(response, 'get_json'):
                    try:
                        response_data = response.get_json()
                        log_response(endpoint_path, status_code, response_data)
                    except:
                        log_response(endpoint_path, status_code, None)
                else:
                    log_response(endpoint_path, status_code, None)

                logger.info(f"Request completed in {duration:.3f}s")
                return result
            else:
                # Single response object
                status_code = 200
                if hasattr(result, 'get_json'):
                    try:
                        response_data = result.get_json()
                        log_response(endpoint_path, status_code, response_data)
                    except:
                        log_response(endpoint_path, status_code, None)
                elif isinstance(result, Response):
                    status_code = result.status_code
                    log_response(endpoint_path, status_code, None)
                else:
                    log_response(endpoint_path, status_code, None)

                logger.info(f"Request completed in {duration:.3f}s")
                return result

        except Exception as e:
            # Log error
            duration = time.time() - start_time
            logger.error(f"Endpoint {endpoint_path} failed after {duration:.3f}s: {str(e)}")
            log_response(endpoint_path, 500, None, str(e))
            raise

    return decorated_function

def remove_thinking_tags(content, model_id, remove_enabled):
    """
    Remove <think> or <thinking> tags from the beginning of content if enabled.
    Logs removed thinking content at DEBUG level.

    Args:
        content: The content to process (may be None or empty)
        model_id: Model identifier for logging
        remove_enabled: Boolean indicating if tag removal is enabled

    Returns:
        Cleaned content or original content if no tags found
    """
    # Early return if feature is disabled or content is empty
    if not remove_enabled or not content:
        return content

    # Regex pattern to match thinking tags at the beginning of content
    # Pattern explanation:
    # ^\s*          - Start of string with optional leading whitespace
    # <think(?:ing)?>  - Match <think> or <thinking>
    # (.*?)         - Capture content inside tags (non-greedy)
    # </think(?:ing)?> - Match closing </think> or </thinking>
    # \s*           - Optional trailing whitespace
    pattern = r'^\s*<think(?:ing)?>(.*?)</think(?:ing)?>\s*'

    match = re.match(pattern, content, re.DOTALL | re.IGNORECASE)
    if match:
        thinking_content = match.group(1)
        cleaned_content = content[match.end():]

        # Log removed thinking content (first 100 chars) at DEBUG level
        thinking_preview = thinking_content[:100] + "..." if len(thinking_content) > 100 else thinking_content
        logger.debug(f"Removed thinking tags from model '{model_id}'. Thinking content ({len(thinking_content)} chars): {thinking_preview}")

        return cleaned_content

    return content

def load_config(path='config.yml'):
    """Loads and validates YAML configuration file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file '{path}' not found")
        exit(1)
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in file '{path}': {e}")
        exit(1)

    if not config.get('openai', {}).get('api_key'):
        raise ValueError("Missing required parameter 'openai.api_key' in config.yml")
    if not config.get('server', {}).get('host'):
        raise ValueError("Missing required parameter 'server.host' in config.yml")
    if not config.get('server', {}).get('port'):
        raise ValueError("Missing required parameter 'server.port' in config.yml")

    # Validate custom_name uniqueness
    models_config = config.get('models', [])
    if models_config:
        custom_names = []
        for model in models_config:
            if isinstance(model, dict) and 'custom_name' in model:
                custom_names.append(model['custom_name'])

        # Check for duplicates
        duplicates = [name for name in set(custom_names) if custom_names.count(name) > 1]
        if duplicates:
            logger.error(f"Duplicate custom_name values found: {duplicates}")
            sys.exit(1)

    return config

CONFIG = load_config()

# Configure logging based on config
log_level = getattr(logging, CONFIG.get('logging', {}).get('log_level', 'INFO').upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

try:
    client = OpenAI(
        api_key=CONFIG['openai']['api_key'],
        base_url=CONFIG['openai'].get('base_url', 'https://api.openai.com/v1')
    )
except Exception as e:
    logger.error(f"Error initializing OpenAI client: {e}")
    exit(1)

app = Flask(__name__)
CACHED_MODELS = []

def get_display_name(original_name):
    """
    Returns custom_name if set, otherwise original_name.

    Args:
        original_name: Original model name from OpenAI

    Returns:
        custom_name or original_name
    """
    models_config = CONFIG.get('models', [])
    for model in models_config:
        if isinstance(model, dict) and model.get('name') == original_name and 'custom_name' in model:
            return model['custom_name']
    return original_name

def resolve_model_name(client_name):
    """
    Resolves model name from client to original OpenAI name.
    Works with both custom_name and original names as synonyms.

    Args:
        client_name: Model name from client request

    Returns:
        Original model name for OpenAI API
    """
    models_config = CONFIG.get('models', [])

    # First check if this is a custom_name
    for model in models_config:
        if isinstance(model, dict) and model.get('custom_name') == client_name:
            return model['name']

    # If not found as custom_name, return as-is (may be original name)
    return client_name

def get_and_cache_models():
    """
    Fetches, filters, maps and caches model list.
    Updated according to Ollama API documentation.
    """
    global CACHED_MODELS
    if CACHED_MODELS:
        return CACHED_MODELS

    logger.info("Model cache is empty. Requesting model list from OpenAI...")
    try:
        all_models_response = client.models.list().data
        models_config = CONFIG.get('models', [])

        # If models list is empty, use all available models
        # Otherwise, filter to only those specified in models list
        filtered_models = all_models_response
        if models_config:
            allowed_model_names = [model_entry['name'] for model_entry in models_config if 'name' in model_entry]
            filtered_models = [m for m in all_models_response if m.id in allowed_model_names]

        CACHED_MODELS = [
            {
                "name": get_display_name(model.id),
                "model": get_display_name(model.id),
                "modified_at": datetime.fromtimestamp(model.created).isoformat() + "Z",
                "size": 0,
                "digest": model.id,  # Keep original name for internal identification
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "llama",
                    "families": ["llama"],
                    "parameter_size": "8.0B",
                    "quantization_level": "Q4_0",
                }
            }
            for model in filtered_models
        ]
        logger.info(f"Models successfully loaded and cached. Found: {len(CACHED_MODELS)}")
        return CACHED_MODELS
    except Exception as e:
        logger.error(f"Critical error getting models from OpenAI: {e}")
        return []

@app.route('/api/tags', methods=['GET', 'POST'])
@log_endpoint
def handle_tags():
    models = get_and_cache_models()
    if not models:
        return jsonify({"error": "Failed to get model list from OpenAI"}), 500

    if request.method == 'GET':
        return jsonify({"models": models})

    if request.method == 'POST':
        data, error = validate_json_request()
        if error:
            return error

        model_identifier, error = validate_model_parameter(data)
        if error:
            return error

        # Resolve model_identifier to original name
        original_name = resolve_model_name(model_identifier)
        # Search by display name (name/model) OR by digest (original name)
        found_model = next((
            m for m in models
            if m['name'] == model_identifier
            or m['model'] == model_identifier
            or m['digest'] == original_name
        ), None)
        return jsonify({"models": [found_model]}) if found_model else (jsonify({"error": "Model not found"}), 404)

@app.route('/api/show', methods=['POST'])
@log_endpoint
def show_model():
    """
    Emulates /api/show endpoint.
    Updated according to Ollama API documentation.
    """
    data, error = validate_json_request()
    if error:
        return error

    model_id, error = validate_model_parameter(data)
    if error:
        return error

    # Resolve to original name for modelfile comment
    original_name = resolve_model_name(model_id)

    response_data = {
        "modelfile": "\n".join([
            f'# Modelfile generated by "ollama show"',
            f'# To build a new Modelfile based on this one, replace the FROM line with:',
            f'# FROM {original_name}',
            '',
            f'FROM /Users/matt/.ollama/models/blobs/sha256:200765e1283640ffbd013184bf496e261032fa75b99498a9613be4e94d63ad52',
            'TEMPLATE """{{ .System }}',
            'USER: {{ .Prompt }}',
            'ASSISTANT: """',
            'PARAMETER num_ctx 100000',
            'PARAMETER stop "</s>"',
            'PARAMETER stop "USER:"',
            'PARAMETER stop "ASSISTANT:"'
        ]),

        "parameters": "\n".join([
            'num_keep                       24',
            'stop                           "<|start_header_id|>"',
            'stop                           "<|end_header_id|>"',
            'stop                           "<|eot_id|>"'
        ]),

        "template": "\n".join([
            '{{ if .System }}<|start_header_id|>system<|end_header_id|>',
            '',
            '{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>',
            '',
            '{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>',
            '',
            '{{ .Response }}<|eot_id|>'
        ]),

        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": "llama",
            "families": ["llama"],
            "parameter_size": "8.0B",
            "quantization_level": "Q4_0",
        },

        "model_info": {
            "general.architecture": "llama",
            "general.file_type": 2,
            "llama.context_length": 256000,
        },

        "capabilities": ["completion", "vision"]
    }
    return jsonify(response_data)

def get_model_config(model_id):
    """
    Returns model configuration split into OpenAI and adapter parameters.
    Works with models list format: [{name: "model-name", temperature: 0.7, max_tokens: 1000}, ...]
    Passes through OpenAI parameters without validation.
    Supports both custom_name and original model names.

    Args:
        model_id: Model name (custom_name or original)

    Returns:
        tuple: (openai_params, adapter_params)
            - openai_params: dict with parameters for OpenAI API (includes model_id)
            - adapter_params: dict with adapter-specific parameters (remove_thinking_tags, etc.)
    """
    # Define adapter-specific parameters that should not be sent to OpenAI
    ADAPTER_PARAMS = {'remove_thinking_tags'}

    # Check for model-specific configuration in config
    models_config = CONFIG.get('models', [])

    # Resolve to original name
    original_name = resolve_model_name(model_id)

    # Find model entry by original name
    model_entry = None
    for model_config in models_config:
        if isinstance(model_config, dict) and model_config.get('name') == original_name:
            model_entry = model_config
            break

    if model_entry is None:
        # Model not in config, return empty parameters
        return {'model_id': original_name}, {}
    else:
        # Model found, split parameters into OpenAI and adapter params
        # CRITICAL: Exclude 'name' and 'custom_name' from both dicts
        openai_params = {}
        adapter_params = {}

        for k, v in model_entry.items():
            if k in ('name', 'custom_name'):
                # Skip these fields entirely
                continue
            elif k in ADAPTER_PARAMS:
                # Adapter-specific parameter
                adapter_params[k] = v
            else:
                # OpenAI API parameter
                openai_params[k] = v

        openai_params['model_id'] = original_name  # Always use original name
        return openai_params, adapter_params

def create_final_response(model_name, prompt_tokens, completion_tokens, total_duration_ns):
    """
    Helper function for creating final response in Ollama format.
    Updated according to Ollama API documentation.
    """
    return {
        "model": model_name,
        "created_at": datetime.now().isoformat() + "Z",
        "done": True,
        "prompt_eval_count": prompt_tokens,
        "eval_count": completion_tokens,
        "total_duration": total_duration_ns,
        "load_duration": 0, # OpenAI doesn't provide this info
        "prompt_eval_duration": 0, # OpenAI doesn't provide this info
        "eval_duration": int(total_duration_ns * 0.9) if total_duration_ns > 0 else 0 # Approximate value
    }

@app.route('/api/chat', methods=['POST'])
@log_endpoint
def chat():
    """
    Chat endpoint that forwards requests to OpenAI.
    Updated according to Ollama API documentation.
    """
    start_time = time.time()
    try:
        data, error = validate_json_request()
        if error:
            return error

        model_id, error = validate_model_parameter(data)
        if error:
            return error

        # Separate display name and OpenAI name
        display_name = model_id
        original_name = resolve_model_name(model_id)

        messages = data.get("messages")
        if not messages:
            return jsonify({"error": "Parameter 'messages' is required"}), 400

        if not isinstance(messages, list) or not messages:
            return jsonify({"error": "Parameter 'messages' must be a non-empty list"}), 400

        stream = data.get("stream", False)

        if stream:
            def generate_stream():
                completion_tokens = 0
                prompt_tokens = -1 # Will be determined later if OpenAI returns it

                try:
                    # Get model configuration
                    openai_params, adapter_params = get_model_config(model_id)

                    # Buffering for thinking tag removal
                    buffer = ""
                    tag_processed = False
                    remove_tags = adapter_params.get('remove_thinking_tags', False)

                    # Prepare OpenAI API parameters
                    api_params = {
                        'model': original_name,  # Use original name for OpenAI
                        'messages': messages,
                        'stream': True,
                        'stream_options': {"include_usage": True}
                    }

                    # Add all model-specific parameters from config
                    for key, value in openai_params.items():
                        if key != 'model_id':  # Skip our internal field
                            api_params[key] = value

                    response_stream = client.chat.completions.create(**api_params)
                    for chunk in response_stream:
                        if chunk.usage: # OpenAI may send usage in the last chunk
                            prompt_tokens = chunk.usage.prompt_tokens
                            completion_tokens = chunk.usage.completion_tokens

                        if not chunk.choices:
                            continue
                        content = chunk.choices[0].delta.content
                        if content:
                            if not tag_processed:
                                # Buffer content until we can detect thinking tags
                                buffer += content

                                # Check if we have enough content to detect tags
                                if len(buffer) >= 50 or '>' in buffer:
                                    # Process buffer for thinking tags
                                    processed_buffer = remove_thinking_tags(buffer, model_id, remove_tags)
                                    tag_processed = True

                                    # Yield processed buffer if not empty
                                    if processed_buffer:
                                        ollama_chunk = {
                                            "model": display_name,
                                            "created_at": datetime.now().isoformat() + "Z",
                                            "message": {"role": "assistant", "content": processed_buffer},
                                            "done": False
                                        }
                                        yield json.dumps(ollama_chunk) + '\n'
                            else:
                                # Tags already processed, yield content immediately
                                ollama_chunk = {
                                    "model": display_name,  # Return display name to client
                                    "created_at": datetime.now().isoformat() + "Z",
                                    "message": {"role": "assistant", "content": content},
                                    "done": False
                                }
                                yield json.dumps(ollama_chunk) + '\n'

                    # Handle case where response was too short to trigger tag processing
                    if not tag_processed and buffer:
                        processed_buffer = remove_thinking_tags(buffer, model_id, remove_tags)
                        if processed_buffer:
                            ollama_chunk = {
                                "model": display_name,
                                "created_at": datetime.now().isoformat() + "Z",
                                "message": {"role": "assistant", "content": processed_buffer},
                                "done": False
                            }
                            yield json.dumps(ollama_chunk) + '\n'

                    duration_ns = int((time.time() - start_time) * 1e9)
                    final_response = create_final_response(display_name, prompt_tokens if prompt_tokens != -1 else 0, completion_tokens, duration_ns)
                    final_response["message"] = {"role": "assistant", "content": ""}
                    yield json.dumps(final_response) + '\n'

                except Exception as e:
                    yield json.dumps({"error": f"Streaming error: {str(e)}"}) + '\n'

            return Response(stream_with_context(generate_stream()), mimetype='application/x-ndjson')
        else:
            openai_params, adapter_params = get_model_config(model_id)

            # Prepare OpenAI API parameters
            api_params = {
                'model': original_name,  # Use original name for OpenAI
                'messages': messages,
                'stream': False
            }

            # Add all model-specific parameters from config
            for key, value in openai_params.items():
                if key != 'model_id':  # Skip our internal field
                    api_params[key] = value

            response = client.chat.completions.create(**api_params)
            duration_ns = int((time.time() - start_time) * 1e9)

            final_response = create_final_response(
                display_name,  # Return display name to client
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                duration_ns
            )
            if not response.choices:
                return jsonify({"error": "No response choices returned from OpenAI"}), 500

            # Remove thinking tags if enabled
            raw_content = response.choices[0].message.content
            cleaned_content = remove_thinking_tags(
                raw_content,
                model_id,
                adapter_params.get('remove_thinking_tags', False)
            )

            final_response["message"] = {
                "role": "assistant",
                "content": cleaned_content
            }
            return jsonify(final_response)

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/generate', methods=['POST'])
@log_endpoint
def generate():
    """
    Ollama /api/generate endpoint - generates completions from a prompt.
    Converts to OpenAI completions API.
    """
    start_time = time.time()
    try:
        data, error = validate_json_request()
        if error:
            return error

        model_id, error = validate_model_parameter(data)
        if error:
            return error

        # Separate display name and OpenAI name
        display_name = model_id
        original_name = resolve_model_name(model_id)

        prompt = data.get("prompt")
        if not prompt:
            return jsonify({"error": "Parameter 'prompt' is required"}), 400

        if not isinstance(prompt, str) or not prompt.strip():
            return jsonify({"error": "Parameter 'prompt' must be a non-empty string"}), 400

        stream = data.get("stream", False)

        # Convert prompt to messages format for OpenAI
        messages = [{"role": "user", "content": prompt}]

        if stream:
            def generate_stream():
                completion_tokens = 0
                prompt_tokens = 0
                full_response = ""

                try:
                    # Get model configuration
                    openai_params, adapter_params = get_model_config(model_id)

                    # Buffering for thinking tag removal
                    buffer = ""
                    tag_processed = False
                    remove_tags = adapter_params.get('remove_thinking_tags', False)

                    # Prepare OpenAI API parameters
                    api_params = {
                        'model': original_name,  # Use original name for OpenAI
                        'messages': messages,
                        'stream': True,
                        'stream_options': {"include_usage": True}
                    }

                    # Add all model-specific parameters from config
                    for key, value in openai_params.items():
                        if key != 'model_id':  # Skip our internal field
                            api_params[key] = value

                    response_stream = client.chat.completions.create(**api_params)
                    for chunk in response_stream:
                        if chunk.usage:
                            prompt_tokens = chunk.usage.prompt_tokens
                            completion_tokens = chunk.usage.completion_tokens

                        if not chunk.choices:
                            continue

                        content = chunk.choices[0].delta.content
                        if content:
                            full_response += content

                            if not tag_processed:
                                # Buffer content until we can detect thinking tags
                                buffer += content

                                # Check if we have enough content to detect tags
                                if len(buffer) >= 50 or '>' in buffer:
                                    # Process buffer for thinking tags
                                    processed_buffer = remove_thinking_tags(buffer, model_id, remove_tags)
                                    tag_processed = True

                                    # Yield processed buffer if not empty
                                    if processed_buffer:
                                        ollama_chunk = {
                                            "model": display_name,
                                            "created_at": datetime.now().isoformat() + "Z",
                                            "response": processed_buffer,
                                            "done": False
                                        }
                                        yield json.dumps(ollama_chunk) + '\n'
                            else:
                                # Tags already processed, yield content immediately
                                ollama_chunk = {
                                    "model": display_name,  # BUGFIX: was undefined 'model', now use display_name
                                    "created_at": datetime.now().isoformat() + "Z",
                                    "response": content,
                                    "done": False
                                }
                                yield json.dumps(ollama_chunk) + '\n'

                    # Handle case where response was too short to trigger tag processing
                    if not tag_processed and buffer:
                        processed_buffer = remove_thinking_tags(buffer, model_id, remove_tags)
                        if processed_buffer:
                            ollama_chunk = {
                                "model": display_name,
                                "created_at": datetime.now().isoformat() + "Z",
                                "response": processed_buffer,
                                "done": False
                            }
                            yield json.dumps(ollama_chunk) + '\n'

                    duration_ns = int((time.time() - start_time) * 1e9)
                    final_response = create_final_response(display_name, prompt_tokens, completion_tokens, duration_ns)  # BUGFIX: was undefined 'model'
                    final_response["response"] = ""
                    yield json.dumps(final_response) + '\n'

                except Exception as e:
                    yield json.dumps({"error": f"Streaming error: {str(e)}"}) + '\n'

            return Response(stream_with_context(generate_stream()), mimetype='application/x-ndjson')
        else:
            openai_params, adapter_params = get_model_config(model_id)

            # Prepare OpenAI API parameters
            api_params = {
                'model': original_name,  # Use original name for OpenAI
                'messages': messages,
                'stream': False
            }

            # Add all model-specific parameters from config
            for key, value in openai_params.items():
                if key != 'model_id':  # Skip our internal field
                    api_params[key] = value

            response = client.chat.completions.create(**api_params)
            duration_ns = int((time.time() - start_time) * 1e9)

            if not response.choices:
                return jsonify({"error": "No response choices returned from OpenAI"}), 500

            final_response = create_final_response(
                display_name,  # Return display name to client
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                duration_ns
            )

            # Remove thinking tags if enabled
            raw_content = response.choices[0].message.content
            cleaned_content = remove_thinking_tags(
                raw_content,
                model_id,
                adapter_params.get('remove_thinking_tags', False)
            )

            final_response["response"] = cleaned_content
            return jsonify(final_response)

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/version', methods=['GET'])
@log_endpoint
def version():
    """Return version information."""
    return jsonify({"version": "0.1.0"})

@app.route('/api/ps', methods=['GET'])
@log_endpoint
def list_running_models():
    """List currently loaded models (mock implementation)."""
    return jsonify({"models": []})

@app.route('/api/embed', methods=['POST'])
@log_endpoint
def embed():
    """Generate embeddings using OpenAI embeddings API."""
    try:
        data, error = validate_json_request()
        if error:
            return error

        model, error = validate_model_parameter(data)
        if error:
            return error

        # Separate display name and OpenAI name
        display_name = model
        original_name = resolve_model_name(model)

        input_text = data.get("input")
        if not input_text:
            return jsonify({"error": "Parameter 'input' is required"}), 400

        if not isinstance(input_text, (str, list)):
            return jsonify({"error": "Parameter 'input' must be a string or list of strings"}), 400

        if isinstance(input_text, list) and not input_text:
            return jsonify({"error": "Parameter 'input' cannot be an empty list"}), 400

        if isinstance(input_text, str):
            input_text = [input_text]

        response = client.embeddings.create(
            model=original_name,  # Use resolved model name instead of hardcoded
            input=input_text
        )

        embeddings = [embedding.embedding for embedding in response.data]

        return jsonify({
            "model": display_name,  # Return display name to client
            "embeddings": embeddings,
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": response.usage.prompt_tokens if response.usage else 0
        })

    except Exception as e:
        return jsonify({"error": f"Embedding error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
@log_endpoint
def health_check():
    """Health check endpoint to verify service status."""
    try:
        # Test OpenAI client connection
        models = client.models.list()
        openai_status = "healthy" if models else "unhealthy"

        # Check if we have cached models
        cached_models_count = len(CACHED_MODELS)

        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat() + "Z",
            "services": {
                "openai": openai_status,
                "cached_models": cached_models_count
            },
            "version": "0.1.0"
        }), 200

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat() + "Z",
            "error": str(e)
        }), 503

@app.route('/', methods=['GET'])
@log_endpoint
def root():
    """Root endpoint with basic service information."""
    return jsonify({
        "service": "Ollama to OpenAI Adapter",
        "version": "0.1.0",
        "endpoints": [
            "/api/tags",
            "/api/show",
            "/api/chat",
            "/api/generate",
            "/api/embed",
            "/api/version",
            "/api/ps",
            "/health"
        ]
    })


if __name__ == '__main__':
    get_and_cache_models()

    server_config = CONFIG['server']
    logger.info(f"Starting Ollama -> OpenAI adapter on http://{server_config['host']}:{server_config['port']}")

    # Enable debug mode for auto-reload on code changes
    app.run(
        host=server_config['host'],
        port=server_config['port'],
        debug=True,
        use_reloader=True,
        reloader_type='stat'
    )
