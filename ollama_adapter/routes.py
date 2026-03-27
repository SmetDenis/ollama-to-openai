import time
import json
from datetime import datetime
from flask import Blueprint, request, jsonify, Response, stream_with_context

from ollama_adapter import state
from ollama_adapter.logging_utils import (
    log_endpoint, validate_json_request, validate_model_parameter, get_client_ip,
)
from ollama_adapter.tracing import (
    build_trace_headers, build_trace_body_metadata,
    _tracing_log_headers_enabled, _capture_litellm_headers, _log_litellm_headers,
)
from ollama_adapter.thinking import remove_thinking_tags, _process_stream
from ollama_adapter.models import (
    resolve_model_name, get_and_cache_models, get_model_config,
    create_final_response, apply_system_prompt, apply_prompt_caching,
)

bp = Blueprint('api', __name__)


# --- Shared completion helper for chat() and generate() ---

def _call_openai_streaming(model_id, display_name, original_name, messages, start_time, make_chunk, response_key):
    """Streaming completion call shared by chat() and generate().
    Returns a generator that yields JSON lines."""
    client_ip = get_client_ip()

    def generate_stream():
        _raw_ctx = None
        try:
            openai_params, adapter_params, extra_headers = get_model_config(model_id, client_ip=client_ip)

            api_params = {
                'model': openai_params.get('model_id', original_name),
                'messages': messages,
                'stream': True,
                'stream_options': {"include_usage": True}
            }

            extra_body = {k: v for k, v in openai_params.items() if k != 'model_id'}

            api_params['messages'] = apply_system_prompt(
                api_params['messages'], adapter_params, model_id
            )
            api_params['messages'] = apply_prompt_caching(
                api_params['messages'], adapter_params, model_id
            )

            merged_headers = build_trace_headers(extra_headers, display_name) or None
            trace_meta = build_trace_body_metadata(display_name)
            if trace_meta:
                extra_body['metadata'] = trace_meta

            if _tracing_log_headers_enabled():
                _raw_ctx = state.client.chat.completions.with_streaming_response.create(
                    **api_params,
                    extra_body=extra_body or None,
                    extra_headers=merged_headers
                )
                _raw_response = _raw_ctx.__enter__()
                _capture_litellm_headers(_raw_response.headers)
                _log_litellm_headers()
                response_stream = _raw_response.parse()
            else:
                response_stream = state.client.chat.completions.create(
                    **api_params,
                    extra_body=extra_body or None,
                    extra_headers=merged_headers
                )

            remove_tags = adapter_params.get('remove_thinking_tags', False)
            usage = {'prompt_tokens': 0, 'completion_tokens': 0}

            yield from _process_stream(
                response_stream, remove_tags, model_id,
                display_name, make_chunk, usage
            )

            duration_ns = int((time.time() - start_time) * 1e9)
            final_response = create_final_response(
                display_name,
                usage['prompt_tokens'],
                usage['completion_tokens'],
                duration_ns
            )

            if response_key == "message":
                final_response["message"] = {"role": "assistant", "content": ""}
            else:
                final_response["response"] = ""

            yield json.dumps(final_response) + '\n'

        except Exception as e:
            yield json.dumps({"error": f"Streaming error: {str(e)}"}) + '\n'
        finally:
            if _raw_ctx is not None:
                _raw_ctx.__exit__(None, None, None)

    return Response(stream_with_context(generate_stream()), mimetype='application/x-ndjson')


def _call_openai_non_streaming(model_id, display_name, original_name, messages, start_time, response_key):
    """Non-streaming completion call shared by chat() and generate()."""
    client_ip = get_client_ip()
    openai_params, adapter_params, extra_headers = get_model_config(model_id, client_ip=client_ip)

    api_params = {
        'model': openai_params.get('model_id', original_name),
        'messages': messages,
        'stream': False
    }

    extra_body = {k: v for k, v in openai_params.items() if k != 'model_id'}

    api_params['messages'] = apply_system_prompt(
        api_params['messages'], adapter_params, model_id
    )
    api_params['messages'] = apply_prompt_caching(
        api_params['messages'], adapter_params, model_id
    )

    merged_headers = build_trace_headers(extra_headers, display_name) or None
    trace_meta = build_trace_body_metadata(display_name)
    if trace_meta:
        extra_body['metadata'] = trace_meta

    if _tracing_log_headers_enabled():
        raw = state.client.chat.completions.with_raw_response.create(
            **api_params,
            extra_body=extra_body or None,
            extra_headers=merged_headers
        )
        response = raw.parse()
        _capture_litellm_headers(raw.headers)
        _log_litellm_headers()
    else:
        response = state.client.chat.completions.create(
            **api_params,
            extra_body=extra_body or None,
            extra_headers=merged_headers
        )

    duration_ns = int((time.time() - start_time) * 1e9)

    if not response.choices:
        return jsonify({"error": "No response choices returned from OpenAI"}), 500

    final_response = create_final_response(
        display_name,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        duration_ns
    )

    raw_content = response.choices[0].message.content
    cleaned_content = remove_thinking_tags(
        raw_content,
        model_id,
        adapter_params.get('remove_thinking_tags', False)
    )

    if response_key == "message":
        final_response["message"] = {"role": "assistant", "content": cleaned_content}
    else:
        final_response["response"] = cleaned_content

    return jsonify(final_response)


# --- Endpoints ---

@bp.route('/api/tags', methods=['GET', 'POST'])
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

        original_name = resolve_model_name(model_identifier)
        found_model = next((
            m for m in models
            if m['name'] == model_identifier
            or m['model'] == model_identifier
            or m['digest'] == original_name
        ), None)
        return jsonify({"models": [found_model]}) if found_model else (jsonify({"error": "Model not found"}), 404)


@bp.route('/api/show', methods=['POST'])
@log_endpoint
def show_model():
    """Emulates /api/show endpoint."""
    data, error = validate_json_request()
    if error:
        return error

    model_id, error = validate_model_parameter(data)
    if error:
        return error

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


@bp.route('/api/chat', methods=['POST'])
@log_endpoint
def chat():
    """Chat endpoint that forwards requests to OpenAI."""
    start_time = time.time()
    try:
        data, error = validate_json_request()
        if error:
            return error

        model_id, error = validate_model_parameter(data)
        if error:
            return error

        display_name = model_id
        original_name = resolve_model_name(model_id)

        messages = data.get("messages")
        if not messages:
            return jsonify({"error": "Parameter 'messages' is required"}), 400

        if not isinstance(messages, list) or not messages:
            return jsonify({"error": "Parameter 'messages' must be a non-empty list"}), 400

        stream = data.get("stream", False)

        if stream:
            def make_chat_chunk(dn, content):
                return {
                    "model": dn,
                    "created_at": datetime.now().isoformat() + "Z",
                    "message": {"role": "assistant", "content": content},
                    "done": False
                }
            return _call_openai_streaming(
                model_id, display_name, original_name, messages,
                start_time, make_chat_chunk, "message"
            )
        else:
            return _call_openai_non_streaming(
                model_id, display_name, original_name, messages,
                start_time, "message"
            )

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@bp.route('/api/generate', methods=['POST'])
@log_endpoint
def generate():
    """Ollama /api/generate endpoint - generates completions from a prompt."""
    start_time = time.time()
    try:
        data, error = validate_json_request()
        if error:
            return error

        model_id, error = validate_model_parameter(data)
        if error:
            return error

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

        # Handle Ollama's 'system' parameter from request
        request_system = data.get("system")
        if request_system and isinstance(request_system, str) and request_system.strip():
            messages.insert(0, {"role": "system", "content": request_system.strip()})

        if stream:
            def make_generate_chunk(dn, content):
                return {
                    "model": dn,
                    "created_at": datetime.now().isoformat() + "Z",
                    "response": content,
                    "done": False
                }
            return _call_openai_streaming(
                model_id, display_name, original_name, messages,
                start_time, make_generate_chunk, "response"
            )
        else:
            return _call_openai_non_streaming(
                model_id, display_name, original_name, messages,
                start_time, "response"
            )

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@bp.route('/api/version', methods=['GET'])
@log_endpoint
def version():
    """Return version information."""
    return jsonify({"version": "0.1.0"})


@bp.route('/api/ps', methods=['GET'])
@log_endpoint
def list_running_models():
    """List currently loaded models (mock implementation)."""
    return jsonify({"models": []})


@bp.route('/api/embed', methods=['POST'])
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

        display_name = model
        original_name = resolve_model_name(model)
        client_ip = get_client_ip()

        openai_params, _, extra_headers = get_model_config(model, client_ip=client_ip)
        resolved_model = openai_params.get('model_id', original_name)

        input_text = data.get("input")
        if not input_text:
            return jsonify({"error": "Parameter 'input' is required"}), 400

        if not isinstance(input_text, (str, list)):
            return jsonify({"error": "Parameter 'input' must be a string or list of strings"}), 400

        if isinstance(input_text, list) and not input_text:
            return jsonify({"error": "Parameter 'input' cannot be an empty list"}), 400

        if isinstance(input_text, str):
            input_text = [input_text]

        merged_headers = build_trace_headers(extra_headers, display_name) or None

        if _tracing_log_headers_enabled():
            raw = state.client.embeddings.with_raw_response.create(
                model=resolved_model,
                input=input_text,
                extra_headers=merged_headers
            )
            response = raw.parse()
            _capture_litellm_headers(raw.headers)
            _log_litellm_headers()
        else:
            response = state.client.embeddings.create(
                model=resolved_model,
                input=input_text,
                extra_headers=merged_headers
            )

        embeddings = [embedding.embedding for embedding in response.data]

        return jsonify({
            "model": display_name,
            "embeddings": embeddings,
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": response.usage.prompt_tokens if response.usage else 0
        })

    except Exception as e:
        return jsonify({"error": f"Embedding error: {str(e)}"}), 500


@bp.route('/health', methods=['GET'])
@log_endpoint
def health_check():
    """Health check endpoint to verify service status."""
    try:
        models = state.client.with_options(timeout=5.0).models.list()
        openai_status = "healthy" if models else "unhealthy"

        cached_models_count = len(state.CACHED_MODELS)

        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat() + "Z",
            "services": {
                "openai": openai_status,
                "cached_models": cached_models_count
            },
            "last_config_reload": state._last_config_reload_time,
            "version": "0.1.0"
        }), 200

    except Exception as e:
        state.logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat() + "Z",
            "error": str(e)
        }), 503


@bp.route('/', methods=['GET'])
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
