import os
from datetime import datetime

from ollama_adapter import state


def resolve_system_prompt(value):
    """
    Resolve system_prompt value to actual prompt text.
    If value ends with '.md', reads content from file (relative to CWD).
    Otherwise returns the string as-is.
    """
    if not value or not isinstance(value, str):
        return None

    value = value.strip()
    if not value:
        return None

    if value.endswith('.md'):
        file_path = value if os.path.isabs(value) else os.path.join(os.getcwd(), value)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        return content if content else None

    return value


def apply_system_prompt(messages, adapter_params, model_id):
    """
    Apply system prompt from model config to messages list.
    If config has system_prompt and request has system message — replaces it.
    If config has system_prompt and request has no system message — prepends it.
    If config has no system_prompt — returns messages unchanged.
    """
    system_prompt_value = adapter_params.get('system_prompt')
    if not system_prompt_value:
        return messages

    try:
        resolved = resolve_system_prompt(system_prompt_value)
    except FileNotFoundError:
        state.logger.error(f"System prompt file not found for model '{model_id}': {system_prompt_value}")
        return messages
    except Exception as e:
        state.logger.error(f"Failed to read system prompt for model '{model_id}': {e}")
        return messages

    if not resolved:
        return messages

    result = list(messages)
    system_msg = {"role": "system", "content": resolved}

    system_idx = None
    for i, msg in enumerate(result):
        if isinstance(msg, dict) and msg.get('role') == 'system':
            system_idx = i
            break

    if system_idx is not None:
        result[system_idx] = system_msg
        state.logger.debug(f"Replaced system message for model '{model_id}' with config system_prompt")
    else:
        result.insert(0, system_msg)
        state.logger.debug(f"Prepended system message for model '{model_id}' from config system_prompt")

    return result


def apply_prompt_caching(messages, adapter_params, model_id):
    """
    Add cache_control markers to system message content for provider-side prompt caching.
    Enables prompt caching on Anthropic and Google Gemini via LiteLLM.
    """
    if not adapter_params.get('prompt_caching'):
        return messages

    result = list(messages)

    for i, msg in enumerate(result):
        if not isinstance(msg, dict) or msg.get('role') != 'system':
            continue

        content = msg.get('content')
        if not content or not isinstance(content, str):
            continue

        result[i] = {
            "role": "system",
            "content": [{
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"}
            }]
        }
        state.logger.debug(
            f"Added cache_control to system message for model '{model_id}' "
            f"({len(content)} chars)"
        )
        break

    return result


def get_display_name(original_name):
    """Returns custom_name if set, otherwise original_name."""
    models_config = state.CONFIG.get('models', [])
    for model in models_config:
        if isinstance(model, dict) and model.get('name') == original_name and 'custom_name' in model:
            return model['custom_name']
    return original_name


def resolve_model_name(client_name):
    """
    Resolves model name from client to original OpenAI name.
    Works with both custom_name and original names as synonyms.
    """
    models_config = state.CONFIG.get('models', [])

    for model in models_config:
        if isinstance(model, dict) and model.get('custom_name') == client_name:
            return model['name']

    return client_name


def get_and_cache_models(force_refresh=False):
    """
    Fetches, filters, maps and caches model list.
    Updated according to Ollama API documentation.
    """
    if state.CACHED_MODELS and not force_refresh:
        return state.CACHED_MODELS

    action = "Refreshing" if force_refresh else "Requesting"
    state.logger.info(f"Model cache: {action} model list from OpenAI...")
    try:
        all_models_response = state.client.models.list().data
        models_config = state.CONFIG.get('models', [])

        model_details = {
            "parent_model": "",
            "format": "gguf",
            "family": "llama",
            "families": ["llama"],
            "parameter_size": "8.0B",
            "quantization_level": "Q4_0",
        }

        new_models = []
        if models_config:
            openai_models_by_id = {m.id: m for m in all_models_response}

            for model_entry in models_config:
                model_name = model_entry.get('name') if isinstance(model_entry, dict) else None
                if not model_name or model_name not in openai_models_by_id:
                    continue
                openai_model = openai_models_by_id[model_name]
                display_name = model_entry.get('custom_name', model_name)
                new_models.append({
                    "name": display_name,
                    "model": display_name,
                    "modified_at": datetime.fromtimestamp(openai_model.created).isoformat() + "Z",
                    "size": 0,
                    "digest": model_name,
                    "details": model_details,
                })
        else:
            for model in all_models_response:
                new_models.append({
                    "name": model.id,
                    "model": model.id,
                    "modified_at": datetime.fromtimestamp(model.created).isoformat() + "Z",
                    "size": 0,
                    "digest": model.id,
                    "details": model_details,
                })

        state.CACHED_MODELS = new_models
        state.logger.info(f"Models successfully loaded and cached. Found: {len(state.CACHED_MODELS)}")
        return state.CACHED_MODELS
    except Exception as e:
        state.logger.error(f"Critical error getting models from OpenAI: {e}")
        if force_refresh:
            state.logger.warning("Keeping previous model cache after refresh failure")
            return state.CACHED_MODELS
        return []


def resolve_ip_list(ip_value):
    """Resolve an ip_routing 'ip' field to a list of IP addresses.
    If ip_value matches a key in CONFIG['clients'], returns that client group's IPs.
    Otherwise treats ip_value as a direct IP address."""
    clients = state.CONFIG.get('clients', {}) or {}
    if isinstance(ip_value, str) and ip_value in clients:
        client_ips = clients[ip_value]
        if isinstance(client_ips, str):
            return [client_ips]
        return list(client_ips)
    if isinstance(ip_value, str):
        return [ip_value]
    return []


def apply_ip_routing(model_entry, client_ip):
    """Apply IP-based routing overrides to a model config entry.
    Returns a new dict with overrides merged; does not mutate the original.
    Scalar fields are replaced. Dict fields (params, headers) are shallow-merged."""
    ip_routing = model_entry.get('ip_routing')
    if not ip_routing or not client_ip:
        return model_entry

    matching_rule = None
    for rule in ip_routing:
        resolved_ips = resolve_ip_list(rule.get('ip', ''))
        if client_ip in resolved_ips:
            matching_rule = rule
            break

    if matching_rule is None:
        return model_entry

    merged = dict(model_entry)
    merged.pop('ip_routing', None)

    for key in ('name', 'system_prompt', 'remove_thinking_tags', 'prompt_caching'):
        if key in matching_rule:
            merged[key] = matching_rule[key]

    for key in ('params', 'headers'):
        if key in matching_rule:
            parent_dict = dict(model_entry.get(key, {}) or {})
            override_dict = matching_rule[key]
            if override_dict is not None:
                parent_dict.update(override_dict)
            merged[key] = parent_dict

    state.logger.info(
        f"IP routing applied: client_ip={client_ip}, "
        f"model='{model_entry.get('custom_name') or model_entry.get('name')}', "
        f"routed_to='{merged.get('name')}'"
    )

    return merged


def get_model_config(model_id, client_ip=None):
    """
    Returns model configuration split into OpenAI params, adapter params, and headers.
    If client_ip is provided, applies IP-based routing overrides.
    """
    models_config = state.CONFIG.get('models', [])

    original_name = resolve_model_name(model_id)

    # Find model entry: prefer match by custom_name, then fall back to original name
    model_entry = None
    for model_config in models_config:
        if isinstance(model_config, dict) and model_config.get('custom_name') == model_id:
            model_entry = model_config
            break

    if model_entry is None:
        for model_config in models_config:
            if isinstance(model_config, dict) and model_config.get('name') == original_name:
                model_entry = model_config
                break

    if model_entry is None:
        return {'model_id': original_name}, {}, {}

    if client_ip:
        model_entry = apply_ip_routing(model_entry, client_ip)

    adapter_params = {}
    if 'remove_thinking_tags' in model_entry:
        adapter_params['remove_thinking_tags'] = model_entry['remove_thinking_tags']
    if 'system_prompt' in model_entry:
        adapter_params['system_prompt'] = model_entry['system_prompt']
    if 'prompt_caching' in model_entry:
        adapter_params['prompt_caching'] = model_entry['prompt_caching']

    openai_params = dict(model_entry.get('params', {}) or {})
    openai_params['model_id'] = model_entry.get('name', original_name)

    headers = dict(model_entry.get('headers', {}) or {})

    return openai_params, adapter_params, headers


def create_final_response(model_name, prompt_tokens, completion_tokens, total_duration_ns):
    """Helper function for creating final response in Ollama format."""
    return {
        "model": model_name,
        "created_at": datetime.now().isoformat() + "Z",
        "done": True,
        "prompt_eval_count": prompt_tokens,
        "eval_count": completion_tokens,
        "total_duration": total_duration_ns,
        "load_duration": 0,
        "prompt_eval_duration": 0,
        "eval_duration": int(total_duration_ns * 0.9) if total_duration_ns > 0 else 0
    }
