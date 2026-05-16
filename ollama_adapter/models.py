"""Model configuration, resolution, caching, and prompt handling."""

from datetime import UTC, datetime
from typing import Any

from ollama_adapter import state
from ollama_adapter.prompt_renderer import render_file, render_inline

_EVAL_DURATION_RATIO = 0.9

_MODEL_DETAILS: dict[str, Any] = {
    "parent_model": "",
    "format": "gguf",
    "family": "llama",
    "families": ["llama"],
    "parameter_size": "8.0B",
    "quantization_level": "Q4_0",
}


def _normalize_prompt_value(value: Any) -> str | None:
    """Return stripped non-empty string, or None for missing/blank/non-string values."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _collect_prompt_vars(adapter_params: dict[str, Any]) -> dict[str, Any]:
    """Merge global `prompts.vars` with per-model `prompt_vars` (model overrides global)."""
    prompts_section = state.CONFIG.get("prompts") or {}
    global_vars = prompts_section.get("vars")
    merged: dict[str, Any] = dict(global_vars) if isinstance(global_vars, dict) else {}
    model_vars = adapter_params.get("prompt_vars")
    if isinstance(model_vars, dict):
        merged.update(model_vars)
    return merged


def _resolve_system_prompt(adapter_params: dict[str, Any], model_id: str) -> str | None:
    """Resolve and render the system prompt via the global Jinja2 environment.

    Raises `PromptRenderError` on rendering failure (propagated to caller).
    """
    inline = _normalize_prompt_value(adapter_params.get("system_prompt_inline"))
    file_path = _normalize_prompt_value(adapter_params.get("system_prompt_file"))

    if file_path and inline:
        state.logger.warning(
            "Model '%s': both 'system_prompt_inline' and 'system_prompt_file' are set; "
            "using 'system_prompt_file' ('%s') and ignoring inline value",
            model_id,
            file_path,
        )

    if not file_path and not inline:
        return None

    env = state.jinja_env
    assert env is not None, "Jinja2 environment is not initialized"  # noqa: S101
    variables = _collect_prompt_vars(adapter_params)

    if file_path:
        rendered = render_file(env, file_path, variables)
    else:
        assert inline is not None  # noqa: S101
        rendered = render_inline(env, inline, variables)

    stripped = rendered.strip()
    return stripped or None


def apply_system_prompt(
    messages: list[dict[str, Any]], adapter_params: dict[str, Any], model_id: str
) -> list[dict[str, Any]]:
    """Apply rendered system prompt from model config to the messages list.

    Raises `PromptRenderError` on rendering failure; callers translate it into
    a client-facing assistant message.
    """
    resolved = _resolve_system_prompt(adapter_params, model_id)
    if not resolved:
        return messages

    result = list(messages)
    system_msg = {"role": "system", "content": resolved}

    system_idx = None
    for i, msg in enumerate(result):
        if isinstance(msg, dict) and msg.get("role") == "system":
            system_idx = i
            break

    if system_idx is not None:
        result[system_idx] = system_msg
        state.logger.debug("Replaced system message for model '%s' with config system_prompt", model_id)
    else:
        result.insert(0, system_msg)
        state.logger.debug("Prepended system message for model '%s' from config system_prompt", model_id)

    return result


def apply_prompt_caching(
    messages: list[dict[str, Any]], adapter_params: dict[str, Any], model_id: str
) -> list[dict[str, Any]]:
    """Add cache_control markers to system message content for provider-side prompt caching.

    Enable prompt caching on Anthropic and Google Gemini via LiteLLM.
    """
    if not adapter_params.get("prompt_caching"):
        return messages

    result = list(messages)

    for i, msg in enumerate(result):
        if not isinstance(msg, dict) or msg.get("role") != "system":
            continue

        content = msg.get("content")
        if not content or not isinstance(content, str):
            continue

        result[i] = {
            "role": "system",
            "content": [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}],
        }
        state.logger.debug("Added cache_control to system message for model '%s' (%d chars)", model_id, len(content))
        break

    return result


def get_display_name(original_name: str) -> str:
    """Return custom_name if set, otherwise original_name."""
    models_config = state.CONFIG.get("models", [])
    for model in models_config:
        if isinstance(model, dict) and model.get("name") == original_name and "custom_name" in model:
            return str(model["custom_name"])
    return original_name


def resolve_model_name(client_name: str) -> str:
    """Resolve model name from client to original OpenAI name.

    Work with both custom_name and original names as synonyms.
    """
    models_config = state.CONFIG.get("models", [])

    for model in models_config:
        if isinstance(model, dict) and model.get("custom_name") == client_name:
            return str(model["name"])

    return client_name


def get_and_cache_models(*, force_refresh: bool = False) -> list[dict[str, Any]]:
    """Fetch, filter, map and cache model list from OpenAI API."""
    if state.CACHED_MODELS and not force_refresh:
        return state.CACHED_MODELS

    action = "Refreshing" if force_refresh else "Requesting"
    state.logger.info("Model cache: %s model list from OpenAI...", action)
    try:
        assert state.client is not None  # noqa: S101
        all_models_response = state.client.models.list().data
        models_config = state.CONFIG.get("models", [])
        new_models = _build_model_list(all_models_response, models_config)
    except Exception:  # noqa: BLE001
        state.logger.exception("Critical error getting models from OpenAI")
        if force_refresh:
            state.logger.warning("Keeping previous model cache after refresh failure")
            return state.CACHED_MODELS
        return []
    else:
        state.CACHED_MODELS = new_models
        state.logger.info("Models successfully loaded and cached. Found: %d", len(state.CACHED_MODELS))
        return state.CACHED_MODELS


def _build_model_list(all_models_response: Any, models_config: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the Ollama-format model list from OpenAI models."""
    if models_config:
        return _build_filtered_model_list(all_models_response, models_config)
    return [
        {
            "name": model.id,
            "model": model.id,
            "modified_at": datetime.fromtimestamp(model.created, tz=UTC).isoformat(),
            "size": 0,
            "digest": model.id,
            "details": _MODEL_DETAILS,
        }
        for model in all_models_response
    ]


def _build_filtered_model_list(all_models_response: Any, models_config: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build model list filtered by configured model entries."""
    openai_models_by_id = {m.id: m for m in all_models_response}
    new_models: list[dict[str, Any]] = []

    for model_entry in models_config:
        model_name = model_entry.get("name") if isinstance(model_entry, dict) else None
        if not model_name or model_name not in openai_models_by_id:
            continue
        openai_model = openai_models_by_id[model_name]
        display_name = model_entry.get("custom_name", model_name)
        new_models.append(
            {
                "name": display_name,
                "model": display_name,
                "modified_at": datetime.fromtimestamp(openai_model.created, tz=UTC).isoformat(),
                "size": 0,
                "digest": model_name,
                "details": _MODEL_DETAILS,
            }
        )

    return new_models


def resolve_ip_list(ip_value: str) -> list[str]:
    """Resolve an ip_routing 'ip' field to a list of IP addresses.

    If ip_value matches a key in CONFIG['clients'], return that client group's IPs.
    Otherwise treat ip_value as a direct IP address.
    """
    clients: dict[str, Any] = state.CONFIG.get("clients", {}) or {}
    if ip_value in clients:
        client_ips = clients[ip_value]
        return [client_ips] if isinstance(client_ips, str) else list(client_ips)
    return [ip_value]


def apply_ip_routing(model_entry: dict[str, Any], client_ip: str) -> dict[str, Any]:
    """Apply IP-based routing overrides to a model config entry.

    Return a new dict with overrides merged; does not mutate the original.
    Scalar fields are replaced. Dict fields (params, headers) are shallow-merged.
    """
    ip_routing = model_entry.get("ip_routing")
    if not ip_routing or not client_ip:
        return model_entry

    matching_rule = None
    for rule in ip_routing:
        resolved_ips = resolve_ip_list(rule.get("ip", ""))
        if client_ip in resolved_ips:
            matching_rule = rule
            break

    if matching_rule is None:
        return model_entry

    merged = dict(model_entry)
    merged.pop("ip_routing", None)

    for key in ("name", "system_prompt_inline", "system_prompt_file", "remove_thinking_tags", "prompt_caching"):
        if key in matching_rule:
            merged[key] = matching_rule[key]

    for key in ("params", "headers", "prompt_vars"):
        if key in matching_rule:
            parent_dict = dict(model_entry.get(key, {}) or {})
            override_dict = matching_rule[key]
            if override_dict is not None:
                parent_dict.update(override_dict)
            merged[key] = parent_dict

    state.logger.info(
        "IP routing applied: client_ip=%s, model='%s', routed_to='%s'",
        client_ip,
        model_entry.get("custom_name") or model_entry.get("name"),
        merged.get("name"),
    )

    return merged


def get_model_config(
    model_id: str, *, client_ip: str | None = None
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Return model configuration split into OpenAI params, adapter params, and headers.

    If client_ip is provided, apply IP-based routing overrides.
    """
    models_config = state.CONFIG.get("models", [])
    original_name = resolve_model_name(model_id)

    model_entry = _find_model_entry(models_config, model_id, original_name)

    if model_entry is None:
        return {"model_id": original_name}, {}, {}

    if client_ip:
        model_entry = apply_ip_routing(model_entry, client_ip)

    adapter_params: dict[str, Any] = {}
    for adapter_key in (
        "remove_thinking_tags",
        "system_prompt_inline",
        "system_prompt_file",
        "prompt_caching",
        "prompt_vars",
    ):
        if adapter_key in model_entry:
            adapter_params[adapter_key] = model_entry[adapter_key]

    openai_params: dict[str, Any] = dict(model_entry.get("params", {}) or {})
    openai_params["model_id"] = model_entry.get("name", original_name)

    headers: dict[str, str] = dict(model_entry.get("headers", {}) or {})

    return openai_params, adapter_params, headers


def _find_model_entry(models_config: list[dict[str, Any]], model_id: str, original_name: str) -> dict[str, Any] | None:
    """Find model entry by custom_name first, then by original name."""
    for model_config in models_config:
        if isinstance(model_config, dict) and model_config.get("custom_name") == model_id:
            return model_config

    for model_config in models_config:
        if isinstance(model_config, dict) and model_config.get("name") == original_name:
            return model_config

    return None


def create_final_response(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_duration_ns: int,
) -> dict[str, Any]:
    """Create final response dict in Ollama format."""
    return {
        "model": model_name,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "done": True,
        "prompt_eval_count": prompt_tokens,
        "eval_count": completion_tokens,
        "total_duration": total_duration_ns,
        "load_duration": 0,
        "prompt_eval_duration": 0,
        "eval_duration": int(total_duration_ns * _EVAL_DURATION_RATIO) if total_duration_ns > 0 else 0,
    }
