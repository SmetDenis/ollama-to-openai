"""Configuration loading, validation, and hot-reload logic."""

import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

from ollama_adapter import state
from ollama_adapter.logging_utils import TraceContextFilter


def _validate_required_fields(config: dict[str, Any]) -> None:
    """Validate required top-level configuration fields exist."""
    if not config.get("openai", {}).get("api_key"):
        msg = "Missing required parameter 'openai.api_key' in config.yml"
        raise ValueError(msg)
    if not config.get("server", {}).get("host"):
        msg = "Missing required parameter 'server.host' in config.yml"
        raise ValueError(msg)
    if not config.get("server", {}).get("port"):
        msg = "Missing required parameter 'server.port' in config.yml"
        raise ValueError(msg)


def _normalize_model_names(models_config: list[dict[str, Any]]) -> None:
    """Strip whitespace from custom_name values in-place."""
    for model in models_config:
        if isinstance(model, dict) and "custom_name" in model:
            model["custom_name"] = model["custom_name"].strip()


def _validate_custom_name_uniqueness(models_config: list[dict[str, Any]]) -> None:
    """Ensure all custom_name values are unique across models."""
    custom_names = [
        model["custom_name"] for model in models_config if isinstance(model, dict) and "custom_name" in model
    ]
    duplicates = [name for name in set(custom_names) if custom_names.count(name) > 1]
    if duplicates:
        msg = f"Duplicate custom_name values found: {duplicates}"
        raise ValueError(msg)


def _validate_clients(clients_config: Any) -> None:
    """Validate the clients section: alias -> IP mappings."""
    if not isinstance(clients_config, dict):
        msg = "'clients' must be a dict mapping alias names to IP addresses"
        raise TypeError(msg)
    for alias, ips in clients_config.items():
        if isinstance(ips, str):
            if not ips.strip():
                msg = f"Client alias '{alias}': IP address cannot be empty"
                raise ValueError(msg)
            clients_config[alias] = ips.strip()
        elif isinstance(ips, list):
            if not ips:
                msg = f"Client alias '{alias}': IP list cannot be empty"
                raise ValueError(msg)
            for i, ip in enumerate(ips):
                if not isinstance(ip, str) or not ip.strip():
                    msg = f"Client alias '{alias}': ip[{i}] must be a non-empty string"
                    raise ValueError(msg)
            clients_config[alias] = [ip.strip() for ip in ips]
        else:
            msg = f"Client alias '{alias}': value must be a string or list of strings"
            raise TypeError(msg)


def _validate_single_ip_route(
    rule: Any,
    rule_idx: int,
    model_display: str,
    clients_config: dict[str, Any] | None,
    seen_ips: set[str],
) -> None:
    """Validate a single ip_routing rule entry."""
    if not isinstance(rule, dict):
        msg = f"Model '{model_display}': ip_routing[{rule_idx}] must be a dict"
        raise TypeError(msg)

    ip_value = rule.get("ip")
    if not ip_value or not isinstance(ip_value, str) or not ip_value.strip():
        msg = (
            f"Model '{model_display}': ip_routing[{rule_idx}] "
            f"must have a non-empty 'ip' field (alias name or direct IP)"
        )
        raise ValueError(msg)

    ip_value = ip_value.strip()
    rule["ip"] = ip_value

    resolved_ips = _resolve_ips_for_validation(ip_value, clients_config)
    for ip in resolved_ips:
        if ip in seen_ips:
            msg = f"Model '{model_display}': duplicate IP '{ip}' in ip_routing"
            raise ValueError(msg)
        seen_ips.add(ip)

    for dict_key in ("params", "headers"):
        if dict_key in rule and rule[dict_key] is not None and not isinstance(rule[dict_key], dict):
            msg = f"Model '{model_display}': ip_routing[{rule_idx}] '{dict_key}' must be a dict"
            raise TypeError(msg)

    recognized_keys = {
        "ip",
        "name",
        "system_prompt_inline",
        "system_prompt_file",
        "remove_thinking_tags",
        "prompt_caching",
        "params",
        "headers",
    }
    unknown_keys = set(rule.keys()) - recognized_keys
    if unknown_keys:
        state.logger.warning(
            "Model '%s': ip_routing[%d] has unrecognized keys %s. "
            "These will be ignored. Did you mean to put them inside 'params'?",
            model_display,
            rule_idx,
            unknown_keys,
        )


def _resolve_ips_for_validation(ip_value: str, clients_config: dict[str, Any] | None) -> list[str]:
    """Resolve an IP value to a list of IPs for duplicate checking."""
    if clients_config and ip_value in clients_config:
        resolved = clients_config[ip_value]
        return [resolved] if isinstance(resolved, str) else list(resolved)
    return [ip_value]


def _check_system_prompt_keys(entry: dict[str, Any], model_display: str, location: str) -> None:
    """Warn about deprecated `system_prompt` key and inline/file conflict.

    Drops the deprecated key from `entry` so it does not propagate further.
    """
    if "system_prompt" in entry:
        state.logger.warning(
            "Model '%s'%s: key 'system_prompt' is deprecated and ignored. "
            "Use 'system_prompt_inline' for an inline string or 'system_prompt_file' for a file path.",
            model_display,
            location,
        )
        del entry["system_prompt"]

    inline = entry.get("system_prompt_inline")
    file_path = entry.get("system_prompt_file")
    if isinstance(inline, str) and inline.strip() and isinstance(file_path, str) and file_path.strip():
        state.logger.warning(
            "Model '%s'%s: both 'system_prompt_inline' and 'system_prompt_file' are set; "
            "'system_prompt_file' will take precedence at request time.",
            model_display,
            location,
        )


def _validate_model_entries(models_config: list[Any]) -> None:
    """Run per-model checks for system prompt fields at root and ip_routing levels."""
    for idx, model in enumerate(models_config):
        if not isinstance(model, dict):
            continue
        model_display = model.get("custom_name") or model.get("name", f"models[{idx}]")
        _check_system_prompt_keys(model, model_display, "")

        ip_routing = model.get("ip_routing")
        if not isinstance(ip_routing, list):
            continue
        for rule_idx, rule in enumerate(ip_routing):
            if isinstance(rule, dict):
                _check_system_prompt_keys(rule, model_display, f" ip_routing[{rule_idx}]")


def _validate_ip_routing(models_config: list[Any], clients_config: dict[str, Any] | None) -> None:
    """Validate ip_routing entries in all model configurations."""
    for idx, model in enumerate(models_config):
        if not isinstance(model, dict):
            continue
        ip_routing = model.get("ip_routing")
        if ip_routing is None:
            continue

        model_display = model.get("custom_name") or model.get("name", f"models[{idx}]")

        if not isinstance(ip_routing, list):
            msg = f"Model '{model_display}': 'ip_routing' must be a list"
            raise TypeError(msg)

        seen_ips: set[str] = set()
        for rule_idx, rule in enumerate(ip_routing):
            _validate_single_ip_route(rule, rule_idx, model_display, clients_config, seen_ips)


def _validate_tracing(tracing_config: Any) -> None:
    """Validate the tracing configuration section."""
    if not isinstance(tracing_config, dict):
        msg = "'tracing' must be a dict"
        raise TypeError(msg)
    for flag in ("enabled", "log_headers", "send_trace_headers"):
        if flag in tracing_config and not isinstance(tracing_config[flag], bool):
            msg = f"tracing.{flag} must be a boolean (true/false)"
            raise ValueError(msg)
    for field in ("trace_id_prefix", "trace_name_prefix", "tags", "timezone"):
        if field in tracing_config and not isinstance(tracing_config[field], str):
            msg = f"tracing.{field} must be a string"
            raise ValueError(msg)
    if "trace_grouping" in tracing_config:
        allowed = ("hourly", "daily")
        val = tracing_config["trace_grouping"]
        if not isinstance(val, str) or val not in allowed:
            msg = f"tracing.trace_grouping must be one of {allowed}"
            raise ValueError(msg)


def load_config(path: str = "config.yml") -> dict[str, Any]:
    """Load and validate YAML configuration file.

    Raise ValueError/TypeError on validation errors.
    """
    with Path(path).open(encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    _validate_required_fields(config)

    models_config = config.get("models", [])
    if models_config:
        _normalize_model_names(models_config)
        _validate_custom_name_uniqueness(models_config)

    clients_config = config.get("clients")
    if clients_config is not None:
        _validate_clients(clients_config)

    if models_config:
        _validate_model_entries(models_config)
        _validate_ip_routing(models_config, clients_config)

    tracing_config = config.get("tracing")
    if tracing_config is not None:
        _validate_tracing(tracing_config)

    return config


def _configure_log_format(config: dict[str, Any]) -> None:
    """Configure log format and filters based on tracing config.

    When tracing is enabled, adds [request_id|trace_id] to every log line.
    """
    root_logger = logging.getLogger()
    log_level = getattr(logging, config.get("logging", {}).get("log_level", "INFO").upper(), logging.INFO)
    root_logger.setLevel(log_level)

    tracing_on = config.get("tracing", {}).get("enabled", False)

    if tracing_on:
        fmt = "%(asctime)s - %(levelname)s - [%(trace_context)s] %(message)s"
    else:
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(fmt)

    if not root_logger.handlers:
        new_handler = logging.StreamHandler()
        root_logger.addHandler(new_handler)

    for h in root_logger.handlers:
        h.setFormatter(formatter)
        h.filters = [f for f in h.filters if not isinstance(f, TraceContextFilter)]
        if tracing_on:
            h.addFilter(TraceContextFilter())


def init_state(config_path: str = "config.yml") -> None:
    """Initialize all global state: load config, create OpenAI client, configure logging."""
    state.config_file_path = config_path

    try:
        state.CONFIG = load_config(config_path)
    except (ValueError, TypeError, yaml.YAMLError, OSError) as e:
        state.logger.error("Failed to load config: %s", e)
        sys.exit(1)

    _configure_log_format(state.CONFIG)

    try:
        state.client = OpenAI(
            api_key=state.CONFIG["openai"]["api_key"],
            base_url=state.CONFIG["openai"].get("base_url", "https://api.openai.com/v1"),
        )
    except (ValueError, TypeError) as e:
        state.logger.error("Error initializing OpenAI client: %s", e)
        sys.exit(1)

    try:
        state.last_config_mtime = Path(config_path).stat().st_mtime
    except OSError:
        state.last_config_mtime = 0.0


def check_and_reload_config() -> None:
    """Check config.yml mtime and reload everything from scratch if changed.

    Thread-safe: uses config_reload_lock to prevent concurrent reloads.
    """
    from ollama_adapter.models import get_and_cache_models  # noqa: PLC0415

    try:
        current_mtime = Path(state.config_file_path).stat().st_mtime
    except OSError:
        return

    if current_mtime == state.last_config_mtime:
        return

    with state.config_reload_lock:
        if current_mtime == state.last_config_mtime:
            return

        state.last_config_mtime = current_mtime

        try:
            new_config = load_config(state.config_file_path)
        except (ValueError, TypeError, yaml.YAMLError, OSError) as e:
            state.logger.warning("Config reload failed, keeping current config: %s", e)
            return

        try:
            new_client = OpenAI(
                api_key=new_config["openai"]["api_key"],
                base_url=new_config["openai"].get("base_url", "https://api.openai.com/v1"),
            )
        except (ValueError, TypeError) as e:
            state.logger.warning("Failed to recreate OpenAI client: %s", e)
            return

        state.CONFIG = new_config
        state.client = new_client

        _configure_log_format(new_config)

        try:
            get_and_cache_models(force_refresh=True)
        except (OSError, RuntimeError) as e:
            state.logger.warning("Failed to rebuild model cache: %s", e)

        state.last_config_reload_time = datetime.now(tz=UTC).isoformat()
        state.logger.info("Config changed, reloaded")
