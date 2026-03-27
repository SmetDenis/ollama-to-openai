import sys
import os
import logging
import yaml
from datetime import datetime
from openai import OpenAI

from ollama_adapter import state
from ollama_adapter.logging_utils import TraceContextFilter


def load_config(path='config.yml'):
    """Loads and validates YAML configuration file. Raises on any error."""
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if not config.get('openai', {}).get('api_key'):
        raise ValueError("Missing required parameter 'openai.api_key' in config.yml")
    if not config.get('server', {}).get('host'):
        raise ValueError("Missing required parameter 'server.host' in config.yml")
    if not config.get('server', {}).get('port'):
        raise ValueError("Missing required parameter 'server.port' in config.yml")

    # Normalize custom_name values (strip whitespace)
    models_config = config.get('models', [])
    if models_config:
        for model in models_config:
            if isinstance(model, dict) and 'custom_name' in model:
                model['custom_name'] = model['custom_name'].strip()

    # Validate custom_name uniqueness
    if models_config:
        custom_names = []
        for model in models_config:
            if isinstance(model, dict) and 'custom_name' in model:
                custom_names.append(model['custom_name'])

        duplicates = [name for name in set(custom_names) if custom_names.count(name) > 1]
        if duplicates:
            raise ValueError(f"Duplicate custom_name values found: {duplicates}")

    # Validate clients section
    clients_config = config.get('clients')
    if clients_config is not None:
        if not isinstance(clients_config, dict):
            raise ValueError("'clients' must be a dict mapping alias names to IP addresses")
        for alias, ips in clients_config.items():
            if isinstance(ips, str):
                if not ips.strip():
                    raise ValueError(f"Client alias '{alias}': IP address cannot be empty")
                clients_config[alias] = ips.strip()
            elif isinstance(ips, list):
                if not ips:
                    raise ValueError(f"Client alias '{alias}': IP list cannot be empty")
                for i, ip in enumerate(ips):
                    if not isinstance(ip, str) or not ip.strip():
                        raise ValueError(f"Client alias '{alias}': ip[{i}] must be a non-empty string")
                clients_config[alias] = [ip.strip() for ip in ips]
            else:
                raise ValueError(f"Client alias '{alias}': value must be a string or list of strings")

    # Validate ip_routing in model entries
    if models_config:
        for idx, model in enumerate(models_config):
            if not isinstance(model, dict):
                continue
            ip_routing = model.get('ip_routing')
            if ip_routing is None:
                continue

            model_display = model.get('custom_name') or model.get('name', f'models[{idx}]')

            if not isinstance(ip_routing, list):
                raise ValueError(f"Model '{model_display}': 'ip_routing' must be a list")

            seen_ips = set()
            for rule_idx, rule in enumerate(ip_routing):
                if not isinstance(rule, dict):
                    raise ValueError(
                        f"Model '{model_display}': ip_routing[{rule_idx}] must be a dict")

                ip_value = rule.get('ip')
                if not ip_value or not isinstance(ip_value, str) or not ip_value.strip():
                    raise ValueError(
                        f"Model '{model_display}': ip_routing[{rule_idx}] "
                        f"must have a non-empty 'ip' field (alias name or direct IP)")

                ip_value = ip_value.strip()
                rule['ip'] = ip_value

                # Resolve to actual IPs for duplicate checking
                if clients_config and ip_value in clients_config:
                    resolved = clients_config[ip_value]
                    resolved_ips = [resolved] if isinstance(resolved, str) else resolved
                else:
                    resolved_ips = [ip_value]

                for ip in resolved_ips:
                    if ip in seen_ips:
                        raise ValueError(
                            f"Model '{model_display}': duplicate IP '{ip}' in ip_routing")
                    seen_ips.add(ip)

                if 'params' in rule and rule['params'] is not None:
                    if not isinstance(rule['params'], dict):
                        raise ValueError(
                            f"Model '{model_display}': ip_routing[{rule_idx}] 'params' must be a dict")

                if 'headers' in rule and rule['headers'] is not None:
                    if not isinstance(rule['headers'], dict):
                        raise ValueError(
                            f"Model '{model_display}': ip_routing[{rule_idx}] 'headers' must be a dict")

                # Warn about unrecognized keys (likely misplaced params like temperature)
                recognized_keys = {'ip', 'name', 'system_prompt', 'remove_thinking_tags', 'prompt_caching', 'params', 'headers'}
                unknown_keys = set(rule.keys()) - recognized_keys
                if unknown_keys:
                    state.logger.warning(
                        f"Model '{model_display}': ip_routing[{rule_idx}] has unrecognized keys "
                        f"{unknown_keys}. These will be ignored. "
                        f"Did you mean to put them inside 'params'?")

    # Validate tracing section
    tracing_config = config.get('tracing')
    if tracing_config is not None:
        if not isinstance(tracing_config, dict):
            raise ValueError("'tracing' must be a dict")
        for flag in ('enabled', 'log_headers', 'send_trace_headers'):
            if flag in tracing_config and not isinstance(tracing_config[flag], bool):
                raise ValueError(f"tracing.{flag} must be a boolean (true/false)")
        for field in ('trace_id_prefix', 'tags'):
            if field in tracing_config and not isinstance(tracing_config[field], str):
                raise ValueError(f"tracing.{field} must be a string")

    return config


def _configure_log_format(config):
    """Configure log format and filters based on tracing config.
    When tracing is enabled, adds [request_id|trace_id] to every log line."""
    root_logger = logging.getLogger()
    log_level = getattr(logging, config.get('logging', {}).get('log_level', 'INFO').upper(), logging.INFO)
    root_logger.setLevel(log_level)

    tracing_on = config.get('tracing', {}).get('enabled', False)

    if tracing_on:
        fmt = '%(asctime)s - %(levelname)s - [%(trace_context)s] %(message)s'
    else:
        fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(fmt)

    if not root_logger.handlers:
        handler = logging.StreamHandler()
        root_logger.addHandler(handler)

    for handler in root_logger.handlers:
        handler.setFormatter(formatter)
        handler.filters = [f for f in handler.filters if not isinstance(f, TraceContextFilter)]
        if tracing_on:
            handler.addFilter(TraceContextFilter())


def init_state(config_path='config.yml'):
    """Initialize all global state: load config, create OpenAI client, configure logging."""
    state._config_file_path = config_path

    try:
        state.CONFIG = load_config(config_path)
    except Exception as e:
        state.logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    _configure_log_format(state.CONFIG)

    try:
        state.client = OpenAI(
            api_key=state.CONFIG['openai']['api_key'],
            base_url=state.CONFIG['openai'].get('base_url', 'https://api.openai.com/v1')
        )
    except Exception as e:
        state.logger.error(f"Error initializing OpenAI client: {e}")
        sys.exit(1)

    try:
        state._last_config_mtime = os.path.getmtime(config_path)
    except OSError:
        state._last_config_mtime = 0.0


def _check_and_reload_config():
    """Check config.yml mtime; reload everything from scratch if changed.
    Thread-safe: uses _config_reload_lock to prevent concurrent reloads."""
    from ollama_adapter.models import get_and_cache_models

    try:
        current_mtime = os.path.getmtime(state._config_file_path)
    except OSError:
        return

    if current_mtime == state._last_config_mtime:
        return

    with state._config_reload_lock:
        # Re-check after acquiring lock (another thread may have reloaded already)
        if current_mtime == state._last_config_mtime:
            return

        state._last_config_mtime = current_mtime

        try:
            new_config = load_config(state._config_file_path)
        except Exception as e:
            state.logger.warning(f"Config reload failed, keeping current config: {e}")
            return

        try:
            new_client = OpenAI(
                api_key=new_config['openai']['api_key'],
                base_url=new_config['openai'].get('base_url', 'https://api.openai.com/v1')
            )
        except Exception as e:
            state.logger.warning(f"Failed to recreate OpenAI client: {e}")
            return

        # Atomic swap: update all globals together
        state.CONFIG = new_config
        state.client = new_client

        _configure_log_format(new_config)

        try:
            get_and_cache_models(force_refresh=True)
        except Exception as e:
            state.logger.warning(f"Failed to rebuild model cache: {e}")

        state._last_config_reload_time = datetime.now().isoformat() + "Z"
        state.logger.info("Config changed, reloaded")
