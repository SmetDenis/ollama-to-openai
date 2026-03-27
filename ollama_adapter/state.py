"""Global shared state for the Ollama-to-OpenAI adapter."""

import logging
import threading

from openai import OpenAI

CONFIG: dict = {}
client: OpenAI | None = None
CACHED_MODELS: list[dict] = []

config_file_path: str = "config.yml"
last_config_mtime: float = 0.0
last_config_reload_time: str | None = None
config_reload_lock: threading.Lock = threading.Lock()

logger: logging.Logger = logging.getLogger("ollama_adapter")
