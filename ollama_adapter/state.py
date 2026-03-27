import logging
import threading

CONFIG = {}
client = None
CACHED_MODELS = []

_config_file_path = 'config.yml'
_last_config_mtime = 0.0
_last_config_reload_time = None
_config_reload_lock = threading.Lock()

logger = logging.getLogger('ollama_adapter')
