"""Entry point for the Ollama-to-OpenAI adapter."""

from ollama_adapter import state
from ollama_adapter.app import create_app
from ollama_adapter.models import get_and_cache_models

app = create_app()
get_and_cache_models()

server_config: dict = state.CONFIG["server"]
state.logger.info(
    "Starting Ollama -> OpenAI adapter on http://%s:%s",
    server_config["host"],
    server_config["port"],
)

app.run(
    host=server_config["host"],
    port=server_config["port"],
    debug=True,
    use_reloader=True,
    reloader_type="stat",
)
