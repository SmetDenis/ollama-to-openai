from ollama_adapter.app import create_app
from ollama_adapter.models import get_and_cache_models
from ollama_adapter import state

app = create_app()
get_and_cache_models()

server_config = state.CONFIG['server']
state.logger.info(f"Starting Ollama -> OpenAI adapter on http://{server_config['host']}:{server_config['port']}")

app.run(
    host=server_config['host'],
    port=server_config['port'],
    debug=True,
    use_reloader=True,
    reloader_type='stat'
)
