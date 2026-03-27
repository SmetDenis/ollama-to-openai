import uuid
from flask import Flask, request, g

from ollama_adapter import state
from ollama_adapter.config import init_state, _check_and_reload_config
from ollama_adapter.routes import bp


def create_app(config_path='config.yml'):
    """Flask application factory."""
    init_state(config_path)

    app = Flask(__name__)

    @app.before_request
    def _before_request_reload_config():
        _check_and_reload_config()

    @app.before_request
    def _before_request_set_trace_context():
        """Generate request_id and trace_id for every request when tracing is enabled."""
        tracing = state.CONFIG.get('tracing', {})
        if not tracing.get('enabled', False):
            g.request_id = None
            g.trace_id = None
            g.litellm_response_headers = {}
            return
        g.request_id = f"req_{uuid.uuid4().hex[:12]}"
        incoming_trace = request.headers.get('x-litellm-trace-id')
        if incoming_trace:
            g.trace_id = incoming_trace
        else:
            prefix = tracing.get('trace_id_prefix', 'oa')
            g.trace_id = f"{prefix}_{uuid.uuid4().hex[:12]}"
        g.litellm_response_headers = {}

    app.register_blueprint(bp)

    return app
