"""Flask application factory."""

import uuid

from flask import Flask, g, request

from ollama_adapter import state
from ollama_adapter.config import check_and_reload_config, init_state
from ollama_adapter.routes import bp


def create_app(config_path: str = "config.yml") -> Flask:
    """Create and configure the Flask application."""
    init_state(config_path)

    app = Flask(__name__)

    @app.before_request
    def _before_request_reload_config() -> None:
        check_and_reload_config()

    @app.before_request
    def _before_request_set_trace_context() -> None:
        """Generate request_id and trace_id when tracing is enabled."""
        tracing = state.CONFIG.get("tracing", {})
        if not tracing.get("enabled", False):
            g.request_id = None
            g.trace_id = None
            g.litellm_response_headers = {}
            return
        g.request_id = f"req_{uuid.uuid4().hex[:12]}"
        incoming_trace = request.headers.get("x-litellm-trace-id")
        if incoming_trace:
            g.trace_id = incoming_trace
            g.trace_id_incoming = True
        else:
            prefix = tracing.get("trace_id_prefix", "oa")
            g.trace_id = f"{prefix}_{uuid.uuid4().hex[:12]}"
            g.trace_id_incoming = False
        g.litellm_response_headers = {}

    app.register_blueprint(bp)

    return app
