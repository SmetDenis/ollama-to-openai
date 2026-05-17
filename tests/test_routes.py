"""Tests for ollama_adapter.routes module — all endpoints via Flask test client."""

from unittest.mock import patch

from ollama_adapter import state

from .conftest import collect_stream, make_mock_chunk, make_mock_completion, make_mock_embedding

# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


class TestRoot:
    def test_service_info(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["service"] == "Ollama to OpenAI Adapter"
        assert data["version"] == "0.1.0"
        assert "/api/chat" in data["endpoints"]


# ---------------------------------------------------------------------------
# GET /api/version
# ---------------------------------------------------------------------------


class TestVersion:
    def test_returns_version(self, client):
        resp = client.get("/api/version")
        assert resp.status_code == 200
        assert resp.get_json() == {"version": "0.1.0"}


# ---------------------------------------------------------------------------
# GET /api/ps
# ---------------------------------------------------------------------------


class TestPs:
    def test_returns_empty_models(self, client):
        resp = client.get("/api/ps")
        assert resp.status_code == 200
        assert resp.get_json() == {"models": []}


# ---------------------------------------------------------------------------
# GET /api/tags
# ---------------------------------------------------------------------------


class TestTagsGet:
    def test_returns_models(self, client):
        with patch("ollama_adapter.routes.get_and_cache_models", return_value=[{"name": "m1"}]):
            resp = client.get("/api/tags")
        assert resp.status_code == 200
        assert resp.get_json()["models"] == [{"name": "m1"}]

    def test_no_models_500(self, client):
        with patch("ollama_adapter.routes.get_and_cache_models", return_value=[]):
            resp = client.get("/api/tags")
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# POST /api/tags
# ---------------------------------------------------------------------------


class TestTagsPost:
    def test_finds_model(self, client):
        models = [{"name": "gpt", "model": "gpt", "digest": "openai/gpt-4o"}]
        with patch("ollama_adapter.routes.get_and_cache_models", return_value=models):
            resp = client.post("/api/tags", json={"model": "gpt"})
        assert resp.status_code == 200
        assert len(resp.get_json()["models"]) == 1

    def test_model_not_found(self, client):
        models = [{"name": "gpt", "model": "gpt", "digest": "openai/gpt-4o"}]
        with patch("ollama_adapter.routes.get_and_cache_models", return_value=models):
            resp = client.post("/api/tags", json={"model": "unknown"})
        assert resp.status_code == 404

    def test_invalid_json(self, client):
        with patch("ollama_adapter.routes.get_and_cache_models", return_value=[{"name": "m"}]):
            resp = client.post("/api/tags", data="not json", content_type="text/plain")
        assert resp.status_code == 400

    def test_missing_model(self, client):
        with patch("ollama_adapter.routes.get_and_cache_models", return_value=[{"name": "m"}]):
            resp = client.post("/api/tags", json={})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /api/show
# ---------------------------------------------------------------------------


class TestShow:
    def test_returns_metadata(self, client):
        resp = client.post("/api/show", json={"model": "gpt-4o"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "modelfile" in data
        assert "details" in data
        assert "capabilities" in data

    def test_invalid_json(self, client):
        resp = client.post("/api/show", data="bad", content_type="text/plain")
        assert resp.status_code == 400

    def test_missing_model(self, client):
        resp = client.post("/api/show", json={})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /api/chat — non-streaming
# ---------------------------------------------------------------------------


class TestChatNonStreaming:
    def test_success(self, client, mock_openai_client):
        mock_openai_client.chat.completions.create.return_value = make_mock_completion("Hello!")
        resp = client.post(
            "/api/chat",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["message"]["content"] == "Hello!"
        assert data["done"] is True

    def test_invalid_json(self, client):
        resp = client.post("/api/chat", data="bad", content_type="text/plain")
        assert resp.status_code == 400

    def test_missing_model(self, client):
        resp = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 400

    def test_missing_messages(self, client):
        resp = client.post("/api/chat", json={"model": "gpt-4o"})
        assert resp.status_code == 400

    def test_empty_messages(self, client):
        resp = client.post("/api/chat", json={"model": "gpt-4o", "messages": []})
        assert resp.status_code == 400

    def test_no_choices_returns_assistant_message(self, client, mock_openai_client):
        response = make_mock_completion()
        response.choices = []
        mock_openai_client.chat.completions.create.return_value = response
        resp = client.post(
            "/api/chat",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["done"] is True
        assert data["message"]["role"] == "assistant"
        assert "[LLM ERROR]" in data["message"]["content"]

    def test_no_choices_legacy_returns_500_when_disabled(self, client, mock_openai_client):
        state.CONFIG = {**state.CONFIG, "error_handling": {"enabled": False}}
        response = make_mock_completion()
        response.choices = []
        mock_openai_client.chat.completions.create.return_value = response
        resp = client.post(
            "/api/chat",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 500

    def test_api_error_returns_assistant_message(self, client, mock_openai_client):
        mock_openai_client.chat.completions.create.side_effect = RuntimeError("API down")
        with patch.object(state.logger, "exception") as mock_log:
            resp = client.post(
                "/api/chat",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["done"] is True
        assert data["message"]["role"] == "assistant"
        assert data["message"]["content"].startswith("[LLM ERROR]")
        assert "API down" in data["message"]["content"]
        mock_log.assert_called_once_with("Chat endpoint error")

    def test_api_error_legacy_returns_500_when_disabled(self, client, mock_openai_client):
        state.CONFIG = {**state.CONFIG, "error_handling": {"enabled": False}}
        mock_openai_client.chat.completions.create.side_effect = RuntimeError("API down")
        resp = client.post(
            "/api/chat",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 500

    def test_thinking_removal(self, client, mock_openai_client, full_config):
        full_config.pop("tracing", None)
        state.CONFIG = full_config
        mock_openai_client.chat.completions.create.return_value = make_mock_completion("<think>reasoning</think>Answer")
        resp = client.post(
            "/api/chat",
            json={
                "model": "Mini",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        data = resp.get_json()
        assert data["message"]["content"] == "Answer"


# ---------------------------------------------------------------------------
# POST /api/chat — streaming
# ---------------------------------------------------------------------------


class TestChatStreaming:
    def _mock_stream(self, mock_openai_client, chunks):
        mock_openai_client.chat.completions.create.return_value = iter(chunks)

    def test_success(self, client, mock_openai_client):
        chunks = [make_mock_chunk("Hello"), make_mock_chunk(" world")]
        self._mock_stream(mock_openai_client, chunks)

        resp = client.post(
            "/api/chat",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        data = collect_stream(resp)
        assert len(data) >= 2
        last = data[-1]
        assert last["done"] is True

    def test_mimetype(self, client, mock_openai_client):
        self._mock_stream(mock_openai_client, [make_mock_chunk("Hi")])
        resp = client.post(
            "/api/chat",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.content_type.startswith("application/x-ndjson")

    def test_final_chunk_has_done(self, client, mock_openai_client):
        self._mock_stream(mock_openai_client, [make_mock_chunk("Hi")])
        resp = client.post(
            "/api/chat",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        data = collect_stream(resp)
        last = data[-1]
        assert last["done"] is True
        assert last["message"]["content"] == ""

    def test_streaming_error_emits_done_chunk(self, client, mock_openai_client):
        mock_openai_client.chat.completions.create.side_effect = RuntimeError("stream fail")
        with patch.object(state.logger, "exception") as mock_log:
            resp = client.post(
                "/api/chat",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
            data = collect_stream(resp)
        assert resp.status_code == 200
        assert any("[LLM ERROR]" in (chunk.get("message") or {}).get("content", "") for chunk in data)
        assert data[-1]["done"] is True
        mock_log.assert_called_once()
        assert "Streaming error" in mock_log.call_args[0][0]

    def test_streaming_legacy_error_when_disabled(self, client, mock_openai_client):
        state.CONFIG = {**state.CONFIG, "error_handling": {"enabled": False}}
        mock_openai_client.chat.completions.create.side_effect = RuntimeError("stream fail")
        resp = client.post(
            "/api/chat",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        data = collect_stream(resp)
        assert any("error" in chunk for chunk in data)


# ---------------------------------------------------------------------------
# POST /api/chat — prompt render errors
# ---------------------------------------------------------------------------


class TestChatPromptRenderError:
    def _config_with_missing_prompt(self):
        return {
            "models": [
                {
                    "name": "openai/gpt-4o",
                    "custom_name": "GPT-4o",
                    "system_prompt_file": "missing.md",
                },
            ],
        }

    def test_non_streaming_returns_assistant_message(self, client, mock_openai_client):
        state.CONFIG = self._config_with_missing_prompt()
        resp = client.post(
            "/api/chat",
            json={"model": "GPT-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["done"] is True
        assert data["message"]["role"] == "assistant"
        assert data["message"]["content"].startswith("[PROMPT ERROR]")
        mock_openai_client.chat.completions.create.assert_not_called()

    def test_streaming_emits_error_chunk_then_done(self, client, mock_openai_client):
        state.CONFIG = self._config_with_missing_prompt()
        resp = client.post(
            "/api/chat",
            json={
                "model": "GPT-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        chunks = collect_stream(resp)
        assert len(chunks) >= 2
        assert chunks[0]["message"]["content"].startswith("[PROMPT ERROR]")
        assert chunks[-1]["done"] is True
        mock_openai_client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# POST /api/generate
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_non_streaming_success(self, client, mock_openai_client):
        mock_openai_client.chat.completions.create.return_value = make_mock_completion("Generated!")
        resp = client.post(
            "/api/generate",
            json={
                "model": "gpt-4o",
                "prompt": "Write a poem",
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["response"] == "Generated!"
        assert data["done"] is True

    def test_missing_prompt(self, client):
        resp = client.post("/api/generate", json={"model": "gpt-4o"})
        assert resp.status_code == 400

    def test_empty_prompt(self, client):
        resp = client.post("/api/generate", json={"model": "gpt-4o", "prompt": "  "})
        assert resp.status_code == 400

    def test_streaming_success(self, client, mock_openai_client):
        mock_openai_client.chat.completions.create.return_value = iter([make_mock_chunk("Hi")])
        resp = client.post(
            "/api/generate",
            json={
                "model": "gpt-4o",
                "prompt": "Write something",
                "stream": True,
            },
        )
        assert resp.status_code == 200
        data = collect_stream(resp)
        last = data[-1]
        assert last["done"] is True
        assert "response" in last

    def test_api_error_returns_response_field(self, client, mock_openai_client):
        mock_openai_client.chat.completions.create.side_effect = RuntimeError("API down")
        with patch.object(state.logger, "exception") as mock_log:
            resp = client.post(
                "/api/generate",
                json={"model": "gpt-4o", "prompt": "Write a poem"},
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["done"] is True
        assert data["response"].startswith("[LLM ERROR]")
        assert "API down" in data["response"]
        mock_log.assert_called_once_with("Generate endpoint error")

    def test_api_error_legacy_returns_500_when_disabled(self, client, mock_openai_client):
        state.CONFIG = {**state.CONFIG, "error_handling": {"enabled": False}}
        mock_openai_client.chat.completions.create.side_effect = RuntimeError("API down")
        resp = client.post(
            "/api/generate",
            json={"model": "gpt-4o", "prompt": "Write a poem"},
        )
        assert resp.status_code == 500

    def test_streaming_api_error_emits_done(self, client, mock_openai_client):
        mock_openai_client.chat.completions.create.side_effect = RuntimeError("boom")
        resp = client.post(
            "/api/generate",
            json={"model": "gpt-4o", "prompt": "Write something", "stream": True},
        )
        data = collect_stream(resp)
        assert resp.status_code == 200
        assert any("[LLM ERROR]" in chunk.get("response", "") for chunk in data)
        assert data[-1]["done"] is True

    def test_prompt_render_error_non_streaming(self, client, mock_openai_client):
        state.CONFIG = {
            "models": [
                {
                    "name": "openai/gpt-4o",
                    "custom_name": "GPT-4o",
                    "system_prompt_file": "missing.md",
                },
            ],
        }
        resp = client.post(
            "/api/generate",
            json={"model": "GPT-4o", "prompt": "Write a poem"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["done"] is True
        assert data["response"].startswith("[PROMPT ERROR]")
        mock_openai_client.chat.completions.create.assert_not_called()

    def test_prompt_render_error_streaming(self, client, mock_openai_client):
        state.CONFIG = {
            "models": [
                {
                    "name": "openai/gpt-4o",
                    "custom_name": "GPT-4o",
                    "system_prompt_file": "missing.md",
                },
            ],
        }
        resp = client.post(
            "/api/generate",
            json={"model": "GPT-4o", "prompt": "Write something", "stream": True},
        )
        chunks = collect_stream(resp)
        assert chunks[0]["response"].startswith("[PROMPT ERROR]")
        assert chunks[-1]["done"] is True
        mock_openai_client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# POST /api/embed
# ---------------------------------------------------------------------------


class TestEmbed:
    def test_success_single_input(self, client, mock_openai_client):
        mock_openai_client.embeddings.create.return_value = make_mock_embedding()
        resp = client.post("/api/embed", json={"model": "emb", "input": "hello"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "embeddings" in data
        assert len(data["embeddings"]) == 1

    def test_success_list_input(self, client, mock_openai_client):
        mock_openai_client.embeddings.create.return_value = make_mock_embedding([[0.1, 0.2], [0.3, 0.4]])
        resp = client.post("/api/embed", json={"model": "emb", "input": ["a", "b"]})
        assert resp.status_code == 200
        assert len(resp.get_json()["embeddings"]) == 2

    def test_missing_input(self, client):
        resp = client.post("/api/embed", json={"model": "emb"})
        assert resp.status_code == 400

    def test_invalid_input_type(self, client):
        resp = client.post("/api/embed", json={"model": "emb", "input": 123})
        assert resp.status_code == 400

    def test_api_error_500(self, client, mock_openai_client):
        mock_openai_client.embeddings.create.side_effect = RuntimeError("fail")
        with patch.object(state.logger, "exception") as mock_log:
            resp = client.post("/api/embed", json={"model": "emb", "input": "hi"})
        assert resp.status_code == 500
        mock_log.assert_called_once_with("Embed endpoint error")


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_healthy(self, client, mock_openai_client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"

    def test_unhealthy(self, client, mock_openai_client):
        mock_openai_client.with_options.return_value.models.list.side_effect = RuntimeError("down")
        resp = client.get("/health")
        assert resp.status_code == 503
        assert resp.get_json()["status"] == "unhealthy"
