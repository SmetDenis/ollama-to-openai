# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ollama-to-OpenAI adapter — Python-сервис (Flask), который транслирует запросы Ollama API в вызовы OpenAI API. Позволяет клиентам Ollama прозрачно использовать модели OpenAI (и совместимые провайдеры через LiteLLM).

## Architecture

Модульная структура в пакете `ollama_adapter/`. Зависимости: Flask, OpenAI SDK, PyYAML.

### Module Structure

```
ollama_adapter/
  __init__.py          # Пустой
  __main__.py          # Entrypoint: create_app + app.run
  state.py             # Глобальное состояние: CONFIG, client, CACHED_MODELS
  config.py            # load_config(), init_state(), hot-reload
  logging_utils.py     # TraceContextFilter, validation, log_request/response, @log_endpoint
  tracing.py           # LiteLLM headers, build_trace_*, capture/log headers
  thinking.py          # remove_thinking_tags(), _process_stream() — 5-state machine
  models.py            # Модели, промпты, IP-routing, конфиг моделей
  routes.py            # Flask Blueprint, все endpoints, shared completion helper
  app.py               # create_app() factory, before_request hooks
```

### Import Graph

`state.py` — лист графа (ничего не импортирует из пакета). Все модули импортируют `state`. `routes.py` импортирует `logging_utils`, `tracing`, `thinking`, `models`. `config.py` импортирует `logging_utils` (TraceContextFilter) и `models` (get_and_cache_models при reload). Циклических зависимостей нет.

### Request Flow

1. `@app.before_request` (`app.py`) — проверяет mtime `config.yml`, при изменении перезагружает конфиг, пересоздаёт OpenAI-клиент, обновляет кеш моделей
2. `@log_endpoint` декоратор (`logging_utils.py`) — логирует запрос/ответ, замеряет время
3. Endpoint handler (`routes.py`) — валидация входных данных, получение конфига модели, вызов OpenAI API, форматирование ответа в формат Ollama
4. Если tracing включён — `@app.before_request` генерирует `request_id`/`trace_id`, которые попадают в логи через `TraceContextFilter`

### Key Functions

- **`get_model_config(model_id, client_ip)`** (`models.py`) — центральная функция конфигурации. Возвращает кортеж `(openai_params, adapter_params, headers)`
- **`resolve_model_name(client_name)`** (`models.py`) — преобразует custom_name в оригинальный OpenAI model ID
- **`get_display_name(original_name)`** (`models.py`) — обратное преобразование для ответов клиентам
- **`apply_ip_routing(model_entry, client_ip)`** (`models.py`) — применяет IP-специфичные override'ы; shallow merge для dict-полей
- **`get_and_cache_models()`** (`models.py`) — загружает модели из OpenAI API, кеширует в `state.CACHED_MODELS`
- **`apply_system_prompt(messages, adapter_params, model_id)`** (`models.py`) — внедряет системный промпт из конфига
- **`resolve_system_prompt(value)`** (`models.py`) — если значение заканчивается на `.md`, читает файл при каждом запросе (hot-reload)
- **`apply_prompt_caching(messages, adapter_params, model_id)`** (`models.py`) — добавляет `cache_control` маркеры для Anthropic/Gemini
- **`remove_thinking_tags(content, model_id, remove_enabled)`** (`thinking.py`) — удаление `<think>`/`<thinking>` тегов
- **`_process_stream()`** (`thinking.py`) — 5-state machine для streaming tag removal
- **`_call_openai_streaming()`** / **`_call_openai_non_streaming()`** (`routes.py`) — shared-хелперы для `chat()` и `generate()`

### Prompts Directory

`prompts/` — файлы системных промптов (`.md`), на которые ссылается `system_prompt` в конфиге моделей. Монтируются read-only в Docker. Файлы перечитываются при каждом запросе — можно менять без рестарта.

## Commands

```bash
# Запуск локально
python3 -m ollama_adapter
# или через venv
./.venv/bin/python3 -m ollama_adapter

# Docker Compose (порт 11345 -> 11434)
docker-compose up -d

# Docker standalone
docker build -t ollama-to-openai .
docker run -p 11434:11434 -v ./config.yml:/app/config.yml:ro ollama-to-openai

# Пересоздание виртуального окружения
uv sync
```

## Configuration

Файл `config.yml` (см. `config-example.yml` для полного примера с комментариями). Ключевые секции:

- **`server`**: `host`, `port`
- **`openai`**: `api_key` (обязательный), `base_url` (опциональный — для LiteLLM, Azure и др.)
- **`clients`**: именованные группы IP-адресов для `ip_routing`
- **`logging`**: `log_level`, `log_requests`
- **`tracing`**: интеграция с LiteLLM proxy — request_id/trace_id, заголовки, теги
- **`models`**: список моделей с двухуровневой структурой:
  - Корневой уровень: `name` (обязательный), `custom_name`, `remove_thinking_tags`, `prompt_caching`, `system_prompt`
  - `params`: dict параметров OpenAI API — passthrough без валидации
  - `headers`: dict кастомных HTTP-заголовков
  - `ip_routing`: список IP-специфичных override'ов (наследование + shallow merge)

Если `models` пуст — выставляются все доступные модели OpenAI. Если заполнен — только указанные.

Config hot-reload: при каждом запросе проверяется mtime файла. При изменении — перезагрузка конфига, пересоздание OpenAI-клиента, обновление кеша моделей. При ошибке загрузки — сохраняется текущий конфиг.

## API Endpoints

| Endpoint | Method | Назначение |
|---|---|---|
| `/api/chat` | POST | Chat completions (streaming/non-streaming) |
| `/api/generate` | POST | Text generation (streaming/non-streaming) |
| `/api/embed` | POST | Embeddings |
| `/api/tags` | GET/POST | Список моделей |
| `/api/show` | POST | Информация о модели |
| `/api/version` | GET | Версия сервиса |
| `/api/ps` | GET | Running models (mock) |
| `/health` | GET | Health check с проверкой OpenAI |
| `/` | GET | Информация о сервисе |

## Testing

**Важно**: НЕ запускать сервер автоматически после изменений. Пользователь тестирует сам.

Ручные тесты: `tests/manual-check.http` — HTTP-запросы для всех эндпоинтов, включая error cases. Порт 11345 (docker-compose) или 11434 (локально).

## Development Notes

- **Package Manager**: UV с lock-файлом (`uv.lock`)
- **Python**: 3.13+ (`.python-version`)
- **Flask debug mode**: включён — auto-reload при изменении кода
- **Версия**: 0.1.0
- **CI/CD**: GitHub Actions (`.github/workflows/ci.yml`) — линтеры + сборка Docker-образа в GHCR
- **Линтеры и тесты**: всегда запускать через `make` (например `make check`), а не напрямую вызывать ruff/mypy/pytest
- **Финальная проверка**: при завершении любой задачи всегда запускать `make pre-commit` — он форматирует код, фиксит линтер и запускает полную проверку (format → lint-fix → check)
