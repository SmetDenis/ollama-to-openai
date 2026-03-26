# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ollama-to-OpenAI adapter — Python-сервис (Flask), который транслирует запросы Ollama API в вызовы OpenAI API. Позволяет клиентам Ollama прозрачно использовать модели OpenAI (и совместимые провайдеры через LiteLLM).

## Architecture

Всё приложение — **один файл** `ollama_to_openai_adapter.py` (~1900 строк). Зависимости: Flask, OpenAI SDK, PyYAML.

### Request Flow

1. `@app.before_request` — проверяет mtime `config.yml`, при изменении перезагружает конфиг, пересоздаёт OpenAI-клиент, обновляет кеш моделей
2. `@log_endpoint` декоратор — логирует запрос/ответ, замеряет время
3. Endpoint handler — валидация входных данных, получение конфига модели, вызов OpenAI API, форматирование ответа в формат Ollama
4. Если tracing включён — `@app.before_request` генерирует `request_id`/`trace_id`, которые попадают в логи через `TraceContextFilter`

### Key Functions

- **`get_model_config(model_id, client_ip)`** — центральная функция конфигурации. Возвращает кортеж `(openai_params, adapter_params, headers)`:
  - `openai_params` — параметры для OpenAI API (temperature, max_tokens и т.д.) + `model_id`
  - `adapter_params` — настройки адаптера (remove_thinking_tags, system_prompt, prompt_caching)
  - `headers` — кастомные HTTP-заголовки для OpenAI API
- **`resolve_model_name(client_name)`** — преобразует custom_name в оригинальный OpenAI model ID
- **`get_display_name(original_name)`** — обратное преобразование для ответов клиентам
- **`apply_ip_routing(model_entry, client_ip)`** — применяет IP-специфичные override'ы; shallow merge для dict-полей (params, headers)
- **`get_and_cache_models()`** — загружает модели из OpenAI API, кеширует в `CACHED_MODELS` с синтетическими метаданными Ollama-формата
- **`apply_system_prompt(messages, adapter_params, model_id)`** — внедряет системный промпт из конфига
- **`resolve_system_prompt(value)`** — если значение заканчивается на `.md`, читает файл при каждом запросе (hot-reload промптов без рестарта)
- **`apply_prompt_caching(messages, adapter_params, model_id)`** — добавляет `cache_control` маркеры в system message для Anthropic/Gemini через LiteLLM
- **`remove_thinking_tags(content, model_id, remove_enabled)`** — удаление `<think>`/`<thinking>` тегов (regex для non-streaming)

### Streaming Thinking Tag Removal

Для стриминга используется **5-state machine** (определена inline в `chat()` и `generate()`):
1. `INITIAL` — захват первого chunk'а
2. `DETECTING_OPEN_TAG` — поиск открывающего `<think>`/`<thinking>`
3. `BUFFERING_THINKING` — накопление содержимого до `</`
4. `DETECTING_CLOSE_TAG` — поиск закрывающего тега
5. `STREAMING_NORMAL` — прямая трансляция остального контента

### Prompts Directory

`prompts/` — файлы системных промптов (`.md`), на которые ссылается `system_prompt` в конфиге моделей. Монтируются read-only в Docker. Файлы перечитываются при каждом запросе — можно менять без рестарта.

## Commands

```bash
# Запуск локально
python3 ollama_to_openai_adapter.py
# или через venv
./.venv/bin/python3 ollama_to_openai_adapter.py

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
