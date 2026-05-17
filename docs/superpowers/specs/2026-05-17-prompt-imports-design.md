# Prompt imports & templating — design spec

**Дата:** 2026-05-17
**Статус:** Draft (готов к ревью)
**Скоуп:** Возможность включать (import) другие файлы внутри промптов и подставлять переменные.

## 1. Контекст и мотивация

Сейчас `system_prompt_file` в `ollama_adapter/models.py` (`_read_prompt_file`, строки 21–29) читает файл целиком и возвращает его содержимое как есть. Hot-reload работает: файл перечитывается на каждый запрос.

Хочется:

- переиспользовать общие сниппеты (safety guidelines, форматирование, персона) между промптами разных моделей;
- параметризовать промпты переменными из `config.yml` без копирования.

Папка `prompts/` сейчас пустая (только `.gitkeep`) — переход безболезненный.

## 2. Принятые решения

| Решение                  | Выбор                                                                                           |
|--------------------------|-------------------------------------------------------------------------------------------------|
| Шаблонизатор             | **Jinja2** (`>=3.1.0`) — общий стандарт на будущее                                              |
| Sandbox                  | `jinja2.sandbox.SandboxedEnvironment`                                                           |
| Синтаксис include        | Родной Jinja2: `{% include "snippets/safety.md" %}`                                             |
| Синтаксис переменных     | Родной Jinja2: `{{ var_name }}`                                                                 |
| Подмножество Jinja2      | Полное: include/extends/blocks/if/for/macros/filters                                            |
| Вложенность include      | Рекурсивная; цикл → `RecursionError` → `PromptRenderError`                                      |
| Резолв путей в include   | Относительно `prompts.base_dir` (встроенная защита `FileSystemLoader` от `..`/абсолютных путей) |
| База для inline-шаблонов | `prompts.base_dir` (настраиваемый в `config.yml`, дефолт `./prompts`)                           |
| Переменные               | Глобальные `prompts.vars` + переопределение `prompt_vars` на уровне модели и `ip_routing`       |
| Inline vs file           | Оба поля проходят через рендер                                                                  |
| Undefined переменная     | `StrictUndefined` → ошибка                                                                      |
| Обработка ошибок         | HTTP 200, ответ как `assistant` сообщение с `[PROMPT ERROR] ...`                                |
| Безопасность путей       | Только внутри `base_dir`; абсолютные пути и `..` запрещены                                      |

## 3. Архитектура и компоненты

### 3.1. Новая зависимость

`Jinja2 >=3.1.0` в `pyproject.toml`. После добавления — `uv sync` для пересборки `uv.lock`.

### 3.2. Новый модуль `ollama_adapter/prompt_renderer.py`

Фасад над Jinja2. Экспортирует:

- `init_jinja_env(base_dir: Path) -> SandboxedEnvironment` — конструирует среду с `FileSystemLoader(base_dir)`, `StrictUndefined`, `auto_reload=True`, `cache_size=0`, `autoescape=False`.
- `class PromptRenderError(Exception)` — с атрибутами `message: str` и `source: str` (путь файла или `"inline"`).
- `render_file(env, template_path: str, vars: dict) -> str` — `env.get_template(template_path).render(vars)`.
- `render_inline(env, text: str, vars: dict) -> str` — `env.from_string(text).render(vars)`.

Внутри обеих функций перехватываются и оборачиваются в `PromptRenderError`:
`TemplateNotFound`, `TemplateSyntaxError`, `UndefinedError`, `SecurityError`, `RecursionError`, `UnicodeDecodeError`.

Сообщения форматируются так:

```
template 'role/main.md': undefined variable 'company_name'
inline prompt: template not found: 'snippets/safety.md'
template 'main.md' line 5: syntax error: unexpected '}'
template 'main.md': recursion limit exceeded (possible include cycle)
template 'evil.md': security violation: access to '__class__' denied
template 'binary.md': not a valid UTF-8 text file
```

### 3.3. Конфигурация Jinja2

```python
SandboxedEnvironment(
    loader=FileSystemLoader(base_dir, encoding="utf-8"),
    undefined=StrictUndefined,
    autoescape=False,
    auto_reload=True,
    cache_size=0,
    keep_trailing_newline=False,
    trim_blocks=False,
    lstrip_blocks=False,
)
```

`autoescape=False` критично — это текстовые промпты, не HTML.

### 3.4. Изменения по модулям

| Модуль               | Изменения                                                                                                                                                                                                                                                                                                          |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `state.py`           | + `jinja_env: SandboxedEnvironment \| None = None`                                                                                                                                                                                                                                                                 |
| `config.py`          | При `init_state` и при reload — пересоздаёт `state.jinja_env` через `init_jinja_env(prompts.base_dir or "./prompts")`                                                                                                                                                                                              |
| `models.py`          | Удалить `_read_prompt_file`. `_resolve_system_prompt` вызывает рендерер. Добавить `_collect_prompt_vars`. `apply_ip_routing` мержит поле `prompt_vars` shallow (как `params`/`headers`). `get_model_config` тянет `prompt_vars` в `adapter_params`. `apply_system_prompt` пробрасывает `PromptRenderError` наверх. |
| `routes.py`          | В `chat()`/`generate()` оборачивает вызов `apply_system_prompt` в `try/except PromptRenderError` и возвращает Ollama-формат ответа с assistant message; для streaming — один SSE-чанк + финальный `done`.                                                                                                          |
| `config-example.yml` | Документирует секцию `prompts: { base_dir, vars }` и поле `prompt_vars`.                                                                                                                                                                                                                                           |
| `README.md`          | Раздел «Prompt templates».                                                                                                                                                                                                                                                                                         |
| `CLAUDE.md`          | Обновление Module Structure, Import Graph, Key Functions.                                                                                                                                                                                                                                                          |
| `pyproject.toml`     | + `Jinja2 >=3.1.0`.                                                                                                                                                                                                                                                                                                |

### 3.5. Граф зависимостей

`prompt_renderer.py` — лист (импортирует только `state` для logger). `state.py` не импортирует `prompt_renderer`. `config.py` и `models.py` импортируют `prompt_renderer`. Циклов нет.

## 4. Конфигурация и приоритет переменных

### 4.1. Секция `prompts` в `config.yml`

```yaml
prompts:
    base_dir: ./prompts          # опционально, дефолт = "./prompts"
    vars: # опционально, дефолт = {}
        company_name: "Acme Corp"
        support_email: "help@acme.com"
        default_role: "junior"
```

### 4.2. Поля модели

```yaml
models:
    -   name: openai/gpt-4o
        custom_name: "GPT-4o"
        system_prompt_file: role/main.md   # путь относительно prompts.base_dir
        prompt_vars: # опционально
            role: "senior"
        ip_routing:
            -   ip: office
                prompt_vars:
                    role: "admin"
```

### 4.3. Правила merge (по возрастанию приоритета)

1. `prompts.vars`
2. `model.prompt_vars`
3. `ip_routing[matched].prompt_vars`

Merge — shallow по верхнеуровневым ключам. Поле `prompt_vars` обрабатывается в `apply_ip_routing` так же, как `params`/`headers`. Финальный merge с global vars выполняется в `_collect_prompt_vars`.

### 4.4. Типы значений

Любые YAML-типы. Скаляры рендерятся через `str()`. Dict/list доступны через атрибутный/индексный доступ (`{{ user.email }}`, `{{ items[0] }}`).

### 4.5. BREAKING CHANGE: резолв `system_prompt_file`

**Было:** путь относительно CWD или абсолютный (`Path.cwd() / path`).
**Стало:** путь относительно `prompts.base_dir`. Абсолютные пути и `..` запрещены.

Зачем: единая семантика — корневой файл и его include находятся в одном пространстве путей. Соответствует выбранному правилу безопасности «всё внутри base_dir».

Миграция: `system_prompt_file: prompts/main.md` → `system_prompt_file: main.md`. Папка `prompts/` сейчас пустая, версия `0.1.0` — допустим breaking change.

### 4.6. Валидация при загрузке

- `prompts.base_dir` указан, но не существует → WARN, env создаётся; первый запрос с include даст `TemplateNotFound`.
- `prompts.vars` не dict → WARN, трактуется как `{}`.
- `prompt_vars` в модели/ip_routing не dict → WARN, `{}`.

## 5. Поток запроса и обработка ошибок

### 5.1. Happy path

1. `@app.before_request` — обнаружив изменение `config.yml`, `load_config()` пересоздаёт `state.jinja_env`.
2. Handler `chat()`/`generate()`:
    - `get_model_config(model_id, client_ip)` возвращает `adapter_params` с финальным `prompt_vars` (после `apply_ip_routing`).
    - `apply_system_prompt`:
        - `_collect_prompt_vars` мержит `state.CONFIG.prompts.vars` + `adapter_params.prompt_vars`.
        - В зависимости от полей: `render_file(env, path, vars)` или `render_inline(env, text, vars)`.
        - Возвращает обновлённый `messages` (replace/prepend, как сейчас).
3. Дальше — обычный flow к OpenAI.

Hot-reload промптов: `auto_reload=True` + `cache_size=0` гарантируют, что Jinja2 перечитывает шаблоны при изменении mtime.

### 5.2. Error path

При `PromptRenderError`:

- Запрос к OpenAI **не отправляется**.
- HTTP 200 с Ollama-форматным ответом, где `assistant` сообщение содержит `[PROMPT ERROR] ...`.

### 5.3. Формат ответа по эндпоинтам

**`/api/chat` non-streaming:**

```json
{
    "model": "<display_name>",
    "created_at": "<iso>",
    "message": {
        "role": "assistant",
        "content": "[PROMPT ERROR] ..."
    },
    "done": true,
    "done_reason": "stop",
    "prompt_eval_count": 0,
    "eval_count": 0,
    "total_duration": 0,
    "load_duration": 0,
    "prompt_eval_duration": 0,
    "eval_duration": 0
}
```

**`/api/chat` streaming:** один SSE-чанк `{"message": {"role":"assistant","content":"..."}, "done": false}` + финальный `{"done": true, ...}`.

**`/api/generate`:** аналогично; поле `response` вместо `message`.

### 5.4. Логирование

```python
state.logger.error("Prompt render failed: %s", exc, exc_info=True)
```

Полный traceback в логе. Сообщение клиенту — однострочное, без stacktrace. `TraceContextFilter` сохраняет `request_id`/`trace_id`.

### 5.5. Затронутые эндпоинты

- `/api/chat`, `/api/generate` — да (вызывают `apply_system_prompt`).
- `/api/embed` — нет (нет system prompt).
- Остальные (`/api/tags`, `/api/show`, `/api/version`, `/api/ps`, `/health`, `/`) — не задеты.

## 6. Безопасность и edge-кейсы

### 6.1. Защита из коробки

| Угроза                             | Защита                                                                    |
|------------------------------------|---------------------------------------------------------------------------|
| `{% include "../../etc/passwd" %}` | `FileSystemLoader.split_template_path` блокирует `..` и абсолютные пути   |
| `{% include "/etc/passwd" %}`      | То же                                                                     |
| `{{ ''.__class__.__mro__ }}`       | `SandboxedEnvironment` → `SecurityError`                                  |
| `{{ undefined_var }}`              | `StrictUndefined` → `UndefinedError`                                      |
| Цикл include `a→b→a`               | Python `RecursionError` → `PromptRenderError("recursion limit exceeded")` |
| Бинарный файл в `prompts/`         | `UnicodeDecodeError` → `PromptRenderError`                                |

### 6.2. Что не защищаем (осознанно)

- **Симлинки изнутри `base_dir` наружу.** `FileSystemLoader` следует симлинкам. Содержимое `prompts/` — ответственность администратора. Документируем.
- **Размер итогового промпта.** Лимита нет; видимо в логах OpenAI.
- **Явный счётчик глубины include.** Полагаемся на `RecursionError`; стандартного лимита (~1000) более чем достаточно.

### 6.3. Sandbox: что разрешено

`SandboxedEnvironment` пропускает: фильтры (`{{ x|upper }}`), методы безопасных объектов (`{{ name.upper() }}`), `if`/`for`/`set`/`macro`, обычный атрибутный/индексный доступ к dict/list.

### 6.4. Edge-кейсы

| Кейс                                       | Поведение                                                       |
|--------------------------------------------|-----------------------------------------------------------------|
| `system_prompt_inline` пустой/whitespace   | Игнорируется                                                    |
| Оба `_inline` и `_file` заданы             | File выигрывает, лог WARN                                       |
| Результат рендера — пустая строка          | После `.strip()` не применяется                                 |
| `prompts.base_dir` не существует на старте | WARN; env создаётся; первый запрос упадёт с `PromptRenderError` |
| `prompts.base_dir` отсутствует в конфиге   | Дефолт = `./prompts`                                            |
| `prompts.vars` не dict                     | WARN, `{}`                                                      |
| `prompt_vars` не dict                      | WARN, `{}`                                                      |
| Литерал `{{` в тексте промпта              | `{% raw %}{{ literal }}{% endraw %}`                            |
| Промпт-файл изменён в runtime              | `auto_reload=True` подхватывает                                 |
| `config.yml` изменён                       | `load_config` пересоздаёт `state.jinja_env`                     |
| `prompts.base_dir` изменён                 | Полная пересборка env при reload                                |

## 7. Тестирование

### 7.1. Unit-тесты `tests/test_prompt_renderer.py` (новый файл)

| Тест                                                 | Ожидание                                |
|------------------------------------------------------|-----------------------------------------|
| `render_inline` простая подстановка                  | OK                                      |
| `render_inline` undefined var                        | `PromptRenderError`                     |
| `render_inline` syntax error                         | `PromptRenderError` с lineno            |
| `render_inline` обход sandbox (`{{ ''.__class__ }}`) | `PromptRenderError`                     |
| `render_file` простой include                        | OK                                      |
| `render_file` вложенный include (3 уровня)           | OK                                      |
| `render_file` цикл                                   | `PromptRenderError` ("recursion limit") |
| `render_file` `{% include "../escape.md" %}`         | `PromptRenderError`                     |
| `render_file` абсолютный путь в include              | `PromptRenderError`                     |
| `render_file` бинарный файл                          | `PromptRenderError`                     |
| `render_file` несуществующий корневой шаблон         | `PromptRenderError`                     |

### 7.2. Unit-тесты `models.py` (расширения)

- `_collect_prompt_vars`: только global, только model, merge (model > global), ip_routing > model > global.
- `apply_ip_routing`: `prompt_vars` мержится shallow, как `params`/`headers`.
- `apply_system_prompt`: пробрасывает `PromptRenderError` наверх (а не глотает).
- `apply_system_prompt`: оба `_inline` и `_file` → file wins + WARN.

### 7.3. Integration-тесты `routes.py`

- `/api/chat` non-streaming с битым промптом → 200, `message.content` начинается с `[PROMPT ERROR]`, OpenAI-mock не вызван.
- `/api/chat` streaming с битым промптом → один SSE-чанк + финальный `done`, OpenAI-mock не вызван.
- `/api/generate` non-streaming и streaming — аналогично, поле `response`.
- Happy path: `/api/chat` с include + vars → корректно отрендерен, запрос дошёл до OpenAI-mock.

### 7.4. Hot-reload тесты

- Изменение `config.yml` (mtime) → `state.jinja_env` пересоздан с новыми параметрами.
- Изменение промпт-файла → следующий запрос видит новый контент.

### 7.5. Manual (`tests/manual-check.http`)

Добавить:

1. `chat` с моделью, у которой `system_prompt_file` использует include.
2. `chat` с заведомо битым промптом (несуществующий include).
3. `generate` с inline-промптом и vars.

## 8. Документация и миграция

### 8.1. Документация

- **`README.md`** — раздел «Prompt templates»: синтаксис include/var, пример многофайловой структуры, таблица приоритета `vars`, breaking change по путям, sandbox-ограничения, симлинки.
- **`config-example.yml`** — секция `prompts: { base_dir, vars }` с комментариями, поле `prompt_vars` в model и ip_routing, обновлённые пути `system_prompt_file` (без префикса `prompts/`).
- **`CLAUDE.md`** — `prompt_renderer.py` в Module Structure, обновлённый Import Graph, описание ключевых функций; убрать упоминание `_read_prompt_file`.

### 8.2. Миграция

1. `system_prompt_file: prompts/main.md` → `system_prompt_file: main.md`.
2. Старый ключ `system_prompt` уже deprecated и игнорируется — не трогаем.
3. Папка `prompts/` пустая — реальных файлов мигрировать нет.
4. Версия `0.1.0` — допустим breaking change; отметить в commit message и README.

## 9. Вне scope

- Передача параметров в `{% include %}` (Jinja2 не поддерживает напрямую; обходится через `{% set %}` или макросы).
- UI/CLI для отладки промптов вне runtime.
- Версионирование промптов / хранение в БД.
- Расширение `prompt_renderer` на другие части кода (динамические headers и т.д.) — модуль независимый, при необходимости легко переиспользуется.
