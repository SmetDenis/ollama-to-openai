# Design: Debug Prompt Output (`debug` keyword)

**Date:** 2026-06-03
**Status:** Approved design, pending implementation
**Area:** `ollama_adapter/prompt_renderer.py`, `ollama_adapter/models.py`, `ollama_adapter/debug_prompt.py` (new), `ollama_adapter/routes.py`, `config-example.yml`, `CLAUDE.md`

## Overview

When a client's input reduces to the single word `debug` (after stripping markup,
case-insensitive, trimmed), the adapter must **not** forward anything to the
upstream model. Instead it returns a successful Ollama-format response whose
assistant content is the **fully compiled prompt** — the exact final `messages`
array that *would* have been sent to the model — annotated with visible text
markers that show where each `{% include %}` snippet was pulled in.

The real client (Raycast) wraps the selection in an XML-like envelope, so the
literal request content is e.g.:

```
<user_input some=attr some="another attr">
debug
</user_input>
```

Trigger detection therefore strips tags first; this stripping affects **only**
detection — the debug output still shows every message **verbatim** (with tags),
since that is what would actually be sent to the model.

The purpose is prompt debugging: seeing how a Jinja2 system prompt actually
compiles (variables substituted, includes resolved, IP routing applied) without
round-tripping through the LLM.

## Goals

- Trigger only on an **exact** match after stripping markup:
  `_strip_tags(text).strip().lower() == "debug"`.
- Return the **full final `messages` array** (system + entire conversation),
  formatted as readable role-labelled sections.
- In the system section, show the rendered prompt with **visible boundary
  markers** for the root template and every executed include (nesting included).
- Show variables already substituted and IP routing already applied (the real
  per-request compiled form).
- Surface prompt render errors inline (the broken template is exactly what the
  user wants to inspect) instead of failing.
- Work for both `/api/chat` and `/api/generate`, streaming and non-streaming.
- **Never** call the upstream OpenAI API on a debug request.

## Non-Goals

- No config flag. The feature is always on; the keyword `debug` is hardcoded.
  (Accepted trade-off: any client can read the compiled system prompt.)
- No HTML comments. HTML comments render invisibly in markdown/HTML clients
  (Open WebUI, etc.); visible text markers are used instead.
- No `prompt_caching` structural transform in the output — the system block is
  shown as readable text, with a one-line note when caching is enabled.
- No stripping of the trigger message — the trigger message is shown verbatim
  (including its `<user_input>` tags) as the last array element. Tag stripping is
  a detection-only transform and never rewrites displayed/forwarded content.
- No `remove_thinking_tags` involvement (that affects model *output*, not input).

## Behavior

### Trigger detection

`is_debug_trigger(text: str) -> bool` returns
`_strip_tags(text).strip().lower() == "debug"`, where `_strip_tags` removes every
XML/HTML-like tag via the regex `<[^>]*>` (compiled once at module level). This
unwraps the Raycast `<user_input ...>...</user_input>` envelope — including
unquoted attributes like `some=attr` — independent of the tag name. Stripping is
used **only** for this comparison; the forwarded/displayed content is never
altered.

- **`/api/chat`**: take the **last** element of `messages`; trigger only if its
  `role == "user"` **and** its `content` is a `str` that satisfies the predicate.
  Non-string content (multimodal lists) never triggers.
- **`/api/generate`**: apply the predicate to the `prompt` string.

Detection happens in the route **after** `ctx` is built and **before** dispatch
to `_call_openai_*`. On a match the route returns the debug response directly.

### Marker grammar

```
── BEGIN file: assistant.md ──            root template loaded as system_prompt_file
── END file: assistant.md ──
── BEGIN inline system_prompt ──          root template from system_prompt_inline
── END inline system_prompt ──
── include: snippets/style.md ──          start of an executed {% include %}
── end include: snippets/style.md ──      end of that include (symmetric, named)
── note: prompt_caching enabled (sent as cache_control text block) ──
── PROMPT RENDER ERROR: <message> ──      render failed; shown instead of content
═══ message[0] role=system ═══            section header per messages[] element
```

Marker names are the exact strings passed to Jinja: the root file name as written
in `system_prompt_file`, and each include target as written in `{% include "..." %}`.

### Include marking mechanism

A `_DebugMarkerLoader(FileSystemLoader)` overrides `get_source`: it calls
`super().get_source(...)` and wraps the returned `source` with begin/end markers
before returning `(wrapped_source, filename, uptodate)`. Because the markers are
literal text, they pass through rendering verbatim, and an included template
rendered inside its parent nests **automatically** — no depth tracking needed.

The loader is constructed per debug request with a `root_name`:

- `template == root_name` → `── BEGIN file: {name} ──` / `── END file: {name} ──`
- otherwise → `── include: {name} ──` / `── end include: {name} ──`

For `system_prompt_inline`, rendering goes through `env.from_string(...)` which
bypasses the loader, so `root_name=None` (every loaded template is an include) and
the inline root is wrapped manually with `── BEGIN/END inline system_prompt ──`.

`base_dir` for the debug loader is derived from the **active** environment —
`state.jinja_env.loader.searchpath[0]` — so it automatically tracks config
hot-reload, exactly like the normal render path.

### Assembly (`build_debug_content`)

1. `get_model_config(model_id, client_ip)` → `adapter_params` (so IP routing is
   already applied).
2. `_select_prompt_source(adapter_params, model_id)` → `("file", path)`,
   `("inline", text)`, or `(None, None)` (reuses the existing "both set → file
   wins, log warning" rule).
3. If a source exists: `_collect_prompt_vars(adapter_params)` → debug-render via
   the marker loader. On `PromptRenderError`, the system text becomes a single
   `── PROMPT RENDER ERROR: <msg> ──` marker (non-fatal). Then
   `place_system_message(messages, rendered_text)` positions it (replace first
   existing system message, else insert at index 0) — identical placement to the
   normal flow.
4. If no source: the array is unchanged (includes only originate from config
   templates; any client/request system message is shown plainly).
5. Format every element as:
   ```
   ═══ message[{i}] role={role} ═══
   {content}
   ```
   String `content` is shown verbatim. Non-string `content` (e.g. multimodal
   list) is shown as pretty-printed JSON.
6. If `adapter_params.get("prompt_caching")` is truthy, prepend the
   `── note: prompt_caching enabled ... ──` line inside the system section.
7. Wrap the entire assembled text in one fenced code block. The fence length is
   chosen dynamically: `max(3, longest_backtick_run_in_content + 1)` backticks,
   no language tag — preserves whitespace/newlines and prevents the prompt's own
   markdown from rendering, while never colliding with backticks in the content.

### Response shaping (routes)

OpenAI is never called. The compiled text is delivered as a normal Ollama
response, mirroring the existing runtime-error helpers:

- **Non-streaming**: `_build_debug_payload(ctx, response_key, content)` builds a
  `create_final_response(display_name, 0, 0, 0)` payload and sets
  `message`/`response` to the content; returned via `jsonify`.
- **Streaming**: `_yield_debug_chunks(ctx, response_key, make_chunk, content)`
  yields one content chunk (`make_chunk(display_name, content)`) followed by the
  final `done` marker chunk, as ndjson — modelled on
  `_yield_runtime_error_chunks`.

`make_chunk` (currently defined inside the streaming branch of `chat()` /
`generate()`) is hoisted so the debug branch can reuse it for both modes.

## Files to Change

1. **`ollama_adapter/prompt_renderer.py`**
   - Add `_DebugMarkerLoader(FileSystemLoader)` with the wrapping `get_source`.
   - Add `render_file_debug(env, template_path, variables)` and
     `render_inline_debug(env, text, variables)` — build a sibling
     `SandboxedEnvironment` reusing `env.loader.searchpath[0]` and the marker
     loader; reuse the existing `_wrap_render` error translation. `render_inline_debug`
     wraps its result with the inline root markers.

2. **`ollama_adapter/models.py`**
   - Extract `_select_prompt_source(adapter_params, model_id) -> tuple[str | None, str | None]`
     from `_resolve_system_prompt` (the inline-vs-file decision + the
     both-set warning), and refactor `_resolve_system_prompt` to call it
     (no behavior change).
   - Extract `place_system_message(messages, content) -> list[dict]` from
     `apply_system_prompt` (the replace-first-system-else-insert logic), and
     refactor `apply_system_prompt` to call it (no behavior change).

3. **`ollama_adapter/debug_prompt.py`** (new leaf module; imports `state`,
   `prompt_renderer`, and the two `models` helpers)
   - `is_debug_trigger(text: str) -> bool`
   - `last_user_text(messages: list) -> str | None`
   - `build_debug_content(messages, adapter_params, model_id) -> str` (full
     assembly + fence)

4. **`ollama_adapter/routes.py`**
   - Hoist `make_chat_chunk` / `make_generate_chunk` out of the streaming branch.
   - In `chat()` / `generate()`: after `ctx` is built, detect the trigger and
     short-circuit to a debug response.
   - Add `_debug_response(ctx, response_key, make_chunk, *, streaming)` (calls
     `get_model_config` for `adapter_params`, then `build_debug_content`),
     `_build_debug_payload`, and `_yield_debug_chunks` — paralleling the existing
     runtime-error helpers.

5. **`config-example.yml`**
   - One-line comment under `prompts` documenting the `debug` keyword behavior.

6. **`CLAUDE.md`**
   - Short note in the *Prompt Templating* / *Runtime Error Handling* area
     describing the debug short-circuit and listing the new module/functions in
     the *Key Functions* section.

## Testing (TDD)

Run via `make` targets only; finish with `make pre-commit`.

1. **`tests/test_prompt_renderer.py`**
   - Single root file → output wrapped in `── BEGIN/END file: <name> ──`.
   - Nested includes → markers nest in the correct order
     (`BEGIN file` → `include A` → `include B` → `end include B` →
     `end include A` → `END file`).
   - Inline root with an include → `── BEGIN inline system_prompt ──` wrapper plus
     `── include: ... ──` for the snippet.
   - Render error (missing include / undefined var) → `PromptRenderError` still
     raised by the debug render functions (assembly layer converts it to a marker).

2. **`tests/test_debug_prompt.py`** (new)
   - `is_debug_trigger`: `"debug"`, `"DEBUG"`, `"  Debug  "` → true; `"debug me"`,
     `"debugger"`, `""`, non-str → false.
   - `is_debug_trigger` with the Raycast envelope:
     `"<user_input some=attr some=\"x\">\ndebug\n</user_input>"` → true;
     the same envelope wrapping `please debug` → false (exact match preserved
     after stripping); envelope wrapping `debug` with surrounding inner tags
     (`<b>debug</b>`) → true.
   - `last_user_text`: returns last user string; `None` when last message is not a
     user/string.
   - `build_debug_content`: system placement (replace vs insert); full array
     including the trailing `debug` message; `prompt_caching` note appears when
     enabled; no-system-prompt case leaves the array untouched; non-string content
     rendered as JSON.
   - Dynamic fence: content containing ```` ``` ```` runs is wrapped in a longer
     fence.

3. **`tests/test_routes.py`**
   - `/api/chat` and `/api/generate` non-streaming with input `debug`: response
     carries assistant content containing the markers, **and**
     `client.chat.completions.create` is asserted **not** called.
   - Streaming variant returns ndjson with the debug content then a `done` chunk.
   - Exact-match guard: input `"debug me"` is forwarded normally (create *is*
     called).
   - Raycast envelope guard: a `<user_input>`-wrapped `debug` triggers the debug
     short-circuit (create *not* called), and its verbatim tagged content appears
     in the output.

## Risks & Mitigations

- **Markers shift template line numbers in errors.** The wrapped source adds
  marker lines, so a `PromptRenderError` line number may be off by a few. Accepted
  for a debug-only path; the error message itself is surfaced inline.
- **Prompt leakage.** Any client can read the compiled system prompt. Accepted per
  decision (no flag); the adapter is intended for self-hosted/trusted clients.
- **Backtick collision breaking the code fence.** Mitigated by dynamic fence
  sizing (longer than any backtick run in the content).
- **Marker text colliding with Jinja syntax.** Markers are plain text (`──`,
  names); they are injected into source and emitted verbatim. Filenames realistically
  never contain `{{`/`{%`. Acceptable.
- **Tag-strip regex on malformed markup.** `<[^>]*>` stops at the first `>`, so a
  `>` inside a quoted attribute value would truncate stripping. The Raycast
  envelope (`some="another attr"`) contains no such case; treated as a known,
  acceptable limitation for a detection-only heuristic.

## Backward Compatibility

Fully compatible. The refactors in `models.py` (`_select_prompt_source`,
`place_system_message`) preserve existing behavior — they only extract reusable
helpers. No config key is added, renamed, or removed. Non-debug requests are
unaffected; the only new behavior is the exact-match `debug` short-circuit.
