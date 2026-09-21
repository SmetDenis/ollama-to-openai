# Design: Client Label Prefix Cleanup (`Text:` / `Текст:`)

**Date:** 2026-09-21
**Status:** Implemented
**Area:** `ollama_adapter/input_cleanup.py` (new), `ollama_adapter/config.py`, `ollama_adapter/routes.py`, `ollama_adapter/openai_routes.py`, `config-example.yml`, `README.md`, `CLAUDE.md`

## Problem

Some clients prepend a label line to the text they send. Raycast sends:

```json
{"role": "user", "content": "Text:\ndebug"}
```

The label is pure noise and it hurts twice:

1. **Debug detection breaks.** `is_debug_trigger("Text:\ndebug")` is false, because the
   text no longer reduces to `debug`. The user typed `debug` and got a model answer.
2. **The model sees the label.** It burns tokens and can steer the answer ("Text:" reads
   as an instruction-ish header).

Removal must therefore happen **before** debug detection and **before** the upstream
request is assembled — i.e. right after input validation in the route handlers.

## Solution

A new leaf module `ollama_adapter/input_cleanup.py` (imports only `state`) exposing:

- `strip_input_prefix(text)` — removes one leading label line from a plain string;
- `clean_last_user_message(messages)` — applies it to the last message iff that message
  has `role: "user"`; returns the original list unchanged when nothing matched, and never
  mutates the input (a hit produces a new list with a copied last message);
- `configured_prefixes()` / `DEFAULT_STRIP_PREFIXES` / `INPUT_CLEANUP_KEYS` — config surface.

### Matching rule

Per prefix, the compiled matcher is:

```python
re.compile(rf"^\s*{re.escape(prefix)}[ \t]*\r?\n\s*", re.IGNORECASE)
```

That is: optional leading whitespace, the literal prefix, optional spaces/tabs, a line
break (LF or CRLF), then any blank lines. Case-insensitive, so `text:` and `ТЕКСТ:` match.
The label must own the whole first line. A match is **rejected** when only whitespace would
remain (`"Text:"` alone stays as-is rather than becoming an empty request).

Only the **first** text part of an OpenAI content-part list is considered — a later part is
not at the start of the message.

### Wiring (4 entry points)

| Endpoint | Call |
|---|---|
| `/api/chat` | `messages = clean_last_user_message(messages)` after the `messages` validation |
| `/api/generate` | `prompt = strip_input_prefix(prompt)` after the `prompt` validation |
| `/v1/chat/completions` | `messages = clean_last_user_message(messages)` before `last_user_text` |
| `/v1/completions` | `prompt = strip_input_prefix(_parse_prompt(...))` |

In every case the cleanup sits **before** the debug check and before `_complete` /
`_CompletionContext`, so the debug dump also shows the already-cleaned messages — which is
what actually goes upstream.

`/api/generate` and `/v1/completions` validate a non-empty prompt *before* the cleanup; the
"never strip to empty" rule keeps that invariant intact afterwards.

## Configuration

```yaml
input_cleanup:
  enabled: true              # false -> nothing is stripped
  strip_prefixes:            # replaces the defaults entirely
    - "Text:"
    - "Текст:"
```

Defaults when the section (or the key) is absent: `enabled: true`,
`strip_prefixes: ["Text:", "Текст:"]` — so it works out of the box for Raycast.
Validated in `config.py` (`_validate_input_cleanup`): dict; `enabled` bool; `strip_prefixes`
a list of non-empty strings; unknown keys warn and are ignored. Hot-reload applies as usual.

## Decisions on forks (and why)

| Fork | Decision | Rationale |
|---|---|---|
| Config section vs hardcoded list | **Config section with working defaults** | Works immediately for the Raycast case, but a new client label ("Prompt:", "Вопрос:") needs no code change. `enabled: false` is the escape hatch for anyone who wants the raw client text. |
| Which messages | **Last user message only** | That is the message the user just typed and the one debug detection reads. Rewriting the whole history would touch content the adapter never needed to touch, and the prefix in older turns is harmless. |
| Match shape | **Prefix + line break only** (label owns its line) | `"Text: some sentence"` mid-prose is legitimate user text; requiring the label to be the whole first line makes a false positive essentially impossible. Accepted cost: a single-line `"Text: debug"` is **not** stripped. |
| Markup envelope (`<user_input>Text: …</user_input>`) | **Literal start only** — not searched inside tags | Keeps the rule "what you see at position 0" predictable. Tag stripping stays a detection-only transform (per the debug-prompt spec, forwarded content is never tag-rewritten), so extending it here would have meant either mutating client markup or a second, divergent notion of "start". |
| Nothing left after stripping | **Do not strip** | A label-only message would otherwise become an empty prompt — an upstream 400 instead of a harmless echo. |
| Case sensitivity | **Case-insensitive** | Clients are inconsistent about capitalization; the label is a fixed marker, not user prose. |

## Testing

- `tests/test_input_cleanup.py` — prefix table (LF/CRLF, trailing spaces, blank lines,
  leading blank line, case, Cyrillic), the non-match table (same line, label-only,
  mid-text, inside tags, other word), config defaults/disabled/custom/malformed,
  content-part handling, non-mutation, role and shape guards.
- `tests/test_routes.py::TestInputPrefixCleanup` — `/api/chat` and `/api/generate`:
  `Text:\ndebug` reaches the debug short-circuit; the prefix is gone from the upstream
  `messages`; `enabled: false` keeps it.
- `tests/test_openai_routes.py::TestInputPrefixCleanup` — the same for
  `/v1/chat/completions` (string and content-part content) and `/v1/completions`.
- `tests/test_config.py::TestLoadConfigInputCleanup` — validation errors and the warning
  on unknown keys.

## Risks

- **Legitimate `Text:` header.** A user whose message genuinely starts with a lone `Text:`
  line loses it. Mitigated by the line-break requirement and `enabled: false`.
- **Ruff confusables.** Cyrillic literals trip `RUF001`/`RUF002`/`RUF003`; these three rules
  are now ignored project-wide in `pyproject.toml` (the project legitimately carries Russian
  labels).

## Backward Compatibility

Behavior changes only for a request whose last user message starts with a `Text:` /
`Текст:` label line: that line is dropped. Everything else — including every existing test —
is unaffected. `input_cleanup.enabled: false` restores the previous behavior exactly.
