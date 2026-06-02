# Design: Built-in Date/Time Placeholders in Prompts

**Date:** 2026-06-03
**Status:** Approved design, pending implementation
**Area:** `ollama_adapter/models.py`, `ollama_adapter/config.py`, `config-example.yml`, `CLAUDE.md`, `README.md`

## Overview

Today the only way to put a date into a rendered system prompt is to declare a
static variable in `prompts.vars` (e.g. `year: 2026`) and template `{{ year }}`.
The value never updates on its own.

This change injects **current** date/time into every rendered prompt as built-in
Jinja2 variables, computed fresh on each request (prompts already re-render per
request). Three layers of access are provided: a raw `now` datetime object for
full formatting power, flat scalar variables for convenience, and ready-made
human-readable / ISO presets.

## Goals

- Expose current date/time to every system prompt without manual config.
- Cover the explicitly requested parts: year, month, day, hour, minute, weekday.
- Provide human-readable preset strings combining those parts.
- Configurable timezone; deterministic, locale-independent output.
- Full backward compatibility: existing static `{{ year }}` configs keep working.

## Non-Goals

- Localized (non-English) month/weekday names. English only (per decision).
  Locale-aware formatting remains available to users via `{{ now.strftime(...) }}`.
- Per-model timezone overrides. Single global `prompts.timezone`.
- 12-hour / seconds variants as presets. (Available via `now.strftime`.)
- Adding numeric `month_num` / `weekday_num` flat vars — numeric forms are
  reachable through the `now` object (`{{ now.month }}`, `{{ now.isoweekday() }}`).

## Built-in Variable Reference

All variables below are injected as a base layer before user-defined vars.
Example moment used throughout: **Monday, 2 June 2026, 09:05** in the configured
timezone.

### `now` object — timezone-aware `datetime`

Full formatting flexibility for power users:

| Expression | Result |
|---|---|
| `{{ now.year }}` | `2026` |
| `{{ now.month }}` | `6` (numeric) |
| `{{ now.day }}` | `2` |
| `{{ now.strftime('%H:%M') }}` | `09:05` |
| `{{ now.strftime('%A') }}` | `Monday` (locale-aware) |
| `{{ now.isoweekday() }}` | `1` |

### Flat scalar variables (English names, deterministic)

| Variable | Type | Example |
|---|---|---|
| `year` | int | `2026` |
| `month` | str (English name) | `June` |
| `day` | int (day of month) | `2` |
| `hour` | int (0–23) | `9` |
| `minute` | int (0–59) | `5` |
| `weekday` | str (English name) | `Monday` |

`hour` / `minute` are plain ints (render as `9` / `5`). For zero-padded time use a
preset or `now.strftime('%H:%M')`.

### Preset variables (combined, ready-to-use)

| Variable | Example | Construction |
|---|---|---|
| `date_human` | `Monday, June 2, 2026` | `f"{weekday}, {month} {day}, {year}"` |
| `time_human` | `09:05` | `f"{hour:02d}:{minute:02d}"` |
| `datetime_human` | `Monday, June 2, 2026, 09:05` | `f"{date_human}, {time_human}"` |
| `date_iso` | `2026-06-02` | `now.date().isoformat()` |
| `datetime_iso` | `2026-06-02 09:05` | `f"{date_iso} {time_human}"` |

Full injected dict (12 keys): `now`, `year`, `month`, `day`, `hour`, `minute`,
`weekday`, `date_human`, `time_human`, `datetime_human`, `date_iso`, `datetime_iso`.

## Behavior

### Computation point

`_build_datetime_vars()` (new, in `models.py`) computes the dict above and is
injected as the **base layer** of `_collect_prompt_vars()`, before global and
per-model vars are merged on top.

### Precedence (collision policy)

Layering order (low → high), unchanged except for the new base:

```
built-in datetime vars  →  prompts.vars  →  model.prompt_vars  →  ip_routing.prompt_vars
```

`ip_routing.prompt_vars` is already shallow-merged into `model.prompt_vars` upstream
in `apply_ip_routing` (`models.py`), so `_collect_prompt_vars` sees only two user
layers. **User vars always win** over built-ins: a config with a static
`prompts.vars.year: 2026` continues to override the built-in `year` exactly as
before — this is what preserves backward compatibility.

### Timezone

- New optional key `prompts.timezone`: an IANA name (e.g. `Europe/Moscow`).
  Default `UTC`.
- Resolved per request via `ZoneInfo(tz_name)` (instances are cached by Python),
  mirroring the existing pattern in `tracing.py`.
- Validated at config load in `_validate_prompts_section`, mirroring the existing
  `tracing.timezone` convention:
  - **Non-string** → raise `ValueError("prompts.timezone must be a string")` (loud
    config-type error, consistent with tracing).
  - **String but unresolvable** IANA name (e.g. `Mars/Phobos`) → `ZoneInfo(...)`
    raises `ZoneInfoNotFoundError`; **warn and drop the key** (falls back to `UTC`).
    Rationale: a typo in a zone name must not crash load nor break rendering
    (render failures surface to API clients).
- Defensive fallback at use site as well: `_build_datetime_vars` wraps
  `ZoneInfo(tz_name)` in try/except; on failure it logs a warning and uses `UTC`.
  This keeps the function correct even when `state.CONFIG` is set programmatically
  (e.g. in tests) bypassing load-time validation.

### Locale / platform independence

`month` and `weekday` names come from hardcoded English tuples
(`_MONTH_NAMES`, `_WEEKDAY_NAMES`), **not** `strftime('%B'/'%A')`, because those
depend on the process `LC_TIME` locale (Docker containers may differ). Presets are
likewise assembled from integer components and these tuples, avoiding the
platform-specific `%-d` directive. Result is identical on every OS/locale.

## Files to Change

1. **`ollama_adapter/models.py`**
   - Add `from zoneinfo import ZoneInfo`.
   - Add module-level `_MONTH_NAMES` (12, January–December) and `_WEEKDAY_NAMES`
     (7, Monday–Sunday).
   - Add `_build_datetime_vars() -> dict[str, Any]`: reads `prompts.timezone`,
     computes `now`, returns the 12-key dict.
   - Modify `_collect_prompt_vars()` to start from `_build_datetime_vars()` instead
     of `{}`.

2. **`ollama_adapter/config.py`**
   - Add `from zoneinfo import ZoneInfo` (and `ZoneInfoNotFoundError` handling).
   - Extend `_validate_prompts_section()` with `timezone` validation
     (warn-and-drop on non-string / unresolvable).

3. **`config-example.yml`**
   - Document `prompts.timezone` and the built-in placeholders (object, flat,
     presets) with the example table.

4. **`CLAUDE.md`**
   - Update the *Prompt Templating* and *Variable priority* notes to mention the
     built-in datetime layer and `prompts.timezone`.

5. **`README.md`**
   - In the `### Prompt Templates` section, add a **Built-in date/time variables**
     subsection (right after the *Variable priority* line, `README.md:154`) with
     the three access layers (`now` object, flat vars, presets), the example table,
     and a `prompts.timezone` snippet.
   - Update the *Variable priority* line so the built-in datetime layer is shown as
     the lowest-priority base: `built-in date/time → prompts.vars → model.prompt_vars
     → ip_routing[matched].prompt_vars`.
   - Add a one-line mention to the `## Features` list (line ~5) that prompts get
     auto-injected current date/time placeholders.

## Testing (TDD)

New / updated tests:

1. **`tests/test_models.py`**
   - `_build_datetime_vars`: patch `ollama_adapter.models.datetime` to a fixed
     moment (pattern from `test_tracing.py`), assert every flat value, every preset
     string, and that `now` is the expected tz-aware datetime.
   - Timezone applied: set `prompts.timezone` and assert `now`/values reflect it.
   - Collision: `prompts.vars.year` (and a preset key) override the built-in.
   - **Update existing** `_collect_prompt_vars` tests that assert exact equality
     (`== {}`, `== {"company_name": "Acme"}`, etc.) — results now contain datetime
     keys. Patch the clock and assert against the full expected dict, or switch to
     subset/`items() <=` assertions.

2. **Sandbox render test** (in `test_models.py` or `test_prompt_renderer.py`) —
   the key technical risk: render `{{ now.strftime('%H:%M') }}`, `{{ now.year }}`,
   `{{ month }}`, `{{ weekday }}`, `{{ date_human }}` through the real
   `SandboxedEnvironment` and assert correct output (confirms the sandbox does not
   block `datetime` attribute access / method calls).

3. **`tests/test_config.py`**
   - `prompts.timezone` valid (e.g. `Europe/Moscow`) passes validation untouched.
   - Invalid timezone → warning logged and key removed (effective default UTC).
   - Non-string timezone → same warn-and-drop.

Run via `make` targets only; finish with `make pre-commit`.

## Risks & Mitigations

- **Jinja2 sandbox blocking `datetime` methods.** `SandboxedEnvironment` only
  blocks callables flagged `unsafe_callable` / `alters_data` and underscore
  attributes; `strftime`, `weekday`, `.year`, `.date()` qualify as safe. Confirmed
  first by the sandbox render test.
- **Existing exact-equality tests breaking.** Expected and explicitly handled in
  the Testing section (patch clock, assert full dict).
- **Bad timezone crashing load/render.** Prevented by validation (warn-and-drop)
  plus a defensive UTC fallback at the use site.

## Backward Compatibility

Fully compatible. Built-ins are a new low-priority base layer; any existing
`prompts.vars` / `prompt_vars` keys (including a static `year`) override them, so
current configs behave identically. No config key is renamed or removed.
