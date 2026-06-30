# Model Selection & Single-Turn Default — Design Spec

**Date:** 2026-06-30
**Status:** Implemented
**Branch:** worktree-gemini-model-dropdown

## Why

1. The previous default model `gemini-2.0-flash` was **retired on 2026-06-01** —
   live calls would fail. Default must move to a supported model.
2. Users want to pick a model from a **dropdown** (not free-text) and **search**
   the available list rather than memorize IDs.
3. Initial functional testing should be cheap: run a **single turn** by default,
   not a full multi-turn experiment, so a first smoke test costs almost nothing.

## Decisions

- **Default model = `gemini-2.5-flash-lite`** — the cheapest GA model
  ($0.10 / 1M input, $0.40 / 1M output as of 2026-06). GA and stable (retires
  2026-10-16), so safe as a default. Preview models (3.1) are selectable but not
  the default.
- **Default turns = 1** — a single-turn smoke query for "does it work" testing.
  Note: at 1 turn stateless == delta (no history to resend), so the comparison is
  trivial; raise turns in the UI to actually measure the stateless-vs-delta gap.
- **Dropdown + search + custom** — a `<select>` populated from `/models`, a filter
  box that narrows the list client-side, and a `custom…` option that reveals a
  free-text field for any model ID not in the list.

## Model list source (`/models`)

`gemini_client.list_models()` returns `{source, default, models:[{id,label,status}]}`:

- **Live** (real `GOOGLE_CLOUD_PROJECT` + ADC creds, not mock): query Vertex
  `GET /v1beta1/publishers/google/models`, keep `gemini*` IDs, sort by ID.
- **Mock / no creds / any failure**: a curated `STATIC_MODELS` fallback (the
  current supported set). `source` records which path was taken
  (`vertex` | `static` | `static-fallback (...)`). The endpoint **always** returns
  a usable list — the UI never ends up empty.

Curated fallback (2026-06):

| id | status |
|----|--------|
| `gemini-2.5-flash-lite` | GA · cheapest (default) |
| `gemini-2.5-flash` | GA |
| `gemini-2.5-pro` | GA |
| `gemini-3.1-flash-lite` | preview |
| `gemini-3.1-pro` | preview |

## UI flow

- On load, `app.js loadModels()` fetches `/models`, fills the `<select>`, selects
  the returned `default`. The server also renders the default as the initial
  `<option>` so the control is usable before JS runs.
- **Filter models** box narrows options live (matches id or label).
- **⟳** button reloads the list (e.g. after creds become available → switches from
  static to live Vertex list).
- Selecting `custom…` reveals a free-text field; `selectedModel()` returns either
  the chosen option or the custom value, sent to `/run` as `model`.

## Endpoints / files touched

- `gemini_client.py` — `DEFAULT_MODEL`, `STATIC_MODELS`, `list_models()`.
- `app.py` — `GET /models`; `/run` default model = `DEFAULT_MODEL`, default turns = 1.
- `templates/index.html` — model `<select>`, filter box, refresh button, custom field.
- `static/app.js` — `loadModels`, `renderModels` (filter), `selectedModel`, custom toggle.

## Out of scope

- Per-model pricing in the cost estimate (still a single flat rate). A separate
  follow-up would price input/output tokens per model.
- Caching the model list; it is fetched on load and on refresh.
