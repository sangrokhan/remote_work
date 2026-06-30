# Mode Toggle · Request File · History Viewer · Graph-from-History — Design

**Date:** 2026-06-30
**Status:** Implemented

## Goals

1. Run **one mode per execution** — `stateless` OR `stateful` (toggle), not both.
   (`stateful` = the former `delta`: client sends only the new turn; renamed.)
2. Request text is **loaded from a JSON file** with a fixed string per step, so the
   request is constant even when the response size varies.
3. Tag the **pcap filename with the mode**: `capture_<mode>_<ts>_<token>.pcap`.
4. Every execution gets a unique **execution history id** (`exec_id`), stored in
   Firestore (and local JSON). Graphs are plotted **by `exec_id`**.
5. **History viewer**: list executions → click → that execution's single JSON log
   + download. One JSON document per execution.
6. **Graph loads from stored data** (Firestore preferred, recent JSON fallback):
   current run, a selected history id, or two ids compared.
7. **Backup plan**: if no history is found in Firestore (empty / unreachable),
   generate **dummy data** and mark it clearly as `dummy: true` ("DUMMY DATA").

## Data model

Each execution document (Firestore doc id = `exec_id`, same shape in JSON):

```json
{
  "exec_id": "exec_2026-06-30T..._<8hex>",
  "timestamp": "2026-06-30T...",
  "mode": "stateless" | "stateful",
  "mock": true,
  "dummy": false,
  "params": { "turns", "model", "endpoint", "request_source" },
  "summary": { "mode", "series": {...}, "totals": {...} }
}
```

`summary.series` (single mode) holds everything the chart needs:
`turns, per_turn_tokens, per_turn_wire_bytes, cum_tokens, cum_wire_bytes`.

## Request file

`requests/default.json`:
```json
{ "name": "default", "steps": [ {"text": "..."}, {"text": "..."} ] }
```
- `experiment.load_request_steps()` reads it; missing/invalid → synthetic fallback.
- `turns` defaults to `len(steps)`; capped to the step count.
- **stateless**: turn k sends steps 1..k (history grows). **stateful**: turn k
  sends only step k. Texts are fixed by the file.

## Backend

- `experiment.run_experiment(mode, model, steps)` → records for that one mode.
- `metrics.summarize(experiment)` → `{mode, series, totals}` (single mode; no
  cross-mode ratio — comparison happens by loading two history ids).
- `store`:
  - `save_run(exec_id, timestamp, mode, experiment, summary)` → doc id `exec_id`.
  - `list_runs()` → `[{exec_id, timestamp, mode, mock, totals}]`; **if empty →
    `dummy_runs()`** (two synthetic stateless/stateful execs, `dummy: true`).
  - `get_run(exec_id)` → full doc incl. `series` for plotting; dummy id supported.
- `app` routes:
  - `POST /run` — body `{mode, model, turns, capture}`; makes `exec_id`, runs,
    saves, returns the doc (+ capture info).
  - `GET /history` — list (with dummy fallback).
  - `GET /history/<exec_id>` — full doc for the viewer + chart.
  - `GET /download/run/<exec_id>` — the single JSON log.

## Frontend

- **Mode toggle** (radio: stateless / stateful) next to params.
- Chart is generalized: `plotSeries(label, series, …)` — used for the current run,
  a history id, or two ids overlaid (compare).
- **History viewer** (bottom): table of executions (exec_id, mode, time, tokens,
  mock/dummy badges) → row click loads `/history/<id>` → shows JSON + downloads +
  plots that execution. A **Compare** control picks two ids → overlaid chart.
- **DUMMY DATA** badge whenever `dummy: true`; **MOCK** badge when `mock: true`.

## Out of scope

- Real Vertex Context Caching (stateful is still client-side delta).
- Per-model pricing (flat rate kept).
