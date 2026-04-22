# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Mandatory workflow (from `AGENTS.md`)

Every user-facing change must follow this order:

1. Record the user's request (requirements / scope / status) in `prd.md`.
2. Implement the change.
3. Update the matching `prd.md` item to reflect the result.
4. Deploy with `docker compose up --build -d`.
5. Verify with `docker compose ps` (both services `Up`) and by hitting `http://localhost:10000`.

`prd.md` is the source of truth for feature state — keep it synchronized with code. Respond in Korean when the user writes Korean (the project's PRD, comments, and UI strings are Korean).

## Services & ports

`docker-compose.yml` runs two services:

- `chat-front` (React/Vite built into an nginx image) → `http://localhost:10000`
- `workflow-api` (FastAPI + LangGraph) → `http://localhost:10001`
  - REST: `GET /graph`, `GET /health`
  - WebSocket: `/ws/connect`

Frontend discovers the backend via `.env`:
- `VITE_WORKFLOW_WS_URL` (default `ws://localhost:10001/ws/connect`)
- `VITE_WORKFLOW_GRAPH_URL` (default `http://localhost:10001/graph`)

If `VITE_WORKFLOW_WS_URL` is unset, `App.jsx` derives the WS URL from `VITE_WORKFLOW_GRAPH_URL` by swapping scheme and path.

## Commands

- Full dev loop: `docker compose up --build -d` then `docker compose ps`
- Frontend only (host): `npm install && npm run dev` (Vite). `npm run build` / `npm run preview` are also defined. There is **no lint or test script** for the frontend.
- Backend only (host): `pip install -r backend/requirements.txt` then `uvicorn app.main:app --host 0.0.0.0 --port 8000` from inside `backend/`.
- The backend requires `uvicorn[standard]` (not plain `uvicorn`) so the `websockets` runtime is present — otherwise `/ws/connect` returns 404 with "No supported WebSocket library detected". This is a real bug that has happened; see PRD §19.

## Architecture

### Backend (`backend/`)

- `stategraph_workflow.py` — LangGraph `StateGraph` with 4 nodes: `planner → executor → (refiner | synthesizer)`, `refiner → (planner | synthesizer)`. Each node sleeps 1–5s (simulated work). `build_workflow_graph()` returns the compiled graph used for schema serialization; `run_demo_workflow_events(input)` is a **generator that yields per-node events** driving the WebSocket stream — it does not use LangGraph's own execution path, it reimplements the branching so it can emit `node_started` / `node_finished` / `node_routed` events at each stage.
- `graph_schema.py` — Converts a compiled `StateGraph` into `{nodes, edges}` JSON via introspection (handles `__start__` / `__end__` sentinels).
- `app/main.py` — FastAPI app. The WebSocket loop handles: `ping` → `pong`; `graph` / `get_graph` / `refresh` → graph payload; `{type:"run_workflow", input, run_id}` → streams `workflow_started`, many `workflow_event`s, then `workflow_complete` or `workflow_error`. Every workflow event carries `{run_id, event, node, name, stage, message, payload?}` where `stage ∈ {start, end, routing, error}`.

### Frontend (`src/App.jsx`)

One file, ~1000 lines, holds the entire UI. Key mechanisms:

- **WebSocket lifecycle** is mounted once on app start (not tied to panel open). The socket is kept open with a 20s ping. On error/close, the app falls back to `fetch(GRAPH_API_URL)` for the schema.
- **Panel state** (`isPanelOpen`) — when true, triggers `get_graph` on the existing socket (or REST fallback); when false, destroys the Cytoscape instance and clears highlights.
- **Run tracking** — each `sendMessage()` generates a `runId` (via `crypto.randomUUID()`), creates an assistant bubble tagged with that `runId`, and sends `{type:"run_workflow", run_id, input, model, response_mode, max_tokens}`. Subsequent `workflow_event` messages with matching `run_id` append lines to the same bubble. `workflowExecutionRef` (a `useRef`, not state) holds `{runId, isRunning, activeNode}` so highlight updates don't re-render.
- **Node highlight** — on `node_started` or `stage === 'start'`, the current node gets the `wf-active` Cytoscape class. Cleared on `workflow_complete` / `workflow_error` or when the panel closes. `normalizeWorkflowNodeId` maps `START`/`END` → `__start__`/`__end__` to match the backend's LangGraph sentinel names.
- **Graph rendering** — Cytoscape with the `dagre` layout (`rankDir: TB`, `fit: false`, fixed `zoom: 1` so nodes don't auto-shrink). Nodes are fixed 180×80 rounded rectangles; the palette per node kind lives in `WORKFLOW_NODE_PALETTE`. Edge labels for conditions are intentionally **not** rendered (PRD §14.1).
- **Cytoscape style caveats** (PRD §23): use `content: (node) => node.data('label')`, never `label:`. Avoid `max-width`, `max-height`, `shadow-*`, `wheelSensitivity` — they trigger console warnings.
- **Messages container** — custom scroll UX (PRD §20): auto-scrolls to bottom on new messages; the scrollbar is hidden unless the user is actively scrolling an overflowing list (`messages-scrollbar-visible` class toggled with a 1.2s timer).

### nginx (`nginx.conf`)

Serves the built SPA. `index.html` is force-`no-store` so redeploys don't serve a stale shell referencing missing hashed bundles; JS/CSS are cached immutably.

## Reference material

`langgraph_vis/` is the **original** reference implementation the backend was ported from (see PRD §12). Do not edit it as production code — treat it as a spec/reference. The current backend lives in `backend/`.
