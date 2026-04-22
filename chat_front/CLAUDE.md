# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Work approach

Before responding to any request, proactively search for a matching skill or persona and invoke it. If a skill exists that fits the task — even partially — use it. Do not rationalize skipping it.

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

- `chat-front` (React/Vite, served via `vite preview`) → `http://localhost:10000`
- `workflow-api` (FastAPI) → `http://localhost:10001`
  - `GET  /health` — liveness check
  - `GET  /graph`  — LangGraph schema `{nodes, edges}` (loaded once at startup)
  - `POST /api/run` — SSE stream; accepts `RunWorkflowRequest` JSON body, streams `run_started` then per-node events then `workflow_complete` / `workflow_error`

Frontend discovers the backend via `.env`:
- `VITE_WORKFLOW_GRAPH_URL` (default `http://localhost:10001/graph`)
- `VITE_WORKFLOW_RUN_URL`   (default `http://localhost:10001/api/run`)

## Commands

- Full dev loop: `docker compose up --build -d` then `docker compose ps`
- Frontend only (host): `npm install && npm run dev` (Vite). `npm run build` / `npm run preview` also defined. No lint or test script.
- Backend only (host): `pip install -r backend/requirements.txt` then `uvicorn app.main:app --host 0.0.0.0 --port 8000` from inside `backend/`.

## Source tree

```
chat_front/
├── backend/                        # FastAPI service
│   ├── app/
│   │   ├── main.py                 # FastAPI app; GET /graph, POST /api/run (SSE)
│   │   └── models.py               # RunWorkflowRequest (model, agentic_rag, response_mode, max_tokens, api_url, api_key)
│   ├── graph_schema.py             # StateGraph → {nodes, edges} JSON
│   ├── stategraph_workflow.py      # Demo graph + run_demo_workflow_events_async (thread+asyncio.Queue)
│   └── requirements.txt
│
├── langgraph_flow/                      # Real LangGraph agent implementation (WIP, all files empty)
│   ├── agents/
│   │   ├── graph.py                # Compiled StateGraph wiring all nodes/edges
│   │   ├── state.py                # AgentState TypedDict
│   │   ├── nodes/
│   │   │   ├── planner_node.py     # Generates execution plan
│   │   │   ├── executor_node.py    # Executes plan step
│   │   │   ├── refiner_node.py     # Refines executor output
│   │   │   ├── synthesizer_node.py # Final answer synthesis
│   │   │   ├── retriever_node.py   # Agentic RAG retrieval
│   │   │   ├── var_binder_node.py  # Binds retrieved vars to state
│   │   │   └── var_constructor_node.py  # Constructs query variables
│   │   └── edges/
│   │       └── routing_logic.py    # Conditional edge functions
│   ├── core/
│   │   ├── base.py                 # BaseLLM abstract class (generate(prompt, context) -> str)
│   │   ├── factory.py              # get_llm(model_name, api_url, api_key) -> BaseLLM
│   │   └── bge3_provider.py        # BGE3 embedding provider for RAG
│   ├── prompts/
│   │   ├── planner.py
│   │   ├── executor.py
│   │   ├── refiner.py
│   │   ├── synthesizer.py
│   │   ├── var_binder.py
│   │   └── var_constructor.py
│   └── tools/
│       └── registry.py             # Tool registry for agent tool use
│
├── frontend/                       # React/Vite frontend
│   ├── App.jsx                     # Root component; split-mode, sendMessage, panel state
│   ├── constants.js                # MODEL_LIST, PANEL_WIDTH, initial messages
│   ├── styles.css
│   ├── components/
│   │   ├── ChatPane.jsx            # Single chat panel (messages + header)
│   │   ├── Composer.jsx            # Input box + model selector + send button
│   │   ├── MessageBubble.jsx       # User/assistant message rendering
│   │   ├── PaneHeader.jsx          # Split-mode per-pane header (model select, RAG toggle)
│   │   ├── SettingsModal.jsx       # response_mode, max_tokens settings
│   │   └── WorkflowPanel.jsx       # Cytoscape graph visualization panel
│   ├── hooks/
│   │   ├── useWorkflowSocket.js    # Graph-only: loadWorkflowGraph (REST), highlight utils
│   │   ├── useWorkflowSSE.js       # POST /api/run + ReadableStream SSE parser; panel-bound callbacks
│   │   ├── useWorkflowGraph.js     # Cytoscape instance lifecycle (create/destroy on isPanelOpen)
│   │   └── useScrollBehavior.js    # Auto-scroll + scrollbar visibility toggle
│   └── utils/
│       └── nodeUtils.js            # normalizeNodeId, resolveNodeVisual, palette
│
├── docker-compose.yml
├── Dockerfile                      # Frontend container
├── backend/Dockerfile              # Backend container
├── prd.md                          # Feature state source of truth
└── docs/superpowers/specs/
    └── 2026-04-22-llm-factory-design.md  # LLM factory + model integration spec
```

## Architecture

### Backend (`backend/`)

- `app/main.py` — FastAPI. `GET /graph` returns static schema. `POST /api/run` accepts `RunWorkflowRequest`, emits SSE: `run_started` (model/RAG info) → many `workflow_event`s → `workflow_complete` / `workflow_error`. Each event: `{event, node, name, stage, message, payload?}` where `stage ∈ {start, end, routing, error}`.
- `app/models.py` — `RunWorkflowRequest`: `run_id, input, model, response_mode, max_tokens, agentic_rag, api_url, api_key`.
- `stategraph_workflow.py` — Demo `StateGraph` (4 nodes, random routing, simulated delays). `run_demo_workflow_events_async(req)` wraps the blocking sync generator in a daemon thread + `asyncio.Queue` so FastAPI can stream without blocking the event loop.
- `graph_schema.py` — Introspects compiled `StateGraph` → `{nodes, edges}` JSON; handles `__start__`/`__end__` sentinels.

### LangGraph (`langgraph_flow/`)

Real agent implementation — currently empty stubs. Intended to replace the demo workflow in `stategraph_workflow.py`.

- `core/base.py` — `BaseLLM` ABC with `generate(prompt, context) -> str`
- `core/factory.py` — `get_llm(model_name, api_url, api_key)` dispatches to concrete model class
- `core/bge3_provider.py` — BGE3 embedding provider for agentic RAG retrieval
- `agents/graph.py` — Full graph: `planner → executor → (refiner | synthesizer)`, with optional `retriever → var_constructor → var_binder` RAG path when `agentic_rag=true`
- `agents/state.py` — `AgentState` TypedDict shared across all nodes
- `agents/nodes/` — One file per node; each receives `(state, config)` where `config["configurable"]["llm"]` is the factory-injected LLM
- `prompts/` — Per-node prompt strings

### Frontend (`frontend/`)

- **Split mode** — `isResultPanelOpen` activates two `ChatPane`s side-by-side. `sendMessage()` fires two independent `POST /api/run` SSE streams with per-panel `model` and `agentic_rag`. Each stream's callbacks are closed over its panel's `setMessages` — no run_id routing needed.
- **SSE flow** — `useWorkflowSSE.streamWorkflow()` fetches `/api/run`, parses `ReadableStream` line-by-line. `run_started` → prepends model/RAG info line. `node_started` → calls `applyWorkflowNodeHighlight`. `workflow_complete` / `workflow_error` → clears highlight, sets bubble status.
- **Graph panel** — `useWorkflowSocket.loadWorkflowGraph()` fetches `GET /graph` (REST only, no WS). `useWorkflowGraph` creates/destroys Cytoscape instance on `isPanelOpen`. `cyRef` lives in `App.jsx` and is passed to both hooks so highlights and the graph share the same instance.
- **Cytoscape caveats**: use `content: (node) => node.data('label')`, never `label:`. Avoid `max-width`, `max-height`, `shadow-*`, `wheelSensitivity`.

## Reference material

LLM factory design spec: `docs/superpowers/specs/2026-04-22-llm-factory-design.md`
