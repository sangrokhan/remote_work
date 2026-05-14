# YANG Schema Viewer Design

## Overview

Add a local web-based YANG schema viewer to the existing FastAPI REST server. The viewer is a single-page app (SPA) served at `GET /` that consumes the already-implemented REST endpoints. No new backend logic, no build tooling, no npm.

## Goals

- Browse YANG schema tree (lazy-load on expand)
- Search nodes by keyword, click result to inspect
- Show full node detail in right panel
- Works locally with modularized YANG files (multi-module support already handled by the store)

## Non-Goals

- XML builder / edit-config preview
- Authentication
- Any build pipeline (Vite, webpack, npm)
- Separate frontend project

## Architecture

```
Browser → GET /           → FastAPI → viewer.html (Jinja2)
Browser → GET /tools/list_modules    (on page load)
Browser → GET /tools/get_children    (on tree node expand)
Browser → GET /tools/get_node        (on node click)
Browser → GET /tools/search_nodes    (on search input, debounced 300ms)
```

The root `/` route serves `server/templates/viewer.html`. All data comes from existing `/tools/*` endpoints. No new API endpoints needed.

## UI Layout

```
┌─────────────────────────────────────────────────────┐
│  YANG Schema Viewer     [search_________]  [Search] │
├──────────────────┬──────────────────────────────────┤
│  ▶ ietf-ifaces   │  schema_path: /ietf-interfa...   │
│    ▼ interfaces  │  module:      ietf-interfaces    │
│      ▶ interface │  kind:        leaf               │
│        ○ name    │  type:        uint16 68..65535   │
│        ○ mtu     │  mandatory:   false              │
│        ○ enabled │  description: The size of ...    │
└──────────────────┴──────────────────────────────────┘
```

- **Left panel**: expandable tree rooted at module nodes
- **Right panel**: detail view for selected node
- **Top bar**: search input + button

## Component Behavior

### Tree

- On load: call `GET /tools/list_modules` → render one root node per module, collapsed
- On expand: call `GET /tools/get_children?node_id=<id>` → append child nodes
- On click: call `GET /tools/get_node?node_id_or_path=<id>` → render detail panel
- Node icons: `▶/▼` container/list, `○` leaf/leaf-list, `◈` choice/case
- Children fetched once per node (cached in JS after first expand)

### Search

- Debounce 300ms on input
- Call `GET /tools/search_nodes?keyword=<q>&top_k=20`
- Show results below search bar as a dropdown list
- Click result: fetch node detail, highlight selected in tree (best-effort — scroll to node if visible)

### Detail Panel

Fields shown (hide if null/empty):
- `schema_path`
- `module` / `namespace` / `prefix`
- `node_kind`
- `config` (true/false)
- `keys` (for list nodes)
- `type_info` (base type + constraints: range, pattern, enum, leafref)
- `mandatory`
- `default`
- `description`
- `when` / `must`

## Files

| File | Action | Responsibility |
|---|---|---|
| `server/rest_server.py` | Modify | Add Jinja2Templates, `GET /` route |
| `server/templates/viewer.html` | Create | Full SPA — HTML + CSS + vanilla JS |
| `Dockerfile` | Create | Build image: install libyang, copy app, run uvicorn |
| `docker-compose.yml` | Create | Service definition: port 8000, volume mounts for YANG data and DB |

## Docker Deployment

### Requirements

- libyang 2.x must be installed inside the container (system package or build from source)
- `LD_LIBRARY_PATH` set correctly inside container
- `data/yang/` mounted as a volume so YANG modules can be added without rebuilding
- `schema.db` persisted via a named volume or bind mount
- Single command: `docker compose up` starts the server

### Dockerfile strategy

Use `python:3.12-slim` base. Install libyang from system packages (`libyang-dev` via apt if available, otherwise build from source). Copy app code, install Python deps via `pip install -e .`, set `LD_LIBRARY_PATH`, run `uvicorn server.rest_server:app --host 0.0.0.0 --port 8000`.

The container startup must auto-index YANG files before starting uvicorn (or on first request via lazy init — use startup event already in `rest_server.py`).

### docker-compose.yml

```yaml
services:
  yang-viewer:
    build: .
    ports:
      - "12000:8000"
    volumes:
      - ./data/yang:/app/data/yang    # YANG module files (editable without rebuild)
      - yang-db:/app/data             # Persist schema.db
    environment:
      - YANG_DIR=/app/data/yang
      - YANG_DB=/app/data/schema.db
      - LD_LIBRARY_PATH=/usr/local/lib

volumes:
  yang-db:
```

## Constraints

- No new Python dependencies beyond `jinja2` (already a FastAPI transitive dep)
- No external JS CDN calls (fully offline-capable)
- Single HTML file — all CSS and JS inline
- Target: ~300 lines HTML+CSS+JS total
- Docker image must work offline after build (no runtime CDN fetches)

## Test Coverage

No new unit tests needed — backend has no new logic. Manual verification:
1. `docker compose up --build`
2. Open `http://localhost:12000/` → tree renders with module list
3. Expand a node → children load
4. Click a leaf → detail panel populates
5. Search "mtu" → result appears, click → detail shows
