# YANG Schema Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local web-based YANG schema viewer served at `GET /` of the existing FastAPI server, plus Docker Compose deployment on port 12000.

**Architecture:** The viewer is a single HTML file with vanilla JS that calls existing REST endpoints (`/tools/list_modules`, `/tools/get_root_nodes`, `/tools/get_children`, `/tools/get_node`, `/tools/search_nodes`). One new endpoint (`GET /tools/get_root_nodes`) and one new function in `tools/tree.py` are added. The root route serves the HTML via `FileResponse`. Docker Compose runs the server on port 12000.

**Tech Stack:** Python 3.12, FastAPI, libyang (C library via apt), uvicorn, vanilla JS (no build step, no CDN, all DOM methods — no innerHTML with dynamic data), Docker Compose v2.

---

## File Map

| File | Action | What changes |
|---|---|---|
| `tools/tree.py` | Modify | Extend `_record_to_dict` with 7 new fields; add `get_root_nodes(module)` |
| `server/rest_server.py` | Modify | Add `FileResponse` import + `GET /` route; add `GET /tools/get_root_nodes` endpoint |
| `server/templates/viewer.html` | Create | Full SPA — HTML + CSS + vanilla JS (all DOM methods, no innerHTML with API data) |
| `Dockerfile` | Create | python:3.12-slim + libyang via apt + pip install + uvicorn |
| `.dockerignore` | Create | Exclude `.git`, `schema.db`, test dirs |
| `docker-compose.yml` | Create | Port 12000:8000, volumes for data/yang and schema.db |
| `tests/test_tree.py` | Modify | Add tests for `get_root_nodes` and extended `_record_to_dict` fields |
| `tests/test_rest_server.py` | Modify | Add tests for `GET /`, `GET /tools/get_root_nodes` |

---

## Task 1: Extend `_record_to_dict` and add `get_root_nodes` to `tools/tree.py`

**Files:**
- Modify: `tools/tree.py`
- Modify: `tests/test_tree.py`

**Context:** `_record_to_dict` currently returns 10 fields. The viewer needs `children_ids` (to know if a node is expandable), `type_info`, `mandatory`, `default`, `when_expr`, `must_exprs`, and `parent_id`. `get_root_nodes(module)` returns top-level nodes (parent_id=None) for one module.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_tree.py`:

```python
def test_get_node_includes_extended_fields(loaded_store):
    import tools
    tools.init_store_from_instance(loaded_store)
    from tools.tree import get_node
    any_id = list(loaded_store._cache.keys())[0]
    result = get_node(any_id)
    node = result["node"]
    for field in ("children_ids", "parent_id", "type_info", "mandatory", "default", "when_expr", "must_exprs"):
        assert field in node, f"Missing field: {field}"


def test_get_root_nodes_returns_module_roots(loaded_store):
    import tools
    tools.init_store_from_instance(loaded_store)
    from tools.tree import get_root_nodes
    result = get_root_nodes("ietf-interfaces")
    assert "nodes" in result
    assert len(result["nodes"]) > 0
    for n in result["nodes"]:
        assert n["module"] == "ietf-interfaces"
        assert n["parent_id"] is None


def test_get_root_nodes_unknown_module_returns_empty(loaded_store):
    import tools
    tools.init_store_from_instance(loaded_store)
    from tools.tree import get_root_nodes
    result = get_root_nodes("no-such-module")
    assert result["nodes"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/han/.openclaw/workspace/remote_work/yang_param
LD_LIBRARY_PATH=$HOME/.local/lib pytest tests/test_tree.py::test_get_node_includes_extended_fields tests/test_tree.py::test_get_root_nodes_returns_module_roots tests/test_tree.py::test_get_root_nodes_unknown_module_returns_empty -v
```

Expected: 3 FAILs (field missing / function not defined).

- [ ] **Step 3: Rewrite `tools/tree.py`**

Full replacement content for `tools/tree.py`:

```python
from __future__ import annotations
from tools import get_store
from indexer.normalizer import NodeRecord


def _record_to_dict(r: NodeRecord) -> dict:
    return {
        "node_id": r.node_id,
        "name": r.name,
        "schema_path": r.schema_path,
        "node_kind": r.node_kind,
        "module": r.module,
        "namespace": r.namespace,
        "prefix": r.prefix,
        "config": r.config,
        "keys": r.keys,
        "description": r.description,
        "children_ids": r.children_ids,
        "parent_id": r.parent_id,
        "type_info": r.type_info,
        "mandatory": r.mandatory,
        "default": r.default,
        "when_expr": r.when_expr,
        "must_exprs": r.must_exprs,
    }


def get_node(node_id_or_path: str) -> dict:
    store = get_store()
    r = store.get_by_id(node_id_or_path) or store.get_by_path(node_id_or_path)
    if not r:
        return {"node": None, "error": f"Node not found: {node_id_or_path}"}
    return {"node": _record_to_dict(r)}


def get_children(node_id: str) -> dict:
    store = get_store()
    parent = store.get_by_id(node_id)
    if not parent:
        return {"children": [], "error": f"Node not found: {node_id}"}

    children = []
    for cid in parent.children_ids:
        child = store.get_by_id(cid)
        if child:
            children.append(_record_to_dict(child))

    if not children:
        children = [
            _record_to_dict(r) for r in store.all_records()
            if r.parent_id == node_id
        ]

    return {"children": children}


def get_ancestors(node_id: str) -> dict:
    store = get_store()
    ancestors = []
    current = store.get_by_id(node_id)
    if not current:
        return {"ancestors": [], "error": f"Node not found: {node_id}"}

    while current and current.parent_id:
        parent = store.get_by_id(current.parent_id)
        if not parent:
            break
        ancestors.append(_record_to_dict(parent))
        current = parent

    ancestors.reverse()
    return {"ancestors": ancestors}


def get_root_nodes(module: str) -> dict:
    store = get_store()
    records = [
        r for r in store.all_records()
        if r.module == module and r.parent_id is None
    ]
    return {"nodes": [_record_to_dict(r) for r in sorted(records, key=lambda r: r.schema_path)]}
```

- [ ] **Step 4: Run tests**

```bash
LD_LIBRARY_PATH=$HOME/.local/lib pytest tests/test_tree.py -v
```

Expected: all tree tests PASS (including 3 new ones).

- [ ] **Step 5: Run full suite**

```bash
LD_LIBRARY_PATH=$HOME/.local/lib pytest tests/ -v --tb=short
```

Expected: all 59 tests PASS (56 existing + 3 new).

- [ ] **Step 6: Commit**

```bash
git add tools/tree.py tests/test_tree.py
git commit -m "feat(tree): extend _record_to_dict with viewer fields; add get_root_nodes"
```

---

## Task 2: Add viewer route and `get_root_nodes` endpoint to `rest_server.py`

**Files:**
- Modify: `server/rest_server.py`
- Modify: `tests/test_rest_server.py`

**Context:** The viewer HTML is served via `FileResponse` at `GET /`. New endpoint `GET /tools/get_root_nodes?module=X` calls `get_root_nodes` from `tools.tree`. `FileResponse` is in `fastapi.responses` — no new deps.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_rest_server.py`:

```python
def test_root_serves_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "YANG Schema Viewer" in resp.text


def test_get_root_nodes_returns_nodes(client):
    resp = client.get("/tools/get_root_nodes?module=ietf-interfaces")
    assert resp.status_code == 200
    data = resp.json()
    assert "nodes" in data
    assert len(data["nodes"]) > 0
    for n in data["nodes"]:
        assert n["module"] == "ietf-interfaces"
        assert n["parent_id"] is None


def test_get_root_nodes_unknown_module(client):
    resp = client.get("/tools/get_root_nodes?module=no-such-module")
    assert resp.status_code == 200
    assert resp.json()["nodes"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
LD_LIBRARY_PATH=$HOME/.local/lib pytest tests/test_rest_server.py::test_root_serves_html tests/test_rest_server.py::test_get_root_nodes_returns_nodes tests/test_rest_server.py::test_get_root_nodes_unknown_module -v
```

Expected: 3 FAILs.

- [ ] **Step 3: Create templates directory and minimal placeholder**

```bash
mkdir -p /home/han/.openclaw/workspace/remote_work/yang_param/server/templates
```

Write to `server/templates/viewer.html` (placeholder for now — Task 3 replaces this):

Content: `<!DOCTYPE html><html><head><title>YANG Schema Viewer</title></head><body>YANG Schema Viewer</body></html>`

- [ ] **Step 4: Update `server/rest_server.py`**

Replace the file header (imports + app definition) with:

```python
from __future__ import annotations
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

import tools
from tools.explore import list_modules, search_nodes, find_leaf
from tools.tree import get_node, get_children, get_ancestors, get_root_nodes
from tools.keys import get_path_to_leaf, get_required_keys, resolve_instance_path
from tools.types import get_type_info, validate_value, resolve_identityref
from tools.builder import build_edit_config, build_get_config, build_delete_config, validate_edit_config

_VIEWER_HTML = Path(__file__).parent / "templates" / "viewer.html"

app = FastAPI(title="YANG Schema Tool", version="0.1.0")
```

Then add these two routes immediately after `app = FastAPI(...)` and before `# --- Explore ---`:

```python
@app.get("/")
def viewer():
    return FileResponse(_VIEWER_HTML, media_type="text/html")


@app.get("/tools/get_root_nodes")
def api_get_root_nodes(module: str) -> dict:
    return get_root_nodes(module)
```

- [ ] **Step 5: Run the 3 new REST tests**

```bash
LD_LIBRARY_PATH=$HOME/.local/lib pytest tests/test_rest_server.py::test_root_serves_html tests/test_rest_server.py::test_get_root_nodes_returns_nodes tests/test_rest_server.py::test_get_root_nodes_unknown_module -v
```

Expected: all 3 PASS.

- [ ] **Step 6: Run full suite**

```bash
LD_LIBRARY_PATH=$HOME/.local/lib pytest tests/ -v --tb=short
```

Expected: all 62 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add server/rest_server.py server/templates/viewer.html tests/test_rest_server.py
git commit -m "feat(server): add GET / viewer route and GET /tools/get_root_nodes endpoint"
```

---

## Task 3: Create `server/templates/viewer.html`

**Files:**
- Replace: `server/templates/viewer.html`

**Context:** Full SPA. All JS uses DOM methods (createElement, textContent, appendChild) — no dynamic data is injected via string interpolation into HTML. Calls same-origin REST endpoints only. No CDN.

The JS structure:
- `apiFetch(path)` — wrapper around fetch, returns parsed JSON
- `kindClass(kind)` — maps node_kind to CSS class name
- `isExpandable(node)` — true when node.children_ids is non-empty or kind is container/list/choice/case
- `makeModuleRow(modName)` — module-level row; on expand calls `/tools/get_root_nodes`
- `makeNodeRow(node, depth)` — regular node row; on expand calls `/tools/get_children`; on click calls `showDetail`
- `renderField(label, value, cls)` — renders one detail row (returns null if value is empty)
- `showDetail(nodeId)` — fetches `/tools/get_node` and populates the right panel
- `doSearch()` — fetches `/tools/search_nodes`, renders results as clickable items in dropdown
- Bootstrap IIFE — fetches `/tools/list_modules`, renders module rows

- [ ] **Step 1: Write `server/templates/viewer.html`**

Write the file with these sections (HTML skeleton, then CSS, then JS):

**HTML skeleton:**
```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>YANG Schema Viewer</title>
<style>/* CSS block — see below */</style>
</head>
<body>
<div id="topbar">
  <span id="topbar-title">YANG Schema Viewer</span>
  <div id="search-wrap">
    <input id="search-input" type="text" placeholder="Search nodes..." autocomplete="off" />
    <div id="search-dropdown"></div>
  </div>
  <button id="search-btn">Search</button>
</div>
<div id="main">
  <div id="tree-panel"></div>
  <div id="detail-panel"></div>
</div>
<script>/* JS block — see below */</script>
</body>
</html>
```

**CSS** (place in the style tag):

```css
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Cascadia Code', 'Fira Code', monospace; background: #1e1e1e; color: #d4d4d4; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
#topbar { display: flex; align-items: center; gap: 8px; padding: 8px 12px; background: #252526; border-bottom: 1px solid #3c3c3c; flex-shrink: 0; }
#topbar-title { font-size: 14px; color: #9cdcfe; font-weight: bold; margin-right: auto; }
#search-wrap { position: relative; }
#search-input { background: #3c3c3c; border: 1px solid #555; color: #d4d4d4; padding: 4px 8px; font-family: inherit; font-size: 13px; width: 280px; outline: none; }
#search-input:focus { border-color: #007acc; }
#search-btn { background: #0e639c; color: #fff; border: none; padding: 5px 12px; cursor: pointer; font-family: inherit; font-size: 13px; }
#search-btn:hover { background: #1177bb; }
#search-dropdown { position: absolute; top: 100%; left: 0; width: 420px; background: #252526; border: 1px solid #555; max-height: 320px; overflow-y: auto; z-index: 100; display: none; }
.sr-item { padding: 7px 10px; cursor: pointer; border-bottom: 1px solid #2d2d2d; font-size: 12px; }
.sr-item:hover { background: #2a2d2e; }
.sr-name { color: #9cdcfe; font-weight: bold; }
.sr-kind { color: #4ec9b0; font-size: 11px; margin-left: 6px; }
.sr-path { color: #858585; font-size: 11px; display: block; margin-top: 2px; }
#main { display: flex; flex: 1; overflow: hidden; }
#tree-panel { width: 380px; min-width: 180px; border-right: 1px solid #3c3c3c; overflow-y: auto; }
#detail-panel { flex: 1; overflow-y: auto; padding: 16px 20px; }
.tn-row { display: flex; align-items: center; cursor: pointer; padding: 2px 0; min-height: 22px; }
.tn-row:hover { background: #2a2d2e; }
.tn-row.selected { background: #094771; }
.tn-toggle { width: 18px; text-align: center; color: #858585; font-size: 11px; flex-shrink: 0; }
.tn-name { font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.tn-badge { font-size: 10px; color: #6a6a6a; margin-left: 5px; flex-shrink: 0; }
.tn-children { display: none; }
.tn-children.open { display: block; }
.tn-msg { font-size: 12px; color: #6a6a6a; font-style: italic; }
.k-module { color: #c586c0; }
.k-container { color: #9cdcfe; }
.k-list { color: #4ec9b0; }
.k-leaf { color: #ce9178; }
.k-leaf-list { color: #d7ba7d; }
.k-choice { color: #c586c0; }
.k-case { color: #858585; }
.detail-hint { color: #555; text-align: center; margin-top: 60px; font-size: 13px; }
.detail-title { font-size: 16px; color: #d4d4d4; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid #3c3c3c; }
.df { margin-bottom: 12px; }
.df-label { font-size: 10px; color: #858585; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 2px; }
.df-value { font-size: 13px; word-break: break-all; }
.v-path { color: #9cdcfe; }
.v-kind { color: #4ec9b0; }
.v-true { color: #4ec9b0; }
.v-false { color: #f44747; }
.df-pre { background: #252526; padding: 8px; font-size: 12px; white-space: pre-wrap; border: 1px solid #333; font-family: inherit; }
```

**JavaScript** (place in the script tag — all DOM methods, no dynamic HTML injection):

```javascript
'use strict';

async function apiFetch(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

function kindClass(kind) {
  const m = { container:'k-container', list:'k-list', leaf:'k-leaf', 'leaf-list':'k-leaf-list', choice:'k-choice', case:'k-case' };
  return m[kind] || 'k-leaf';
}

function isExpandable(node) {
  return ['container','list','choice','case'].includes(node.node_kind)
      || (Array.isArray(node.children_ids) && node.children_ids.length > 0);
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function msgDiv(text, indent) {
  const d = el('div', 'tn-msg');
  d.textContent = text;
  d.style.paddingLeft = indent + 'px';
  return d;
}

let _selected = null;
function selectRow(row) {
  if (_selected) _selected.classList.remove('selected');
  _selected = row;
  row.classList.add('selected');
}

function buildRow(nameText, nameClass, badgeText, paddingLeft) {
  const row = el('div', 'tn-row');
  row.style.paddingLeft = paddingLeft + 'px';
  const toggle = el('span', 'tn-toggle', '▶');
  const name   = el('span', 'tn-name ' + nameClass, nameText);
  const badge  = el('span', 'tn-badge', badgeText);
  row.append(toggle, name, badge);
  return { row, toggle };
}

function makeModuleRow(modName) {
  const wrap = el('div');
  const { row, toggle } = buildRow(modName, 'k-module', 'module', 6);
  const children = el('div', 'tn-children');
  let loaded = false;

  row.addEventListener('click', async () => {
    selectRow(row);
    const open = children.classList.toggle('open');
    toggle.textContent = open ? '▼' : '▶';
    if (open && !loaded) {
      loaded = true;
      children.replaceChildren(msgDiv('Loading…', 24));
      try {
        const data = await apiFetch('/tools/get_root_nodes?module=' + encodeURIComponent(modName));
        children.replaceChildren();
        const nodes = data.nodes || [];
        nodes.forEach(n => children.appendChild(makeNodeRow(n, 1)));
        if (!nodes.length) children.appendChild(msgDiv('(empty)', 24));
      } catch (e) {
        children.replaceChildren(msgDiv('Error: ' + e.message, 24));
      }
    }
  });

  wrap.append(row, children);
  return wrap;
}

function makeNodeRow(node, depth) {
  const indent = depth * 16 + 6;
  const wrap = el('div');
  const localName = (node.schema_path || '').split('/').filter(Boolean).pop() || node.name || node.node_id;
  const expandable = isExpandable(node);
  const { row, toggle } = buildRow(localName, kindClass(node.node_kind), node.node_kind, indent);
  if (!expandable) toggle.textContent = '';
  const children = el('div', 'tn-children');
  let loaded = false;

  row.addEventListener('click', async (e) => {
    e.stopPropagation();
    selectRow(row);
    showDetail(node.node_id);
    if (!expandable) return;
    const open = children.classList.toggle('open');
    toggle.textContent = open ? '▼' : '▶';
    if (open && !loaded) {
      loaded = true;
      children.replaceChildren(msgDiv('Loading…', indent + 16));
      try {
        const data = await apiFetch('/tools/get_children?node_id=' + encodeURIComponent(node.node_id));
        children.replaceChildren();
        const kids = data.children || [];
        kids.forEach(c => children.appendChild(makeNodeRow(c, depth + 1)));
        if (!kids.length) children.appendChild(msgDiv('(empty)', indent + 16));
      } catch (e) {
        children.replaceChildren(msgDiv('Error: ' + e.message, indent + 16));
      }
    }
  });

  wrap.append(row, children);
  return wrap;
}

function renderField(label, value) {
  const isEmpty = value === null || value === undefined || value === ''
               || (Array.isArray(value) && !value.length)
               || (value && typeof value === 'object' && !Array.isArray(value) && !Object.keys(value).length);
  if (isEmpty) return null;
  const d = el('div', 'df');
  d.appendChild(el('div', 'df-label', label));
  if (typeof value === 'object') {
    const pre = el('pre', 'df-pre');
    pre.textContent = JSON.stringify(value, null, 2);
    d.appendChild(pre);
  } else {
    d.appendChild(el('div', 'df-value', String(value)));
  }
  return d;
}

function renderBoolField(label, value) {
  if (value === null || value === undefined) return null;
  const d = el('div', 'df');
  d.appendChild(el('div', 'df-label', label));
  const v = el('div', 'df-value ' + (value ? 'v-true' : 'v-false'), String(value));
  d.appendChild(v);
  return d;
}

function renderTextField(label, value, cls) {
  if (!value) return null;
  const d = el('div', 'df');
  d.appendChild(el('div', 'df-label', label));
  d.appendChild(el('div', 'df-value' + (cls ? ' ' + cls : ''), String(value)));
  return d;
}

async function showDetail(nodeId) {
  const panel = document.getElementById('detail-panel');
  panel.replaceChildren(el('p', 'detail-hint', 'Loading…'));
  try {
    const data = await apiFetch('/tools/get_node?node_id_or_path=' + encodeURIComponent(nodeId));
    const n = data.node;
    if (!n) { panel.replaceChildren(el('p', 'detail-hint', 'Node not found.')); return; }
    panel.replaceChildren();
    panel.appendChild(el('div', 'detail-title', n.name || nodeId));

    const fields = [
      renderTextField('Schema Path', n.schema_path, 'v-path'),
      renderTextField('Module', n.module),
      renderTextField('Namespace', n.namespace),
      renderTextField('Prefix', n.prefix),
      renderTextField('Kind', n.node_kind, 'v-kind'),
      renderBoolField('Config', n.config),
      renderBoolField('Mandatory', n.mandatory),
      renderTextField('Default', n.default),
      renderTextField('Keys', n.keys && n.keys.length ? n.keys.join(', ') : null),
      renderField('Type', n.type_info),
      renderTextField('Description', n.description),
      renderTextField('When', n.when_expr),
      renderTextField('Must', n.must_exprs && n.must_exprs.length ? n.must_exprs.join('; ') : null),
    ];
    fields.forEach(f => { if (f) panel.appendChild(f); });
  } catch (e) {
    panel.replaceChildren(el('p', 'detail-hint', 'Error: ' + e.message));
  }
}

let _searchTimer = null;
const searchInput    = document.getElementById('search-input');
const searchDropdown = document.getElementById('search-dropdown');

async function doSearch() {
  const q = searchInput.value.trim();
  if (!q) { searchDropdown.style.display = 'none'; return; }
  try {
    const data = await apiFetch('/tools/search_nodes?keyword=' + encodeURIComponent(q) + '&top_k=20');
    const nodes = data.nodes || [];
    searchDropdown.replaceChildren();
    if (!nodes.length) {
      const d = el('div', 'sr-item');
      d.textContent = 'No results for "' + q + '"';
      searchDropdown.appendChild(d);
    } else {
      nodes.forEach(n => {
        const item = el('div', 'sr-item');
        item.appendChild(el('span', 'sr-name', n.name || ''));
        item.appendChild(el('span', 'sr-kind', n.node_kind));
        item.appendChild(el('span', 'sr-path', n.schema_path || ''));
        item.addEventListener('click', () => {
          searchDropdown.style.display = 'none';
          showDetail(n.node_id);
        });
        searchDropdown.appendChild(item);
      });
    }
    searchDropdown.style.display = 'block';
  } catch (e) {
    searchDropdown.replaceChildren(el('div', 'sr-item', 'Error: ' + e.message));
    searchDropdown.style.display = 'block';
  }
}

searchInput.addEventListener('input', () => { clearTimeout(_searchTimer); _searchTimer = setTimeout(doSearch, 300); });
searchInput.addEventListener('keydown', e => { if (e.key === 'Enter') { clearTimeout(_searchTimer); doSearch(); } });
document.getElementById('search-btn').addEventListener('click', doSearch);
document.addEventListener('click', e => {
  if (!e.target.closest('#search-wrap') && !e.target.closest('#search-btn'))
    searchDropdown.style.display = 'none';
});

(async () => {
  const panel = document.getElementById('tree-panel');
  panel.replaceChildren(msgDiv('Loading modules…', 8));
  try {
    const data = await apiFetch('/tools/list_modules');
    panel.replaceChildren();
    const mods = data.modules || [];
    mods.forEach(mod => panel.appendChild(makeModuleRow(mod)));
    if (!mods.length) panel.appendChild(msgDiv('No modules loaded.', 8));
  } catch (e) {
    panel.replaceChildren(msgDiv('Error: ' + e.message, 8));
  }
})();
```

- [ ] **Step 2: Verify the viewer file contains "YANG Schema Viewer" (needed by the REST test)**

```bash
grep -c "YANG Schema Viewer" /home/han/.openclaw/workspace/remote_work/yang_param/server/templates/viewer.html
```

Expected: `1`

- [ ] **Step 3: Run full suite**

```bash
LD_LIBRARY_PATH=$HOME/.local/lib pytest tests/ -v --tb=short
```

Expected: all 62 tests PASS.

- [ ] **Step 4: Smoke test the viewer**

```bash
cd /home/han/.openclaw/workspace/remote_work/yang_param
LD_LIBRARY_PATH=$HOME/.local/lib YANG_DIR=data/yang YANG_DB=/tmp/viewer-smoke.db \
  uvicorn server.rest_server:app --port 8888 &
sleep 3
curl -s http://localhost:8888/ | grep -c "YANG Schema Viewer"
curl -s http://localhost:8888/tools/list_modules
curl -s "http://localhost:8888/tools/get_root_nodes?module=ietf-interfaces"
kill %1
```

Expected:
- First curl: `1`
- Second curl: `{"modules":["ietf-interfaces"]}`
- Third curl: JSON with `"nodes"` array containing at least 1 node with `"parent_id": null`

- [ ] **Step 5: Commit**

```bash
git add server/templates/viewer.html
git commit -m "feat(viewer): add YANG schema SPA viewer with tree browse and node search"
```

---

## Task 4: Add `Dockerfile`, `.dockerignore`, `docker-compose.yml`

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`
- Create: `docker-compose.yml`

**Context:** Uses `python:3.12-slim` (Debian Bookworm). `libyang-dev` (v2.1.128) is in Bookworm apt repos. Container exposes 8000; Compose maps host 12000 to container 8000. YANG files bind-mounted so modules can be added without rebuild. `schema.db` persisted in named Docker volume.

- [ ] **Step 1: Create `.dockerignore`**

File content:

```
.git
__pycache__
*.pyc
*.pyo
.pytest_cache
schema.db
*.db
.worktrees
worktrees
docs
tests
```

- [ ] **Step 2: Create `Dockerfile`**

```dockerfile
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libyang-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .

ENV YANG_DIR=/app/data/yang
ENV YANG_DB=/app/data/schema.db

EXPOSE 8000
CMD ["uvicorn", "server.rest_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

**If `libyang-dev` is not found in apt (build fails), use this Dockerfile instead (builds libyang from source):**

```dockerfile
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential cmake git pkg-config libpcre2-dev \
    && git clone --depth 1 --branch v2.1.128 \
       https://github.com/CESNET/libyang.git /tmp/ly \
    && cmake -DCMAKE_INSTALL_PREFIX=/usr/local -B /tmp/ly/build /tmp/ly \
    && cmake --build /tmp/ly/build -j$(nproc) \
    && cmake --install /tmp/ly/build \
    && ldconfig \
    && rm -rf /tmp/ly /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .

ENV YANG_DIR=/app/data/yang
ENV YANG_DB=/app/data/schema.db
ENV LD_LIBRARY_PATH=/usr/local/lib

EXPOSE 8000
CMD ["uvicorn", "server.rest_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Create `docker-compose.yml`**

```yaml
services:
  yang-viewer:
    build: .
    ports:
      - "12000:8000"
    volumes:
      - ./data/yang:/app/data/yang
      - yang-db:/app/data
    environment:
      YANG_DIR: /app/data/yang
      YANG_DB: /app/data/schema.db

volumes:
  yang-db:
```

- [ ] **Step 4: Build the image**

```bash
cd /home/han/.openclaw/workspace/remote_work/yang_param
docker compose build
```

Expected: build completes without error. If `libyang-dev` apt step fails, switch to the source-build Dockerfile variant above.

- [ ] **Step 5: Start and smoke test**

```bash
docker compose up -d
sleep 8
curl -s http://localhost:12000/ | grep -c "YANG Schema Viewer"
curl -s http://localhost:12000/tools/list_modules
docker compose down
```

Expected:
- First curl: `1`
- Second curl: `{"modules":["ietf-interfaces"]}`

- [ ] **Step 6: Commit**

```bash
git add Dockerfile .dockerignore docker-compose.yml
git commit -m "feat(docker): add Dockerfile and docker-compose for viewer on port 12000"
```

---

## Manual Verification Checklist

After all 4 tasks:

```bash
cd /home/han/.openclaw/workspace/remote_work/yang_param
docker compose up --build -d
```

Open `http://localhost:12000/`:

- [ ] Module list appears in left panel
- [ ] Click module name — expands, shows root nodes
- [ ] Click container/list node — expands, shows children
- [ ] Click leaf node — right panel shows schema_path, module, kind, type, description
- [ ] Type `mtu` in search — dropdown appears with mtu leaf
- [ ] Click search result — right panel shows mtu node details

```bash
docker compose down
```
