# NETCONF Builder Page — Design Spec

**Date:** 2026-05-14  
**Scope:** Add a NETCONF edit-config XML generator tab to the existing YANG Schema Viewer single-page app.

---

## Overview

Add a "NETCONF Builder" tab to the existing topbar alongside "Tree Viewer". The tab renders a form where the user types a leaf name, picks from search results, fills in required key values and the leaf value, and generates a valid NETCONF `edit-config` RPC XML.

---

## Architecture

All logic lives in `server/templates/viewer.html` (single file, no new server routes needed). The existing REST API already exposes all required endpoints:

- `GET /tools/search_nodes?keyword=<name>&kind=leaf` — find leaf candidates
- `GET /tools/get_required_keys?node_id=<id>` — get list key names for the path
- `POST /tools/build_edit_config` — generate XML

No backend changes required.

---

## UI Structure

### Topbar

Add two tab buttons to the right of the title:

```
YANG Schema Viewer  [Tree Viewer] [NETCONF Builder]   [search...]
```

Clicking a tab hides/shows the corresponding `#main` section. Active tab is visually highlighted (blue underline or background).

### NETCONF Builder Panel

Replaces `#main` when the tab is active. Layout is a centered form, no tree panel.

**Step 1 — Leaf search**
- Text input + Search button (same style as existing search)
- Results dropdown: name, kind badge, schema_path — filtered to `kind=leaf` only
- Clicking a result selects it and moves to Step 2

**Step 2 — Form**
Shown after a leaf is selected:
- Selected leaf display: name + full schema_path (read-only, styled like existing `v-path`)
- Dynamic key inputs: one labeled input per required key, prefilled with `?`
- Value input: labeled "value"
- Operation dropdown: `merge` (default), `replace`, `delete`
- Datastore dropdown: `running` (default), `candidate`, `startup`
- Generate button

**Step 3 — Output**
Shown after Generate:
- Syntax-highlighted XML block (same `df-pre` style, monospace dark background)
- Copy button (copies raw XML to clipboard)
- Clear/Reset button to start over

---

## Data Flow

```
user types leaf name
  → GET /tools/search_nodes?keyword=<name>&kind=leaf&top_k=20
  → show dropdown

user picks leaf
  → GET /tools/get_required_keys?node_id=<id>
  → render key input fields (prefilled "?")

user fills form → clicks Generate
  → POST /tools/build_edit_config { target_node_id, key_values, value, operation, datastore }
  → render XML output
```

---

## Error Handling

- Search returns empty: show "No leaf nodes found."
- `get_required_keys` fails: show inline error, block Generate
- `build_edit_config` returns `error` field: display error message in output area instead of XML

---

## Future Hook (not in scope now)

The key input fields with `?` defaults are the LLM integration point: later, an LLM call can detect unfilled `?` values before generating and prompt the user for the missing information.

---

## Out of Scope

- `get-config` and `delete-config` builder pages
- Sending generated XML to a real NETCONF device
- Any backend changes
