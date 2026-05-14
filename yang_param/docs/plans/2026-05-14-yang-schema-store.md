# YANG Schema Store + MCP/REST API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parse YANG modules via libyang, persist a normalized schema store in SQLite, and expose 15 tool functions over both MCP (stdio) and REST (FastAPI) interfaces.

**Architecture:** `indexer/` parses YANG files and builds a SQLite node store with in-memory indexes. `tools/` implements the 15 deterministic query functions against the store. `server/` wires those functions into an MCP stdio server and a FastAPI REST server — both share the same tool layer.

**Tech Stack:** `python-libyang` (libyang 2.x C binding), `sqlite3` (stdlib), `lxml` (XML build/validate), `mcp` (Anthropic MCP SDK), `fastapi` + `uvicorn`, `pytest`

---

## File Map

```
yang_param/
├── indexer/
│   ├── __init__.py
│   ├── parser.py       # libyang → raw node dicts
│   ├── normalizer.py   # raw dicts → NodeRecord dataclass + node_id
│   └── store.py        # SchemaStore: SQLite persist + in-memory index
├── tools/
│   ├── __init__.py     # init_store(), get_store()
│   ├── explore.py      # list_modules, search_nodes, find_leaf
│   ├── tree.py         # get_node, get_children, get_ancestors
│   ├── keys.py         # get_path_to_leaf, get_required_keys, resolve_instance_path
│   ├── types.py        # get_type_info, validate_value, resolve_identityref
│   └── builder.py      # build_edit_config, build_get_config, build_delete_config, validate_edit_config
├── server/
│   ├── __init__.py
│   ├── mcp_server.py   # MCP stdio server, registers all 15 tools
│   └── rest_server.py  # FastAPI REST server, same 15 endpoints
├── data/yang/          # sample YANG modules (ietf-interfaces, ietf-inet-types)
├── tests/
│   ├── conftest.py     # fixtures: yang_dir, loaded SchemaStore, sample node_ids
│   ├── test_parser.py
│   ├── test_store.py
│   ├── test_explore.py
│   ├── test_tree.py
│   ├── test_keys.py
│   ├── test_types.py
│   ├── test_builder.py
│   └── test_mcp_server.py
└── pyproject.toml
```

---

## Task 1: Project Setup + Sample YANG Files

**Files:**
- Create: `yang_param/pyproject.toml`
- Create: `yang_param/data/yang/ietf-interfaces.yang`
- Create: `yang_param/data/yang/ietf-inet-types.yang`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "yang-schema-store"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "libyang>=2.1.0",
    "lxml>=5.0",
    "mcp>=1.0",
    "fastapi>=0.111",
    "uvicorn>=0.30",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "httpx>=0.27"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create minimal ietf-inet-types.yang**

```yang
module ietf-inet-types {
  namespace "urn:ietf:params:xml:ns:yang:ietf-inet-types";
  prefix inet;
  revision 2013-07-15 { description "Initial"; }

  typedef domain-name {
    type string { length "1..253"; pattern '[\w\-\.]+'; }
  }
  typedef ipv4-address {
    type string { pattern '(([0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])(%[\p{N}\p{L}]+)?'; }
  }
}
```

- [ ] **Step 3: Create minimal ietf-interfaces.yang**

```yang
module ietf-interfaces {
  namespace "urn:ietf:params:xml:ns:yang:ietf-interfaces";
  prefix if;
  import ietf-inet-types { prefix inet; }
  revision 2018-02-20 { description "Initial"; }

  container interfaces {
    description "Interface list";
    list interface {
      key "name";
      description "Network interface entry";
      leaf name {
        type string;
        description "Interface name";
      }
      leaf description {
        type string;
        config true;
        description "Interface description";
      }
      leaf mtu {
        type uint16 { range "68..65535"; }
        config true;
        description "Interface MTU in bytes";
      }
      leaf enabled {
        type boolean;
        default true;
        config true;
        description "Whether interface is enabled";
      }
      container ipv4 {
        description "IPv4 config";
        leaf address {
          type inet:ipv4-address;
          config true;
          description "IPv4 address";
        }
      }
    }
  }
}
```

- [ ] **Step 4: Install deps**

```bash
cd yang_param
pip install -e ".[dev]"
```

Expected: no errors, `import libyang` works.

- [ ] **Step 5: Verify libyang can load the sample module**

```bash
python -c "
import libyang
ctx = libyang.Context('data/yang')
mod = ctx.load_module('ietf-interfaces')
print('loaded:', mod.name())
"
```

Expected: `loaded: ietf-interfaces`

- [ ] **Step 6: Commit**

```bash
git add yang_param/pyproject.toml yang_param/data/
git commit -m "chore(yang_param): project setup and sample YANG modules"
```

---

## Task 2: YANG Parser

**Files:**
- Create: `yang_param/indexer/__init__.py`
- Create: `yang_param/indexer/parser.py`
- Create: `yang_param/tests/conftest.py`
- Create: `yang_param/tests/test_parser.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_parser.py
from indexer.parser import parse_yang_dir

def test_parse_returns_nodes(yang_dir):
    nodes = list(parse_yang_dir(yang_dir))
    assert len(nodes) > 0

def test_node_has_required_fields(yang_dir):
    nodes = list(parse_yang_dir(yang_dir))
    node = next(n for n in nodes if n["node_kind"] == "leaf" and n["name"] == "mtu")
    assert node["schema_path"] == "/ietf-interfaces:interfaces/interface/mtu"
    assert node["module"] == "ietf-interfaces"
    assert node["node_kind"] == "leaf"
    assert node["config"] is True
    assert "uint16" in node["type_base"]

def test_list_node_has_keys(yang_dir):
    nodes = list(parse_yang_dir(yang_dir))
    lst = next(n for n in nodes if n["node_kind"] == "list")
    assert "name" in lst["keys"]
```

- [ ] **Step 2: Create conftest.py**

```python
# tests/conftest.py
import pytest
from pathlib import Path

YANG_DIR = str(Path(__file__).parent.parent / "data" / "yang")

@pytest.fixture
def yang_dir():
    return YANG_DIR
```

- [ ] **Step 3: Run test — verify fails**

```bash
cd yang_param && pytest tests/test_parser.py -v
```

Expected: `ImportError: No module named 'indexer'`

- [ ] **Step 4: Create `indexer/__init__.py`**

```python
# indexer/__init__.py
```

- [ ] **Step 5: Implement `indexer/parser.py`**

```python
# indexer/parser.py
from __future__ import annotations
import libyang
from typing import Iterator

_TYPE_BASE_MAP = {
    libyang.Type.BINARY: "binary",
    libyang.Type.BITS: "bits",
    libyang.Type.BOOL: "boolean",
    libyang.Type.DEC64: "decimal64",
    libyang.Type.EMPTY: "empty",
    libyang.Type.ENUM: "enumeration",
    libyang.Type.IDENT: "identityref",
    libyang.Type.INST: "instance-identifier",
    libyang.Type.LEAFREF: "leafref",
    libyang.Type.STRING: "string",
    libyang.Type.UNION: "union",
    libyang.Type.INT8: "int8",
    libyang.Type.INT16: "int16",
    libyang.Type.INT32: "int32",
    libyang.Type.INT64: "int64",
    libyang.Type.UINT8: "uint8",
    libyang.Type.UINT16: "uint16",
    libyang.Type.UINT32: "uint32",
    libyang.Type.UINT64: "uint64",
}


def _extract_type_info(node) -> dict:
    """Extract type constraint info from a leaf/leaf-list SNode."""
    try:
        t = node.type()
        base = t.base()
        info = {"base": _TYPE_BASE_MAP.get(base, str(base))}

        if base in (libyang.Type.INT8, libyang.Type.INT16, libyang.Type.INT32,
                    libyang.Type.INT64, libyang.Type.UINT8, libyang.Type.UINT16,
                    libyang.Type.UINT32, libyang.Type.UINT64, libyang.Type.DEC64):
            ranges = list(t.ranges())
            if ranges:
                info["range"] = [{"min": str(r.min()), "max": str(r.max())} for r in ranges]

        if base == libyang.Type.STRING:
            lengths = list(t.lengths())
            if lengths:
                info["length"] = [{"min": r.min(), "max": r.max()} for r in lengths]
            patterns = list(t.patterns())
            if patterns:
                info["pattern"] = [p.pattern() for p in patterns]

        if base == libyang.Type.ENUM:
            info["enum"] = [e.name() for e in t.enums()]

        if base == libyang.Type.LEAFREF:
            info["path"] = t.leafref_path()

        if base == libyang.Type.IDENT:
            info["identity_module"] = t.identity().module().name() if t.identity() else None

        return info
    except Exception:
        return {"base": "unknown"}


def _iter_nodes(node, parent_path: str | None) -> Iterator[dict]:
    """Recursively yield raw node dicts from a schema node."""
    try:
        keyword = node.keyword
    except Exception:
        return

    if keyword not in ("container", "list", "leaf", "leaf-list", "choice", "case"):
        return

    mod = node.module()
    path = node.schema_path()
    name = node.name()

    try:
        config = not node.config_false()
    except Exception:
        config = True

    keys = []
    if keyword == "list":
        try:
            keys = [k.name() for k in node.keys()]
        except Exception:
            pass

    type_info: dict = {}
    type_base = ""
    if keyword in ("leaf", "leaf-list"):
        type_info = _extract_type_info(node)
        type_base = type_info.get("base", "")

    try:
        description = node.description() or ""
    except Exception:
        description = ""

    try:
        mandatory = node.mandatory()
    except Exception:
        mandatory = False

    try:
        default = str(node.default()) if node.default() is not None else None
    except Exception:
        default = None

    try:
        when_expr = node.when_condition() if hasattr(node, "when_condition") else None
    except Exception:
        when_expr = None

    child_paths = []
    try:
        for child in node.children():
            child_paths.append(child.schema_path())
    except Exception:
        pass

    yield {
        "schema_path": path,
        "name": name,
        "module": mod.name(),
        "namespace": mod.ns(),
        "prefix": mod.prefix(),
        "node_kind": keyword,
        "config": config,
        "parent_path": parent_path,
        "child_paths": child_paths,
        "keys": keys,
        "type_info": type_info,
        "type_base": type_base,
        "default": default,
        "mandatory": mandatory,
        "description": description,
        "when_expr": when_expr,
        "must_exprs": [],
    }

    try:
        for child in node.children():
            yield from _iter_nodes(child, path)
    except Exception:
        pass


def parse_yang_dir(yang_dir: str, modules: list[str] | None = None) -> Iterator[dict]:
    """Load all YANG modules in yang_dir and yield raw node dicts."""
    import os
    ctx = libyang.Context(yang_dir)

    if modules:
        to_load = modules
    else:
        to_load = [
            f[:-5] for f in os.listdir(yang_dir)
            if f.endswith(".yang") and "@" not in f
        ]

    loaded = []
    for mod_name in to_load:
        try:
            loaded.append(ctx.load_module(mod_name))
        except Exception:
            pass

    for mod in loaded:
        for top_node in mod.children():
            yield from _iter_nodes(top_node, None)
```

- [ ] **Step 6: Run tests — verify pass**

```bash
cd yang_param && pytest tests/test_parser.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add yang_param/indexer/ yang_param/tests/
git commit -m "feat(indexer): implement YANG parser with libyang"
```

---

## Task 3: Node Normalizer

**Files:**
- Create: `yang_param/indexer/normalizer.py`
- Modify: `yang_param/tests/test_parser.py` (add normalizer tests)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_parser.py — append

from indexer.normalizer import normalize, NodeRecord

def test_normalize_produces_node_record(yang_dir):
    from indexer.parser import parse_yang_dir
    raw = next(
        n for n in parse_yang_dir(yang_dir)
        if n["name"] == "mtu"
    )
    record = normalize(raw, path_to_id={})
    assert isinstance(record, NodeRecord)
    assert record.node_id.startswith("ietf-interfaces:")
    assert record.schema_path == "/ietf-interfaces:interfaces/interface/mtu"
    assert record.config is True

def test_node_id_stable(yang_dir):
    from indexer.parser import parse_yang_dir
    raw = next(n for n in parse_yang_dir(yang_dir) if n["name"] == "mtu")
    r1 = normalize(raw, path_to_id={})
    r2 = normalize(raw, path_to_id={})
    assert r1.node_id == r2.node_id
```

- [ ] **Step 2: Run — verify fails**

```bash
cd yang_param && pytest tests/test_parser.py::test_normalize_produces_node_record -v
```

Expected: `ImportError: cannot import name 'normalize'`

- [ ] **Step 3: Implement `indexer/normalizer.py`**

```python
# indexer/normalizer.py
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field


@dataclass
class NodeRecord:
    node_id: str
    schema_path: str
    name: str
    module: str
    namespace: str
    prefix: str
    node_kind: str
    config: bool
    parent_id: str | None
    children_ids: list[str]
    keys: list[str]
    type_info: dict
    default: str | None
    mandatory: bool
    description: str
    when_expr: str | None
    must_exprs: list[str] = field(default_factory=list)


def _make_node_id(module: str, schema_path: str) -> str:
    digest = hashlib.sha256(schema_path.encode()).hexdigest()[:12]
    return f"{module}:{digest}"


def normalize(raw: dict, path_to_id: dict[str, str]) -> NodeRecord:
    """Convert a raw parser dict to NodeRecord. path_to_id maps schema_path→node_id for linking."""
    node_id = _make_node_id(raw["module"], raw["schema_path"])
    path_to_id[raw["schema_path"]] = node_id

    parent_id = path_to_id.get(raw["parent_path"]) if raw["parent_path"] else None
    children_ids = [
        path_to_id.get(p, _make_node_id(raw["module"], p))
        for p in raw["child_paths"]
    ]

    return NodeRecord(
        node_id=node_id,
        schema_path=raw["schema_path"],
        name=raw["name"],
        module=raw["module"],
        namespace=raw["namespace"],
        prefix=raw["prefix"],
        node_kind=raw["node_kind"],
        config=raw["config"],
        parent_id=parent_id,
        children_ids=children_ids,
        keys=raw["keys"],
        type_info=raw["type_info"],
        default=raw["default"],
        mandatory=raw["mandatory"],
        description=raw["description"],
        when_expr=raw["when_expr"],
        must_exprs=raw["must_exprs"],
    )


def normalize_all(raw_nodes: list[dict]) -> list[NodeRecord]:
    """Normalize a full list in one pass (children IDs need two passes)."""
    # Pass 1: build path→id map
    path_to_id: dict[str, str] = {}
    for raw in raw_nodes:
        path_to_id[raw["schema_path"]] = _make_node_id(raw["module"], raw["schema_path"])

    # Pass 2: normalize with known IDs
    return [normalize(raw, path_to_id) for raw in raw_nodes]
```

- [ ] **Step 4: Run tests — verify pass**

```bash
cd yang_param && pytest tests/test_parser.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add yang_param/indexer/normalizer.py yang_param/tests/test_parser.py
git commit -m "feat(indexer): add NodeRecord normalizer with stable node_id"
```

---

## Task 4: SQLite Store + In-Memory Index

**Files:**
- Create: `yang_param/indexer/store.py`
- Create: `yang_param/tests/test_store.py`
- Modify: `yang_param/tests/conftest.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_store.py
from indexer.store import SchemaStore

def test_build_and_load(yang_dir, tmp_path):
    db = str(tmp_path / "schema.db")
    store = SchemaStore(db)
    store.build(yang_dir)
    assert store.count() > 0

def test_get_by_id(loaded_store):
    results = loaded_store.search_by_name("mtu")
    assert results
    node = loaded_store.get_by_id(results[0].node_id)
    assert node.name == "mtu"

def test_search_by_name(loaded_store):
    results = loaded_store.search_by_name("interface")
    assert any(r.name == "interface" for r in results)

def test_search_by_keyword(loaded_store):
    results = loaded_store.search_by_keyword("MTU")
    assert any(r.name == "mtu" for r in results)

def test_get_by_path(loaded_store):
    node = loaded_store.get_by_path("/ietf-interfaces:interfaces/interface/mtu")
    assert node is not None
    assert node.node_kind == "leaf"

def test_list_modules(loaded_store):
    modules = loaded_store.list_modules()
    assert "ietf-interfaces" in modules
```

- [ ] **Step 2: Add `loaded_store` fixture to conftest.py**

```python
# tests/conftest.py — append

import pytest
from indexer.store import SchemaStore

@pytest.fixture(scope="session")
def loaded_store(tmp_path_factory, yang_dir):
    db = str(tmp_path_factory.mktemp("db") / "schema.db")
    store = SchemaStore(db)
    store.build(yang_dir)
    return store
```

- [ ] **Step 3: Run — verify fails**

```bash
cd yang_param && pytest tests/test_store.py -v
```

Expected: `ImportError: cannot import name 'SchemaStore'`

- [ ] **Step 4: Implement `indexer/store.py`**

```python
# indexer/store.py
from __future__ import annotations
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from indexer.normalizer import NodeRecord, normalize_all
from indexer.parser import parse_yang_dir

_DDL = """
CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY,
    schema_path TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    module TEXT NOT NULL,
    namespace TEXT NOT NULL,
    prefix TEXT NOT NULL,
    node_kind TEXT NOT NULL,
    config INTEGER NOT NULL DEFAULT 1,
    parent_id TEXT,
    children_ids TEXT NOT NULL DEFAULT '[]',
    keys TEXT NOT NULL DEFAULT '[]',
    type_info TEXT NOT NULL DEFAULT '{}',
    default_value TEXT,
    mandatory INTEGER NOT NULL DEFAULT 0,
    description TEXT NOT NULL DEFAULT '',
    when_expr TEXT,
    must_exprs TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_name ON nodes(name);
CREATE INDEX IF NOT EXISTS idx_module ON nodes(module);
CREATE INDEX IF NOT EXISTS idx_kind ON nodes(node_kind);
"""


def _row_to_record(row: dict) -> NodeRecord:
    return NodeRecord(
        node_id=row["node_id"],
        schema_path=row["schema_path"],
        name=row["name"],
        module=row["module"],
        namespace=row["namespace"],
        prefix=row["prefix"],
        node_kind=row["node_kind"],
        config=bool(row["config"]),
        parent_id=row["parent_id"],
        children_ids=json.loads(row["children_ids"]),
        keys=json.loads(row["keys"]),
        type_info=json.loads(row["type_info"]),
        default=row["default_value"],
        mandatory=bool(row["mandatory"]),
        description=row["description"] or "",
        when_expr=row["when_expr"],
        must_exprs=json.loads(row["must_exprs"]),
    )


class SchemaStore:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._cache: dict[str, NodeRecord] = {}   # node_id → NodeRecord
        self._path_index: dict[str, str] = {}     # schema_path → node_id
        self._name_index: dict[str, list[str]] = {}  # name → [node_id]

    def build(self, yang_dir: str, modules: list[str] | None = None) -> None:
        raw_nodes = list(parse_yang_dir(yang_dir, modules))
        records = normalize_all(raw_nodes)
        self._persist(records)
        self._load_indexes()

    def _persist(self, records: list[NodeRecord]) -> None:
        con = sqlite3.connect(self._db_path)
        con.executescript(_DDL)
        con.executemany(
            """INSERT OR REPLACE INTO nodes VALUES (
                :node_id, :schema_path, :name, :module, :namespace, :prefix,
                :node_kind, :config, :parent_id, :children_ids, :keys,
                :type_info, :default_value, :mandatory, :description,
                :when_expr, :must_exprs
            )""",
            [
                {
                    **{k: v for k, v in asdict(r).items()
                       if k not in ("children_ids", "keys", "type_info", "must_exprs", "default", "config", "mandatory")},
                    "children_ids": json.dumps(r.children_ids),
                    "keys": json.dumps(r.keys),
                    "type_info": json.dumps(r.type_info),
                    "must_exprs": json.dumps(r.must_exprs),
                    "default_value": r.default,
                    "config": int(r.config),
                    "mandatory": int(r.mandatory),
                }
                for r in records
            ],
        )
        con.commit()
        con.close()

    def _load_indexes(self) -> None:
        con = sqlite3.connect(self._db_path)
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM nodes").fetchall()
        con.close()

        self._cache.clear()
        self._path_index.clear()
        self._name_index.clear()

        for row in rows:
            rec = _row_to_record(dict(row))
            self._cache[rec.node_id] = rec
            self._path_index[rec.schema_path] = rec.node_id
            self._name_index.setdefault(rec.name.lower(), []).append(rec.node_id)

    def count(self) -> int:
        return len(self._cache)

    def list_modules(self) -> list[str]:
        return sorted({r.module for r in self._cache.values()})

    def get_by_id(self, node_id: str) -> NodeRecord | None:
        return self._cache.get(node_id)

    def get_by_path(self, schema_path: str) -> NodeRecord | None:
        nid = self._path_index.get(schema_path)
        return self._cache.get(nid) if nid else None

    def search_by_name(self, name: str) -> list[NodeRecord]:
        ids = self._name_index.get(name.lower(), [])
        return [self._cache[i] for i in ids if i in self._cache]

    def search_by_keyword(self, keyword: str, kind: str | None = None, top_k: int = 10) -> list[NodeRecord]:
        kw = keyword.lower()
        results = [
            r for r in self._cache.values()
            if kw in r.name.lower() or kw in r.description.lower() or kw in r.schema_path.lower()
        ]
        if kind:
            results = [r for r in results if r.node_kind == kind]
        return results[:top_k]

    def all_records(self) -> list[NodeRecord]:
        return list(self._cache.values())
```

- [ ] **Step 5: Run tests — verify pass**

```bash
cd yang_param && pytest tests/test_store.py -v
```

Expected: 6 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add yang_param/indexer/store.py yang_param/tests/test_store.py yang_param/tests/conftest.py
git commit -m "feat(indexer): SQLite store with in-memory indexes"
```

---

## Task 5: Explore Tools

**Files:**
- Create: `yang_param/tools/__init__.py`
- Create: `yang_param/tools/explore.py`
- Create: `yang_param/tests/test_explore.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_explore.py
import tools
from tools.explore import list_modules, search_nodes, find_leaf

def test_list_modules(loaded_store):
    tools.init_store_from_instance(loaded_store)
    result = list_modules()
    assert "ietf-interfaces" in result["modules"]

def test_search_nodes_by_keyword(loaded_store):
    tools.init_store_from_instance(loaded_store)
    result = search_nodes("mtu")
    assert any(n["name"] == "mtu" for n in result["nodes"])

def test_search_nodes_by_kind(loaded_store):
    tools.init_store_from_instance(loaded_store)
    result = search_nodes("interface", kind="list")
    assert all(n["node_kind"] == "list" for n in result["nodes"])

def test_find_leaf(loaded_store):
    tools.init_store_from_instance(loaded_store)
    result = find_leaf("mtu")
    assert result["node"]["name"] == "mtu"

def test_find_leaf_with_parent_hint(loaded_store):
    tools.init_store_from_instance(loaded_store)
    result = find_leaf("name", parent_hint="interface")
    assert result["node"]["schema_path"].endswith("/name")
```

- [ ] **Step 2: Run — verify fails**

```bash
cd yang_param && pytest tests/test_explore.py -v
```

Expected: `ImportError: No module named 'tools'`

- [ ] **Step 3: Create `tools/__init__.py`**

```python
# tools/__init__.py
from __future__ import annotations
from indexer.store import SchemaStore

_store: SchemaStore | None = None


def init_store(yang_dir: str, db_path: str) -> None:
    global _store
    _store = SchemaStore(db_path)
    _store.build(yang_dir)


def init_store_from_instance(store: SchemaStore) -> None:
    global _store
    _store = store


def get_store() -> SchemaStore:
    if _store is None:
        raise RuntimeError("Store not initialized. Call init_store() first.")
    return _store
```

- [ ] **Step 4: Implement `tools/explore.py`**

```python
# tools/explore.py
from __future__ import annotations
from tools import get_store


def list_modules() -> dict:
    return {"modules": get_store().list_modules()}


def search_nodes(keyword: str, kind: str | None = None, top_k: int = 10) -> dict:
    records = get_store().search_by_keyword(keyword, kind=kind, top_k=top_k)
    return {
        "nodes": [
            {
                "node_id": r.node_id,
                "name": r.name,
                "schema_path": r.schema_path,
                "node_kind": r.node_kind,
                "module": r.module,
                "description": r.description,
            }
            for r in records
        ]
    }


def find_leaf(name: str, parent_hint: str | None = None) -> dict:
    store = get_store()
    candidates = store.search_by_name(name)
    candidates = [r for r in candidates if r.node_kind in ("leaf", "leaf-list")]

    if parent_hint:
        filtered = [
            r for r in candidates
            if parent_hint.lower() in r.schema_path.lower()
        ]
        if filtered:
            candidates = filtered

    if not candidates:
        return {"node": None, "error": f"No leaf named '{name}' found"}

    r = candidates[0]
    return {
        "node": {
            "node_id": r.node_id,
            "name": r.name,
            "schema_path": r.schema_path,
            "node_kind": r.node_kind,
            "module": r.module,
            "type_info": r.type_info,
            "description": r.description,
            "candidates": len(candidates),
        }
    }
```

- [ ] **Step 5: Run tests — verify pass**

```bash
cd yang_param && pytest tests/test_explore.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add yang_param/tools/ yang_param/tests/test_explore.py
git commit -m "feat(tools): explore tools — list_modules, search_nodes, find_leaf"
```

---

## Task 6: Tree Tools

**Files:**
- Create: `yang_param/tools/tree.py`
- Create: `yang_param/tests/test_tree.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tree.py
import tools
from tools.tree import get_node, get_children, get_ancestors

def setup_function():
    pass  # loaded_store fixture handles init

def test_get_node_by_id(loaded_store):
    tools.init_store_from_instance(loaded_store)
    node = loaded_store.get_by_path("/ietf-interfaces:interfaces/interface/mtu")
    result = get_node(node.node_id)
    assert result["node"]["name"] == "mtu"

def test_get_node_by_path(loaded_store):
    tools.init_store_from_instance(loaded_store)
    result = get_node("/ietf-interfaces:interfaces/interface/mtu")
    assert result["node"]["node_kind"] == "leaf"

def test_get_children(loaded_store):
    tools.init_store_from_instance(loaded_store)
    lst = loaded_store.search_by_name("interfaces")[0]
    result = get_children(lst.node_id)
    assert any(c["name"] == "interface" for c in result["children"])

def test_get_ancestors(loaded_store):
    tools.init_store_from_instance(loaded_store)
    mtu = loaded_store.get_by_path("/ietf-interfaces:interfaces/interface/mtu")
    result = get_ancestors(mtu.node_id)
    names = [a["name"] for a in result["ancestors"]]
    assert "interface" in names
    assert "interfaces" in names
```

- [ ] **Step 2: Run — verify fails**

```bash
cd yang_param && pytest tests/test_tree.py -v
```

Expected: `ImportError: cannot import name 'get_node'`

- [ ] **Step 3: Implement `tools/tree.py`**

```python
# tools/tree.py
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

    # Fallback: search by parent_id (for nodes where children_ids may be stale)
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

    ancestors.reverse()  # root-first order
    return {"ancestors": ancestors}
```

- [ ] **Step 4: Run tests — verify pass**

```bash
cd yang_param && pytest tests/test_tree.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add yang_param/tools/tree.py yang_param/tests/test_tree.py
git commit -m "feat(tools): tree tools — get_node, get_children, get_ancestors"
```

---

## Task 7: Key/Hierarchy Tools

**Files:**
- Create: `yang_param/tools/keys.py`
- Create: `yang_param/tests/test_keys.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_keys.py
import tools
from tools.keys import get_path_to_leaf, get_required_keys, resolve_instance_path

def test_get_path_to_leaf(loaded_store):
    tools.init_store_from_instance(loaded_store)
    mtu = loaded_store.get_by_path("/ietf-interfaces:interfaces/interface/mtu")
    result = get_path_to_leaf(mtu.node_id)
    paths = [n["schema_path"] for n in result["path"]]
    assert "/ietf-interfaces:interfaces" in paths
    assert "/ietf-interfaces:interfaces/interface/mtu" in paths

def test_get_required_keys(loaded_store):
    tools.init_store_from_instance(loaded_store)
    mtu = loaded_store.get_by_path("/ietf-interfaces:interfaces/interface/mtu")
    result = get_required_keys(mtu.node_id)
    assert result["target_path"] == "/ietf-interfaces:interfaces/interface/mtu"
    assert len(result["required_keys"]) == 1
    assert result["required_keys"][0]["key_name"] == "name"

def test_resolve_instance_path_complete(loaded_store):
    tools.init_store_from_instance(loaded_store)
    mtu = loaded_store.get_by_path("/ietf-interfaces:interfaces/interface/mtu")
    result = resolve_instance_path(mtu.node_id, {"name": "eth0"})
    assert result["missing_keys"] == []
    assert "eth0" in result["instance_path"]

def test_resolve_instance_path_missing_key(loaded_store):
    tools.init_store_from_instance(loaded_store)
    mtu = loaded_store.get_by_path("/ietf-interfaces:interfaces/interface/mtu")
    result = resolve_instance_path(mtu.node_id, {})
    assert "name" in result["missing_keys"]
```

- [ ] **Step 2: Run — verify fails**

```bash
cd yang_param && pytest tests/test_keys.py -v
```

Expected: `ImportError: cannot import name 'get_path_to_leaf'`

- [ ] **Step 3: Implement `tools/keys.py`**

```python
# tools/keys.py
from __future__ import annotations
from tools import get_store
from tools.tree import get_ancestors


def get_path_to_leaf(node_id: str) -> dict:
    store = get_store()
    target = store.get_by_id(node_id)
    if not target:
        return {"path": [], "error": f"Node not found: {node_id}"}

    ancestors_result = get_ancestors(node_id)
    path_nodes = ancestors_result["ancestors"] + [{
        "node_id": target.node_id,
        "name": target.name,
        "schema_path": target.schema_path,
        "node_kind": target.node_kind,
        "module": target.module,
        "namespace": target.namespace,
        "prefix": target.prefix,
        "config": target.config,
        "keys": target.keys,
        "description": target.description,
    }]
    return {"path": path_nodes}


def get_required_keys(node_id: str) -> dict:
    store = get_store()
    target = store.get_by_id(node_id)
    if not target:
        return {"required_keys": [], "error": f"Node not found: {node_id}"}

    path_result = get_path_to_leaf(node_id)
    required_keys = []

    for node_dict in path_result["path"]:
        if node_dict["node_kind"] == "list":
            n = store.get_by_id(node_dict["node_id"])
            if n:
                for key_name in n.keys:
                    # Find the key leaf to get its type
                    key_leaf = store.get_by_path(f"{n.schema_path}/{key_name}")
                    key_type = key_leaf.type_info if key_leaf else {}
                    required_keys.append({
                        "list_path": n.schema_path,
                        "key_name": key_name,
                        "type": key_type.get("base", "string"),
                        "constraints": {k: v for k, v in key_type.items() if k != "base"},
                    })

    return {
        "target_path": target.schema_path,
        "required_keys": required_keys,
    }


def resolve_instance_path(node_id: str, key_values: dict[str, str]) -> dict:
    store = get_store()
    target = store.get_by_id(node_id)
    if not target:
        return {"instance_path": None, "missing_keys": [], "error": f"Node not found: {node_id}"}

    keys_result = get_required_keys(node_id)
    missing = [k["key_name"] for k in keys_result["required_keys"] if k["key_name"] not in key_values]

    if missing:
        return {"instance_path": None, "missing_keys": missing}

    # Build instance path by inserting key predicates at each list segment
    path_result = get_path_to_leaf(node_id)
    instance_parts = []
    for node_dict in path_result["path"]:
        n = store.get_by_id(node_dict["node_id"])
        if not n:
            continue
        segment = n.name
        if n.node_kind == "list":
            predicates = "".join(f"[{k}='{key_values[k]}']" for k in n.keys)
            segment = f"{n.name}{predicates}"
        instance_parts.append(segment)

    # Reconstruct with module prefix on first segment
    root = path_result["path"][0]
    prefix = store.get_by_id(root["node_id"]).prefix
    first = f"{prefix}:{instance_parts[0]}"
    instance_path = "/" + "/".join([first] + instance_parts[1:])

    return {"instance_path": instance_path, "missing_keys": []}
```

- [ ] **Step 4: Run tests — verify pass**

```bash
cd yang_param && pytest tests/test_keys.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add yang_param/tools/keys.py yang_param/tests/test_keys.py
git commit -m "feat(tools): key tools — get_path_to_leaf, get_required_keys, resolve_instance_path"
```

---

## Task 8: Type/Value Tools

**Files:**
- Create: `yang_param/tools/types.py`
- Create: `yang_param/tests/test_types.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_types.py
import tools
from tools.types import get_type_info, validate_value, resolve_identityref

def test_get_type_info_leaf(loaded_store):
    tools.init_store_from_instance(loaded_store)
    mtu = loaded_store.get_by_path("/ietf-interfaces:interfaces/interface/mtu")
    result = get_type_info(mtu.node_id)
    assert result["type"]["base"] == "uint16"
    assert "range" in result["type"]

def test_validate_value_valid_uint16(loaded_store):
    tools.init_store_from_instance(loaded_store)
    mtu = loaded_store.get_by_path("/ietf-interfaces:interfaces/interface/mtu")
    result = validate_value(mtu.node_id, "1500")
    assert result["valid"] is True

def test_validate_value_out_of_range(loaded_store):
    tools.init_store_from_instance(loaded_store)
    mtu = loaded_store.get_by_path("/ietf-interfaces:interfaces/interface/mtu")
    result = validate_value(mtu.node_id, "50")  # below min 68
    assert result["valid"] is False
    assert "range" in result["error"].lower()

def test_validate_value_boolean(loaded_store):
    tools.init_store_from_instance(loaded_store)
    enabled = loaded_store.get_by_path("/ietf-interfaces:interfaces/interface/enabled")
    assert validate_value(enabled.node_id, "true")["valid"] is True
    assert validate_value(enabled.node_id, "maybe")["valid"] is False
```

- [ ] **Step 2: Run — verify fails**

```bash
cd yang_param && pytest tests/test_types.py -v
```

Expected: `ImportError: cannot import name 'get_type_info'`

- [ ] **Step 3: Implement `tools/types.py`**

```python
# tools/types.py
from __future__ import annotations
import re
from tools import get_store

_INT_BASES = {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
_INT_NATIVE_RANGE = {
    "int8": (-128, 127), "int16": (-32768, 32767),
    "int32": (-2**31, 2**31-1), "int64": (-2**63, 2**63-1),
    "uint8": (0, 255), "uint16": (0, 65535),
    "uint32": (0, 4294967295), "uint64": (0, 2**64-1),
}


def get_type_info(node_id: str) -> dict:
    store = get_store()
    r = store.get_by_id(node_id)
    if not r:
        return {"type": None, "error": f"Node not found: {node_id}"}
    return {"type": r.type_info, "node_kind": r.node_kind}


def validate_value(node_id: str, value: str) -> dict:
    store = get_store()
    r = store.get_by_id(node_id)
    if not r:
        return {"valid": False, "error": f"Node not found: {node_id}"}

    t = r.type_info
    base = t.get("base", "string")

    if base in _INT_BASES:
        try:
            v = int(value)
        except ValueError:
            return {"valid": False, "error": f"Value '{value}' is not an integer"}

        ranges = t.get("range", [])
        if ranges:
            in_range = any(
                int(rng["min"]) <= v <= int(rng["max"])
                for rng in ranges
            )
            if not in_range:
                desc = " or ".join(f"{rng['min']}..{rng['max']}" for rng in ranges)
                return {"valid": False, "error": f"Value out of range: must be {desc}"}
        else:
            lo, hi = _INT_NATIVE_RANGE.get(base, (-2**63, 2**63-1))
            if not (lo <= v <= hi):
                return {"valid": False, "error": f"Value out of native range for {base}"}

    elif base == "boolean":
        if value not in ("true", "false"):
            return {"valid": False, "error": "Boolean value must be 'true' or 'false'"}

    elif base == "string":
        patterns = t.get("pattern", [])
        for pat in patterns:
            if not re.fullmatch(pat, value):
                return {"valid": False, "error": f"Value does not match pattern: {pat}"}
        lengths = t.get("length", [])
        if lengths:
            vlen = len(value)
            in_len = any(rng["min"] <= vlen <= rng["max"] for rng in lengths)
            if not in_len:
                return {"valid": False, "error": "Value length out of allowed range"}

    elif base == "enumeration":
        enums = t.get("enum", [])
        if value not in enums:
            return {"valid": False, "error": f"Value must be one of: {enums}"}

    return {"valid": True}


def resolve_identityref(type_name: str, value: str) -> dict:
    """Look up identityref candidates matching type_name across store."""
    store = get_store()
    candidates = [
        r for r in store.all_records()
        if r.type_info.get("base") == "identityref"
        and (type_name.lower() in r.schema_path.lower() or type_name.lower() in r.name.lower())
    ]
    return {
        "type": type_name,
        "value": value,
        "candidates": [{"node_id": r.node_id, "schema_path": r.schema_path} for r in candidates],
    }
```

- [ ] **Step 4: Run tests — verify pass**

```bash
cd yang_param && pytest tests/test_types.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add yang_param/tools/types.py yang_param/tests/test_types.py
git commit -m "feat(tools): type tools — get_type_info, validate_value, resolve_identityref"
```

---

## Task 9: Builder Tools

**Files:**
- Create: `yang_param/tools/builder.py`
- Create: `yang_param/tests/test_builder.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_builder.py
import tools
from tools.builder import build_edit_config, build_get_config, build_delete_config, validate_edit_config
from lxml import etree

NETCONF_NS = "urn:ietf:params:xml:ns:netconf:base:1.0"

def test_build_edit_config_merge(loaded_store):
    tools.init_store_from_instance(loaded_store)
    mtu = loaded_store.get_by_path("/ietf-interfaces:interfaces/interface/mtu")
    result = build_edit_config(mtu.node_id, {"name": "eth0"}, "1500")
    assert result["xml"] is not None
    root = etree.fromstring(result["xml"].encode())
    assert root.tag == f"{{{NETCONF_NS}}}edit-config"
    # Must contain mtu element with value 1500
    mtu_els = root.findall(".//{urn:ietf:params:xml:ns:yang:ietf-interfaces}mtu")
    assert mtu_els and mtu_els[0].text == "1500"

def test_build_edit_config_delete(loaded_store):
    tools.init_store_from_instance(loaded_store)
    mtu = loaded_store.get_by_path("/ietf-interfaces:interfaces/interface/mtu")
    result = build_edit_config(mtu.node_id, {"name": "eth0"}, None, operation="delete")
    root = etree.fromstring(result["xml"].encode())
    mtu_els = root.findall(".//{urn:ietf:params:xml:ns:yang:ietf-interfaces}mtu")
    assert mtu_els
    assert mtu_els[0].get(f"{{{NETCONF_NS}}}operation") == "delete"

def test_build_get_config(loaded_store):
    tools.init_store_from_instance(loaded_store)
    mtu = loaded_store.get_by_path("/ietf-interfaces:interfaces/interface/mtu")
    result = build_get_config(mtu.node_id, {"name": "eth0"})
    root = etree.fromstring(result["xml"].encode())
    assert root.tag == f"{{{NETCONF_NS}}}get-config"

def test_build_delete_config_startup(loaded_store):
    tools.init_store_from_instance(loaded_store)
    result = build_delete_config("startup")
    root = etree.fromstring(result["xml"].encode())
    assert root.tag == f"{{{NETCONF_NS}}}delete-config"

def test_build_delete_config_running_rejected(loaded_store):
    tools.init_store_from_instance(loaded_store)
    result = build_delete_config("running")
    assert result["xml"] is None
    assert "running" in result["error"].lower()
```

- [ ] **Step 2: Run — verify fails**

```bash
cd yang_param && pytest tests/test_builder.py -v
```

Expected: `ImportError: cannot import name 'build_edit_config'`

- [ ] **Step 3: Implement `tools/builder.py`**

```python
# tools/builder.py
from __future__ import annotations
from lxml import etree
from tools import get_store
from tools.keys import get_path_to_leaf

NETCONF_NS = "urn:ietf:params:xml:ns:netconf:base:1.0"
NC = f"{{{NETCONF_NS}}}"

_ALLOWED_DELETE_DATASTORES = ("startup", "candidate")


def _ns(module: str, store) -> str:
    """Get namespace URI for a module name."""
    for r in store.all_records():
        if r.module == module:
            return r.namespace
    return ""


def _build_ancestor_elements(path_nodes: list[dict], key_values: dict, store, operation: str | None) -> etree._Element:
    """Build nested XML elements from root ancestor down, inserting list key leaves."""
    root_el = None
    parent_el = None

    for node_dict in path_nodes:
        r = store.get_by_id(node_dict["node_id"])
        if not r:
            continue
        ns_uri = r.namespace
        el = etree.SubElement(parent_el, f"{{{ns_uri}}}{r.name}") if parent_el is not None else etree.Element(f"{{{ns_uri}}}{r.name}")
        if root_el is None:
            root_el = el

        # Insert list key children
        if r.node_kind == "list":
            for key_name in r.keys:
                key_leaf = store.get_by_path(f"{r.schema_path}/{key_name}")
                key_ns = key_leaf.namespace if key_leaf else ns_uri
                key_el = etree.SubElement(el, f"{{{key_ns}}}{key_name}")
                key_el.text = key_values.get(key_name, "")

        # Apply operation on the target (last) element
        if operation and node_dict == path_nodes[-1]:
            el.set(f"{NC}operation", operation)

        parent_el = el

    return root_el


def build_edit_config(
    target_node_id: str,
    key_values: dict[str, str],
    value: str | None,
    operation: str = "merge",
    datastore: str = "running",
) -> dict:
    store = get_store()
    target = store.get_by_id(target_node_id)
    if not target:
        return {"xml": None, "error": f"Node not found: {target_node_id}"}

    path_result = get_path_to_leaf(target_node_id)
    path_nodes = path_result["path"]

    rpc = etree.Element(f"{NC}rpc", nsmap={"nc": NETCONF_NS})
    edit = etree.SubElement(rpc, f"{NC}edit-config")
    tgt = etree.SubElement(edit, f"{NC}target")
    etree.SubElement(tgt, f"{NC}{datastore}")

    config_el = etree.SubElement(edit, f"{NC}config")
    content = _build_ancestor_elements(path_nodes, key_values, store, operation)
    if content is not None:
        config_el.append(content)

    # Set leaf value (not for delete)
    if value is not None and operation != "delete":
        leaf_el = config_el.find(f".//{{{target.namespace}}}{target.name}")
        if leaf_el is not None:
            leaf_el.text = value

    xml_str = etree.tostring(rpc, pretty_print=True, encoding="unicode")
    return {"xml": xml_str, "operation": operation, "datastore": datastore}


def build_get_config(
    target_node_id: str,
    key_values: dict[str, str] | None = None,
    datastore: str = "running",
) -> dict:
    store = get_store()
    target = store.get_by_id(target_node_id)
    if not target:
        return {"xml": None, "error": f"Node not found: {target_node_id}"}

    path_result = get_path_to_leaf(target_node_id)
    path_nodes = path_result["path"]

    rpc = etree.Element(f"{NC}rpc", nsmap={"nc": NETCONF_NS})
    get_cfg = etree.SubElement(rpc, f"{NC}get-config")
    src = etree.SubElement(get_cfg, f"{NC}source")
    etree.SubElement(src, f"{NC}{datastore}")

    filter_el = etree.SubElement(get_cfg, f"{NC}filter", type="subtree")
    content = _build_ancestor_elements(path_nodes, key_values or {}, store, None)
    if content is not None:
        filter_el.append(content)

    xml_str = etree.tostring(rpc, pretty_print=True, encoding="unicode")
    return {"xml": xml_str, "datastore": datastore}


def build_delete_config(datastore: str) -> dict:
    if datastore == "running":
        return {"xml": None, "error": "Deleting running datastore is not allowed"}
    if datastore not in _ALLOWED_DELETE_DATASTORES:
        return {"xml": None, "error": f"datastore must be one of {_ALLOWED_DELETE_DATASTORES}"}

    rpc = etree.Element(f"{NC}rpc", nsmap={"nc": NETCONF_NS})
    del_cfg = etree.SubElement(rpc, f"{NC}delete-config")
    tgt = etree.SubElement(del_cfg, f"{NC}target")
    etree.SubElement(tgt, f"{NC}{datastore}")

    xml_str = etree.tostring(rpc, pretty_print=True, encoding="unicode")
    return {"xml": xml_str, "datastore": datastore}


def validate_edit_config(xml: str) -> dict:
    """Parse and basic structure-check the XML."""
    try:
        root = etree.fromstring(xml.encode())
        tag = root.tag
        expected_tags = {f"{NC}rpc", f"{NC}edit-config", f"{NC}get-config", f"{NC}delete-config"}
        if tag not in expected_tags:
            return {"valid": False, "error": f"Unexpected root tag: {tag}"}
        return {"valid": True}
    except etree.XMLSyntaxError as e:
        return {"valid": False, "error": str(e)}
```

- [ ] **Step 4: Run tests — verify pass**

```bash
cd yang_param && pytest tests/test_builder.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Run full suite**

```bash
cd yang_param && pytest -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add yang_param/tools/builder.py yang_param/tests/test_builder.py
git commit -m "feat(tools): builder tools — build_edit_config, build_get_config, build_delete_config, validate_edit_config"
```

---

## Task 10: MCP Server

**Files:**
- Create: `yang_param/server/__init__.py`
- Create: `yang_param/server/mcp_server.py`
- Create: `yang_param/tests/test_mcp_server.py`

- [ ] **Step 1: Write failing smoke test**

```python
# tests/test_mcp_server.py
import json
import importlib

def test_mcp_server_importable():
    mod = importlib.import_module("server.mcp_server")
    assert hasattr(mod, "app")

def test_tool_list_complete():
    from server.mcp_server import TOOLS
    names = {t["name"] for t in TOOLS}
    required = {
        "list_modules", "search_nodes", "find_leaf",
        "get_node", "get_children", "get_ancestors",
        "get_path_to_leaf", "get_required_keys", "resolve_instance_path",
        "get_type_info", "validate_value", "resolve_identityref",
        "build_edit_config", "build_get_config", "build_delete_config",
        "validate_edit_config",
    }
    assert required <= names, f"Missing tools: {required - names}"
```

- [ ] **Step 2: Run — verify fails**

```bash
cd yang_param && pytest tests/test_mcp_server.py -v
```

Expected: `ModuleNotFoundError: No module named 'server'`

- [ ] **Step 3: Create `server/__init__.py`**

```python
# server/__init__.py
```

- [ ] **Step 4: Implement `server/mcp_server.py`**

```python
# server/mcp_server.py
"""MCP stdio server exposing all 15 YANG schema tools."""
from __future__ import annotations
import asyncio
import json
import os

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

import tools
from tools.explore import list_modules, search_nodes, find_leaf
from tools.tree import get_node, get_children, get_ancestors
from tools.keys import get_path_to_leaf, get_required_keys, resolve_instance_path
from tools.types import get_type_info, validate_value, resolve_identityref
from tools.builder import build_edit_config, build_get_config, build_delete_config, validate_edit_config

TOOLS = [
    {"name": "list_modules", "description": "List all loaded YANG modules", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "search_nodes", "description": "Search schema nodes by keyword", "inputSchema": {"type": "object", "properties": {"keyword": {"type": "string"}, "kind": {"type": "string"}, "top_k": {"type": "integer", "default": 10}}, "required": ["keyword"]}},
    {"name": "find_leaf", "description": "Find a leaf/leaf-list node by name", "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}, "parent_hint": {"type": "string"}}, "required": ["name"]}},
    {"name": "get_node", "description": "Get a node by node_id or schema_path", "inputSchema": {"type": "object", "properties": {"node_id_or_path": {"type": "string"}}, "required": ["node_id_or_path"]}},
    {"name": "get_children", "description": "Get children of a node", "inputSchema": {"type": "object", "properties": {"node_id": {"type": "string"}}, "required": ["node_id"]}},
    {"name": "get_ancestors", "description": "Get ancestors from root to a node", "inputSchema": {"type": "object", "properties": {"node_id": {"type": "string"}}, "required": ["node_id"]}},
    {"name": "get_path_to_leaf", "description": "Get full path from root to target node", "inputSchema": {"type": "object", "properties": {"node_id": {"type": "string"}}, "required": ["node_id"]}},
    {"name": "get_required_keys", "description": "Get all list keys required on path to a node", "inputSchema": {"type": "object", "properties": {"node_id": {"type": "string"}}, "required": ["node_id"]}},
    {"name": "resolve_instance_path", "description": "Build instance path with key predicates; returns missing keys if incomplete", "inputSchema": {"type": "object", "properties": {"node_id": {"type": "string"}, "key_values": {"type": "object"}}, "required": ["node_id", "key_values"]}},
    {"name": "get_type_info", "description": "Get type info for a leaf node", "inputSchema": {"type": "object", "properties": {"node_id": {"type": "string"}}, "required": ["node_id"]}},
    {"name": "validate_value", "description": "Validate a string value against a leaf's type constraints", "inputSchema": {"type": "object", "properties": {"node_id": {"type": "string"}, "value": {"type": "string"}}, "required": ["node_id", "value"]}},
    {"name": "resolve_identityref", "description": "Resolve identityref candidates by type name", "inputSchema": {"type": "object", "properties": {"type_name": {"type": "string"}, "value": {"type": "string"}}, "required": ["type_name", "value"]}},
    {"name": "build_edit_config", "description": "Build a NETCONF edit-config RPC XML", "inputSchema": {"type": "object", "properties": {"target_node_id": {"type": "string"}, "key_values": {"type": "object"}, "value": {"type": "string"}, "operation": {"type": "string", "default": "merge"}, "datastore": {"type": "string", "default": "running"}}, "required": ["target_node_id", "key_values"]}},
    {"name": "build_get_config", "description": "Build a NETCONF get-config RPC XML with subtree filter", "inputSchema": {"type": "object", "properties": {"target_node_id": {"type": "string"}, "key_values": {"type": "object"}, "datastore": {"type": "string", "default": "running"}}, "required": ["target_node_id"]}},
    {"name": "build_delete_config", "description": "Build a NETCONF delete-config RPC XML (startup/candidate only)", "inputSchema": {"type": "object", "properties": {"datastore": {"type": "string"}}, "required": ["datastore"]}},
    {"name": "validate_edit_config", "description": "Validate generated NETCONF XML structure", "inputSchema": {"type": "object", "properties": {"xml": {"type": "string"}}, "required": ["xml"]}},
]

_DISPATCH = {
    "list_modules": lambda a: list_modules(),
    "search_nodes": lambda a: search_nodes(**a),
    "find_leaf": lambda a: find_leaf(**a),
    "get_node": lambda a: get_node(a["node_id_or_path"]),
    "get_children": lambda a: get_children(a["node_id"]),
    "get_ancestors": lambda a: get_ancestors(a["node_id"]),
    "get_path_to_leaf": lambda a: get_path_to_leaf(a["node_id"]),
    "get_required_keys": lambda a: get_required_keys(a["node_id"]),
    "resolve_instance_path": lambda a: resolve_instance_path(a["node_id"], a["key_values"]),
    "get_type_info": lambda a: get_type_info(a["node_id"]),
    "validate_value": lambda a: validate_value(a["node_id"], a["value"]),
    "resolve_identityref": lambda a: resolve_identityref(a["type_name"], a["value"]),
    "build_edit_config": lambda a: build_edit_config(**a),
    "build_get_config": lambda a: build_get_config(**a),
    "build_delete_config": lambda a: build_delete_config(a["datastore"]),
    "validate_edit_config": lambda a: validate_edit_config(a["xml"]),
}

app = Server("yang-schema-tool")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"])
        for t in TOOLS
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    handler = _DISPATCH.get(name)
    if not handler:
        return [types.TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]
    result = handler(arguments)
    return [types.TextContent(type="text", text=json.dumps(result))]


async def _main():
    yang_dir = os.environ.get("YANG_DIR", "data/yang")
    db_path = os.environ.get("YANG_DB", "schema.db")
    tools.init_store(yang_dir, db_path)
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(_main())
```

- [ ] **Step 5: Run tests — verify pass**

```bash
cd yang_param && pytest tests/test_mcp_server.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 6: Smoke-test MCP server startup**

```bash
cd yang_param && YANG_DIR=data/yang YANG_DB=/tmp/test_schema.db timeout 3 python server/mcp_server.py || true
```

Expected: starts without crash (timeout kills it after 3s).

- [ ] **Step 7: Commit**

```bash
git add yang_param/server/ yang_param/tests/test_mcp_server.py
git commit -m "feat(server): MCP stdio server with all 16 tools registered"
```

---

## Task 11: REST Server (FastAPI)

**Files:**
- Create: `yang_param/server/rest_server.py`

- [ ] **Step 1: Write failing smoke test**

```python
# tests/test_mcp_server.py — append

def test_rest_server_importable():
    mod = importlib.import_module("server.rest_server")
    assert hasattr(mod, "app")

def test_rest_list_modules(loaded_store):
    import tools as t
    t.init_store_from_instance(loaded_store)
    from fastapi.testclient import TestClient
    from server.rest_server import app
    client = TestClient(app)
    resp = client.get("/tools/list_modules")
    assert resp.status_code == 200
    assert "ietf-interfaces" in resp.json()["modules"]

def test_rest_search_nodes(loaded_store):
    import tools as t
    t.init_store_from_instance(loaded_store)
    from fastapi.testclient import TestClient
    from server.rest_server import app
    client = TestClient(app)
    resp = client.get("/tools/search_nodes", params={"keyword": "mtu"})
    assert resp.status_code == 200
    assert any(n["name"] == "mtu" for n in resp.json()["nodes"])
```

- [ ] **Step 2: Run — verify fails**

```bash
cd yang_param && pytest tests/test_mcp_server.py::test_rest_server_importable -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `server/rest_server.py`**

```python
# server/rest_server.py
"""FastAPI REST server — same 16 tools as MCP, HTTP POST/GET interface."""
from __future__ import annotations
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import tools
from tools.explore import list_modules, search_nodes, find_leaf
from tools.tree import get_node, get_children, get_ancestors
from tools.keys import get_path_to_leaf, get_required_keys, resolve_instance_path
from tools.types import get_type_info, validate_value, resolve_identityref
from tools.builder import build_edit_config, build_get_config, build_delete_config, validate_edit_config

app = FastAPI(title="YANG Schema Tool", version="0.1.0")


# --- Explore ---

@app.get("/tools/list_modules")
def api_list_modules() -> dict:
    return list_modules()


@app.get("/tools/search_nodes")
def api_search_nodes(keyword: str, kind: str | None = None, top_k: int = 10) -> dict:
    return search_nodes(keyword, kind=kind, top_k=top_k)


@app.get("/tools/find_leaf")
def api_find_leaf(name: str, parent_hint: str | None = None) -> dict:
    return find_leaf(name, parent_hint=parent_hint)


# --- Tree ---

@app.get("/tools/get_node")
def api_get_node(node_id_or_path: str) -> dict:
    return get_node(node_id_or_path)


@app.get("/tools/get_children")
def api_get_children(node_id: str) -> dict:
    return get_children(node_id)


@app.get("/tools/get_ancestors")
def api_get_ancestors(node_id: str) -> dict:
    return get_ancestors(node_id)


# --- Keys ---

@app.get("/tools/get_path_to_leaf")
def api_get_path_to_leaf(node_id: str) -> dict:
    return get_path_to_leaf(node_id)


@app.get("/tools/get_required_keys")
def api_get_required_keys(node_id: str) -> dict:
    return get_required_keys(node_id)


class ResolveInstancePathRequest(BaseModel):
    node_id: str
    key_values: dict[str, str]


@app.post("/tools/resolve_instance_path")
def api_resolve_instance_path(req: ResolveInstancePathRequest) -> dict:
    return resolve_instance_path(req.node_id, req.key_values)


# --- Types ---

@app.get("/tools/get_type_info")
def api_get_type_info(node_id: str) -> dict:
    return get_type_info(node_id)


@app.get("/tools/validate_value")
def api_validate_value(node_id: str, value: str) -> dict:
    return validate_value(node_id, value)


@app.get("/tools/resolve_identityref")
def api_resolve_identityref(type_name: str, value: str) -> dict:
    return resolve_identityref(type_name, value)


# --- Builders ---

class EditConfigRequest(BaseModel):
    target_node_id: str
    key_values: dict[str, str]
    value: str | None = None
    operation: str = "merge"
    datastore: str = "running"


@app.post("/tools/build_edit_config")
def api_build_edit_config(req: EditConfigRequest) -> dict:
    return build_edit_config(
        req.target_node_id, req.key_values, req.value,
        operation=req.operation, datastore=req.datastore,
    )


class GetConfigRequest(BaseModel):
    target_node_id: str
    key_values: dict[str, str] | None = None
    datastore: str = "running"


@app.post("/tools/build_get_config")
def api_build_get_config(req: GetConfigRequest) -> dict:
    return build_get_config(req.target_node_id, key_values=req.key_values, datastore=req.datastore)


@app.post("/tools/build_delete_config")
def api_build_delete_config(datastore: str) -> dict:
    return build_delete_config(datastore)


class ValidateRequest(BaseModel):
    xml: str


@app.post("/tools/validate_edit_config")
def api_validate_edit_config(req: ValidateRequest) -> dict:
    return validate_edit_config(req.xml)


# --- Startup ---

@app.on_event("startup")
def startup():
    yang_dir = os.environ.get("YANG_DIR", "data/yang")
    db_path = os.environ.get("YANG_DB", "schema.db")
    if not tools._store:
        tools.init_store(yang_dir, db_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.rest_server:app", host="0.0.0.0", port=8000, reload=False)
```

- [ ] **Step 4: Run tests — verify pass**

```bash
cd yang_param && pytest tests/test_mcp_server.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Run full test suite**

```bash
cd yang_param && pytest -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add yang_param/server/rest_server.py yang_param/tests/test_mcp_server.py
git commit -m "feat(server): FastAPI REST server with 16 tool endpoints"
```

---

## Self-Review

**Spec coverage:**
- ✅ `list_modules` — Task 5
- ✅ `search_nodes`, `find_leaf` — Task 5
- ✅ `get_node`, `get_children`, `get_ancestors` — Task 6
- ✅ `get_path_to_leaf`, `get_required_keys`, `resolve_instance_path` — Task 7
- ✅ `get_type_info`, `validate_value`, `resolve_identityref` — Task 8
- ✅ `build_edit_config`, `build_get_config`, `build_delete_config`, `validate_edit_config` — Task 9
- ✅ MCP stdio server — Task 10
- ✅ REST server — Task 11
- ✅ libyang-based indexer — Tasks 2–4
- ✅ SQLite persistence + in-memory index — Task 4
- ✅ `delete-config` blocks `running` — Task 9 `build_delete_config`

**Placeholders:** None.

**Type consistency:** `NodeRecord` defined in Task 3, used in Tasks 4–9. `SchemaStore` defined in Task 4, injected via `tools.init_store_from_instance` in all tests. `get_path_to_leaf` returns `{"path": list[dict]}` — consumed by `resolve_instance_path` and `build_edit_config` correctly.
