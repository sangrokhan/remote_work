# YANG Schema Store

LLM-facing tool server that parses YANG modules and exposes schema query tools over MCP (stdio) and REST (FastAPI).

## Prerequisites

libyang 2.x+ C library must be installed. If building from source (no sudo):

```bash
# Build libyang into ~/.local
git clone https://github.com/CESNET/libyang.git && cd libyang
cmake -DCMAKE_INSTALL_PREFIX=$HOME/.local -B build && cmake --build build --install

# Required at runtime — add to ~/.bashrc or prefix all commands
export LD_LIBRARY_PATH=$HOME/.local/lib
```

## Setup

```bash
cd yang_param
pip install -e ".[dev]"
```

## Running Tests

```bash
cd yang_param
LD_LIBRARY_PATH=$HOME/.local/lib pytest tests/ -v
```

Expected: **56 tests passing**.

Test files:
| File | Covers |
|---|---|
| `tests/test_parser.py` | YANG parser + node normalizer |
| `tests/test_store.py` | SQLite store + in-memory indexes |
| `tests/test_explore.py` | `list_modules`, `search_nodes`, `find_leaf` |
| `tests/test_tree.py` | `get_node`, `get_children`, `get_ancestors` |
| `tests/test_keys.py` | `get_path_to_leaf`, `get_required_keys`, `resolve_instance_path` |
| `tests/test_types.py` | `get_type_info`, `validate_value`, `resolve_identityref` |
| `tests/test_builder.py` | `build_edit_config`, `build_get_config`, `build_delete_config` |
| `tests/test_mcp_server.py` | MCP server tool registration |
| `tests/test_rest_server.py` | FastAPI REST endpoints |

## Running Servers

### Index YANG modules first

```bash
cd yang_param
LD_LIBRARY_PATH=$HOME/.local/lib python -c "
import tools
tools.init_store('data/yang', 'schema.db')
print('Indexed', tools.get_store().count(), 'nodes')
"
```

### MCP server (stdio — for LLM use)

```bash
cd yang_param
YANG_DIR=data/yang YANG_DB=schema.db \
  LD_LIBRARY_PATH=$HOME/.local/lib python server/mcp_server.py
```

Configure in Claude Desktop / any MCP client:
```json
{
  "mcpServers": {
    "yang-schema-tool": {
      "command": "python",
      "args": ["/path/to/yang_param/server/mcp_server.py"],
      "env": {
        "YANG_DIR": "/path/to/yang_param/data/yang",
        "YANG_DB": "/path/to/yang_param/schema.db",
        "LD_LIBRARY_PATH": "/home/user/.local/lib"
      }
    }
  }
}
```

### REST server (HTTP — for manual exploration)

```bash
cd yang_param
YANG_DIR=data/yang YANG_DB=schema.db \
  LD_LIBRARY_PATH=$HOME/.local/lib uvicorn server.rest_server:app --host 0.0.0.0 --port 8000
```

Example queries:
```bash
curl http://localhost:8000/tools/list_modules
curl "http://localhost:8000/tools/search_nodes?keyword=mtu"
curl "http://localhost:8000/tools/find_leaf?name=mtu"
curl "http://localhost:8000/tools/get_required_keys?node_id=<node_id>"

# Build edit-config XML
curl -X POST http://localhost:8000/tools/build_edit_config \
  -H "Content-Type: application/json" \
  -d '{"target_node_id": "<node_id>", "key_values": {"name": "eth0"}, "value": "1500"}'
```

Swagger UI: http://localhost:8000/docs

## Project Structure

```
yang_param/
├── indexer/
│   ├── parser.py       # libyang → raw node dicts
│   ├── normalizer.py   # raw dicts → NodeRecord (stable node_id)
│   └── store.py        # SchemaStore: SQLite + in-memory indexes
├── tools/
│   ├── explore.py      # list_modules, search_nodes, find_leaf
│   ├── tree.py         # get_node, get_children, get_ancestors
│   ├── keys.py         # get_path_to_leaf, get_required_keys, resolve_instance_path
│   ├── types.py        # get_type_info, validate_value, resolve_identityref
│   └── builder.py      # build_edit_config, build_get_config, build_delete_config
├── server/
│   ├── mcp_server.py   # MCP stdio server (16 tools)
│   └── rest_server.py  # FastAPI REST server (16 endpoints)
├── data/yang/          # YANG module files
└── tests/              # 56 tests
```

## Adding YANG Modules

Drop `.yang` files into `data/yang/` then re-index:

```bash
LD_LIBRARY_PATH=$HOME/.local/lib python -c "
import tools
tools.init_store('data/yang', 'schema.db')
"
```
