# Graph Excel

This project preprocesses Excel files into triples for Neo4j and ingests DRM-style Excel extracts into a graph.

## Requirements

- Python 3.10+
- `xlwings` (and a local Microsoft Excel installation)
- Dependencies in `requirements.txt`

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Notes on xlwings

The project now reads Excel files through `xlwings`, which requires Excel available on the host system.
If you run in an environment without Excel, these scripts cannot open workbooks.

## Scripts

### `read_excel.py`

Converts an `.xlsx` file into JSON-LD-like triples (`.jsonl`) based on sheet structure.

```bash
python read_excel.py <path-to-file.xlsx> [--type {auto,counters,parameters}] [--output output.jsonl]
```

### `main.py`

Reads files listed in `config.yaml`, decrypts if needed, parses each sheet into a DataFrame using `xlwings`, and ingests rows into Neo4j.

```bash
python main.py
```

Each entry in `config.yaml` may include an optional `password` field for encrypted files.

```yaml
- name: "DocumentA"
  path: "data/fileA.xlsx"
  key_column: "SharedID"
  password: "optional-password"
```

## Neo4j: matching nodes by name

If you used `insert_jsonl.py`, all entity nodes are stored as `:JsonlEntity` with a `name` field.  
Use this query to find nodes whose name contains a term (case-insensitive):

```cypher
MATCH (n:JsonlEntity)
WHERE n.name IS NOT NULL
  AND toLower(n.name) CONTAINS toLower($name)
RETURN n.node_key AS nodeKey, n.name AS name, labels(n) AS labels
ORDER BY name
LIMIT 50;
```
