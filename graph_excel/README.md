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
