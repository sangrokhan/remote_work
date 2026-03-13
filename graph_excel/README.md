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

### `read_pdf.py`

Reads a PDF file with PyMuPDF and writes each extracted text line/paragraph as one JSON object per line (JSONL).
Each output line is minimal by default and includes only:

- `page`
- `text`
- `x`
- `y`
- `font_size`
- `rotation` (effective text direction, including tilt where available)
`font_size` is rounded to two decimal places in output (0.01 precision), matching position fields.

Using legacy page JSONL (`--legacy-page-jsonl`) you can still get page records with `header`, `footer`, and `watermark` fields.
That legacy page record includes:

- `regions`: per-line metadata for `header`, `body`, `footer`, and `watermark` lines
- `region_summary`: baseline/position/style stats for each region
- `anomalies`: page-level exception hints (including unusual header/footer baseline and mixed rotation cases)
- `tables`: optional table detections from `--find-tables`, each with `page`, `table_no`, `bbox`, `row_count`, `col_count`, `text`, `start_page`, `end_page`, and `pages` when a table spans multiple pages
- each table object also includes `rows_text` (array of row strings) for easier row-by-row consumption.
- `find-tables` builds table geometry first (`rows`/`cols` from table structure), then re-extracts text per cell rectangle so table rows are filled with layout-aligned text instead of relying only on detector text output.

```bash
python read_pdf.py <path-to-file.pdf> [--output output.jsonl] [--watermark-patterns "CONFIDENTIAL" "COPY"]
python read_pdf.py <path-to-file.pdf> --no-strip-watermarks
python read_pdf.py <path-to-file.pdf> --header-ratio 0.06 --footer-ratio 0.06
python read_pdf.py <path-to-file.pdf> --max-pages 100
python read_pdf.py <path-to-file.pdf> --pages 2-5,10,20-30
python read_pdf.py <path-to-file.pdf> --preserve-newlines
python read_pdf.py <path-to-file.pdf> --find-tables
python read_pdf.py <path-to-file.pdf> --find-tables --tables-markdown
python read_pdf.py <path-to-file.pdf> --raw rawdict
python read_pdf.py <path-to-file.pdf> --pages 2,3,4 --reconstruction --output /tmp/reconstructed.pdf --remove-watermark
python read_pdf.py <path-to-file.pdf> --legacy-page-jsonl
```
When `--find-tables` is used in default mode, table records are emitted with `"type":"table"` in JSONL.
`--tables-markdown` exports each detected table to markdown blocks with sequential names (`Table 1`, `Table 2`, ...).
When watermark removal is enabled (`--strip-watermarks`, default on), rows that match repeated watermark signatures are removed from table rows during post-processing.
By default, rotated watermark-like lines with ~55° angle are also treated as watermarks and excluded before table inference.
Cross-page tables are merged automatically when the continuation table appears on the next page, shares column count and overlapping geometry, and sits at page boundaries; duplicated header rows are removed during merge.
If `--pages` is set, it overrides `--max-pages`.
`--preserve-newlines` keeps original whitespace/newline characters in each extracted line text instead of collapsing them.

Legacy `regions` entries include more metadata if needed, including style and baseline information.

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

Find a subgraph from the matched node and stop when there is no further predicate from the current node:

```cypher
MATCH (start:JsonlEntity)
WHERE toLower(start.name) CONTAINS toLower($name)

MATCH p = (start)-[*1..]->(leaf)
WHERE NOT (leaf)-[]->()

RETURN start, p, leaf
ORDER BY length(p) ASC;
```

If your start node can branch to both directions and you want the full connected subgraph around it, use:

```cypher
MATCH (start:JsonlEntity)
WHERE toLower(start.name) CONTAINS toLower($name)

MATCH p = (start)-[*0..10]-(node)
UNWIND nodes(p) AS n
UNWIND relationships(p) AS r
RETURN
  start AS startNode,
  collect(DISTINCT n) AS subgraphNodes,
  collect(DISTINCT r) AS subgraphRels;
```

### Query parameters

In Neo4j Browser (or any Cypher interface with `:param` support):

```cypher
:param name => "counter";
:param names => ["counter", "temperature"];
```

### Match multiple names in the initial search

Use a list parameter to match several names at once:

```cypher
MATCH (n:JsonlEntity)
WHERE n.name IS NOT NULL
  AND toLower(n.name) IN [name IN $names | toLower(name)]
RETURN n.node_key AS nodeKey, n.name AS name, labels(n) AS labels
ORDER BY name
LIMIT 100;
```

You can combine this with subgraph traversal:

```cypher
MATCH (start:JsonlEntity)
WHERE toLower(start.name) IS NOT NULL
  AND toLower(start.name) IN [name IN $names | toLower(name)]

MATCH p = (start)-[*0..10]-(node)
UNWIND nodes(p) AS n
UNWIND relationships(p) AS r
RETURN
  start AS startNode,
  collect(DISTINCT n) AS subgraphNodes,
  collect(DISTINCT r) AS subgraphRels;
```
