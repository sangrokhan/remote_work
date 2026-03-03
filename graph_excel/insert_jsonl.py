import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from neo4j import GraphDatabase


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return json.dumps(value, ensure_ascii=False)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc.msg}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Invalid JSON object on line {line_no}: expected dict")
            yield item


def _prepare_rows(jsonl_path: Path, skip_invalid: bool) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line_no, item in enumerate(_read_jsonl(jsonl_path), 1):
        subject = item.get("subject")
        predicate = item.get("predicate")
        obj = item.get("object")
        meta = item.get("meta", {})
        if subject is None or predicate is None or obj is None:
            if skip_invalid:
                continue
            raise ValueError(
                f"Line {line_no} missing required key(s): subject/predicate/object"
            )

        # Neo4j relationships do not accept nested property values.
        # Store meta (kept under its original key) as JSON text.
        meta_value = json.dumps(meta, ensure_ascii=False, sort_keys=True) if meta is not None else None

        records.append(
            {
                "subject_value": _stringify(subject),
                "subject_type": type(subject).__name__,
                "object_value": _stringify(obj),
                "object_type": type(obj).__name__,
                "predicate": str(predicate),
                "meta": meta_value,
                "source": str(jsonl_path.name),
                "json_line": line_no,
            }
        )

    return records


def _create_indexes_and_constraints(session, dry_run: bool) -> None:
    if dry_run:
        return
    session.run(
        """
        CREATE CONSTRAINT jsonl_node_value_unique
        IF NOT EXISTS
        FOR (n:JsonlEntity)
        REQUIRE n.value IS UNIQUE
        """
    )


def _insert_batch(tx, rows: List[Dict[str, Any]], ignore_meta: bool = False) -> int:
    if not rows:
        return 0

    if ignore_meta:
        query = """
    UNWIND $rows AS row
    MERGE (s:JsonlEntity {value: row.subject_value})
    SET s.type = row.subject_type,
        s.last_seen_source = row.source
    MERGE (o:JsonlEntity {value: row.object_value})
    SET o.type = row.object_type,
        o.last_seen_source = row.source
    MERGE (s)-[r:HAS_TRIPLE]->(o)
    SET r.predicate = row.predicate,
        r.source_file = row.source,
        r.source_line = row.json_line
    RETURN count(*) AS c
    """
    else:
        query = """
    UNWIND $rows AS row
    MERGE (s:JsonlEntity {value: row.subject_value})
    SET s.type = row.subject_type,
        s.last_seen_source = row.source
    MERGE (o:JsonlEntity {value: row.object_value})
    SET o.type = row.object_type,
        o.last_seen_source = row.source
    MERGE (s)-[r:HAS_TRIPLE]->(o)
    SET r.predicate = row.predicate,
        r.meta = row.meta,
        r.source_file = row.source,
        r.source_line = row.json_line
    RETURN count(*) AS c
    """
    result = tx.run(query, rows=rows)
    return result.single()["c"]


def insert_jsonl(
    uri: str,
    user: str,
    password: str,
    jsonl_path: str,
    batch_size: int = 500,
    database: str = None,
    skip_invalid: bool = False,
    dry_run: bool = False,
    ignore_meta: bool = False,
) -> None:
    path = Path(jsonl_path)
    records = _prepare_rows(path, skip_invalid=skip_invalid)
    if not records:
        print("No valid rows found.")
        return

    print(f"Loaded {len(records)} triples from {path}")

    if dry_run:
        print("Dry-run enabled. No Neo4j writes performed.")
        return

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as session:
            _create_indexes_and_constraints(session, dry_run=False)

            total = 0
            for record in records:
                if ignore_meta:
                    record["meta"] = None

            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                count = session.execute_write(_insert_batch, batch, ignore_meta)
                total += count
            print(f"Inserted/updated {total} triples.")
    finally:
        driver.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Insert preprocessed triple JSONL files into Neo4j."
    )
    parser.add_argument("jsonl_path", help="Path to .jsonl file")
    parser.add_argument(
        "--uri",
        default="bolt://localhost:7687",
        help="Neo4j Bolt URI (default: bolt://localhost:7687)",
    )
    parser.add_argument(
        "--user",
        default="neo4j",
        help="Neo4j username (default: neo4j)",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("NEO4J_PASSWORD", "password"),
        help="Neo4j password (default: env NEO4J_PASSWORD or 'password')",
    )
    parser.add_argument(
        "--database",
        default=None,
        help="Neo4j database name (optional; default is database configured on server)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of rows to insert per transaction (default: 500)",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip lines missing required fields instead of failing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate JSONL and print count without writing to Neo4j.",
    )
    parser.add_argument(
        "--ignore-meta",
        action="store_true",
        help="Do not write meta metadata to Neo4j (ignore meta field).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    insert_jsonl(
        uri=args.uri,
        user=args.user,
        password=args.password,
        jsonl_path=args.jsonl_path,
        batch_size=args.batch_size,
        database=args.database,
        skip_invalid=args.skip_invalid,
        dry_run=args.dry_run,
        ignore_meta=args.ignore_meta,
    )


if __name__ == "__main__":
    main()
