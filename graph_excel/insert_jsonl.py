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


def _make_node_key(subject_value: Any, subject_label: Any, explicit_key: Any) -> str:
    if explicit_key is not None:
        return str(explicit_key)
    if subject_label is not None and str(subject_label).strip():
        return f"{str(subject_label).strip()}:{_stringify(subject_value)}"
    return _stringify(subject_value)


def _is_nullish_object(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"none", "null", "n/a", "na"}
    return False


def _prepare_rows(jsonl_path: Path, skip_invalid: bool) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line_no, item in enumerate(_read_jsonl(jsonl_path), 1):
        subject = item.get("subject")
        predicate = item.get("predicate")
        obj = item.get("object")
        subject_label = item.get("subject_label")
        object_label = item.get("object_label")
        subject_node_key = item.get("subject_node_key", item.get("subject_key"))
        object_node_key = item.get("object_node_key", item.get("object_key"))
        meta = item.get("meta", {})
        if subject is None or predicate is None:
            if skip_invalid:
                continue
            raise ValueError(
                f"Line {line_no} missing required key(s): subject/predicate"
            )

        if _is_nullish_object(obj):
            if skip_invalid:
                continue
            raise ValueError(f"Line {line_no} skipped: object is null-like")
            continue

        # Neo4j relationships do not accept nested property values.
        # Store meta (kept under its original key) as JSON text.
        meta_value = json.dumps(meta, ensure_ascii=False, sort_keys=True) if meta is not None else None
        subject_properties = {}
        object_properties = {}
        if isinstance(meta, dict):
            subject_properties = meta.get("subject_properties", {}) or {}
            object_properties = meta.get("object_properties", {}) or {}
            if subject_label is None:
                subject_label = meta.get("subject_label")
            if object_label is None:
                object_label = meta.get("object_label")
            if not isinstance(subject_properties, dict):
                subject_properties = {}
            if not isinstance(object_properties, dict):
                object_properties = {}
            subject_properties = {str(k): v for k, v in subject_properties.items() if k is not None}
            object_properties = {str(k): v for k, v in object_properties.items() if k is not None}
            if subject_node_key is None:
                subject_node_key = meta.get("subject_node_key")
            if object_node_key is None:
                object_node_key = meta.get("object_node_key")

        subject_node_key = _make_node_key(subject, subject_label, subject_node_key)
        object_node_key = _make_node_key(obj, object_label, object_node_key)

        records.append(
            {
                "subject_value": _stringify(subject),
                "object_value": _stringify(obj),
                "predicate": str(predicate).strip(),
                "meta": meta_value,
                "subject_properties": subject_properties,
                "object_properties": object_properties,
                "subject_label": subject_label,
                "object_label": object_label,
                "subject_node_key": subject_node_key,
                "object_node_key": object_node_key,
                "source": str(jsonl_path.name),
                "json_line": line_no,
            }
        )

    return records


def _create_indexes_and_constraints(session, dry_run: bool) -> None:
    if dry_run:
        return
    session.run("DROP CONSTRAINT jsonl_node_name_unique IF EXISTS")
    session.run(
        """
        CREATE CONSTRAINT jsonl_node_key_unique
        IF NOT EXISTS
        FOR (n:JsonlEntity)
        REQUIRE n.node_key IS UNIQUE
        """
    )


def _insert_batch(tx, rows: List[Dict[str, Any]], ignore_meta: bool = False) -> int:
    if not rows:
        return 0
    total = 0
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        rel_type = row["predicate"] or "HAS_TRIPLE"
        row["predicate"] = rel_type
        grouped.setdefault(rel_type, []).append(row)

    for rel_type, grouped_rows in grouped.items():
        rel_rows: Dict[tuple, List[Dict[str, Any]]] = {}
        for row in grouped_rows:
            key = (
                str(row.get("subject_label") or ""),
                str(row.get("object_label") or ""),
            )
            rel_rows.setdefault(key, []).append(row)

        for (subject_label, object_label), rel_group_rows in rel_rows.items():
            subject_suffix = f":{subject_label}" if subject_label else ""
            object_suffix = f":{object_label}" if object_label else ""
            if ignore_meta:
                query = f"""
                UNWIND $rows AS row
                MERGE (s:JsonlEntity{subject_suffix} {{node_key: row.subject_node_key, name: row.subject_value}})
                SET s += coalesce(row.subject_properties, {{}})
                MERGE (o:JsonlEntity{object_suffix} {{node_key: row.object_node_key, name: row.object_value}})
                SET o += coalesce(row.object_properties, {{}})
                MERGE (s)-[r:`{rel_type}`]->(o)
                SET r.predicate = row.predicate,
                    r.source_file = row.source,
                    r.source_line = row.json_line
                """
            else:
                query = f"""
                UNWIND $rows AS row
                MERGE (s:JsonlEntity{subject_suffix} {{node_key: row.subject_node_key, name: row.subject_value}})
                SET s += coalesce(row.subject_properties, {{}})
                MERGE (o:JsonlEntity{object_suffix} {{node_key: row.object_node_key, name: row.object_value}})
                SET o += coalesce(row.object_properties, {{}})
                MERGE (s)-[r:`{rel_type}`]->(o)
                SET r.predicate = row.predicate,
                    r.meta = row.meta,
                    r.source_file = row.source,
                    r.source_line = row.json_line
                """
            result = tx.run(query + "\nRETURN count(*) AS c", rows=rel_group_rows)
            total += result.single()["c"]
    return total


def insert_jsonl(
    uri: str,
    user: str,
    password: str,
    jsonl_path: str,
    batch_size: int = 500,
    database: str = None,
    skip_invalid: bool = True,
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
        "--no-skip-invalid",
        dest="skip_invalid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip invalid lines (default: True).",
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
