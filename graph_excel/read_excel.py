import argparse
import json
import re
from pathlib import Path

import xlwings as xw


def _normalize_cell(value):
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value).strip()
    return text if text else None


def _split_system_ids(value):
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        ids = []
        for item in value:
            ids.extend(_split_system_ids(item))
        return ids

    raw = str(value).strip()
    if not raw:
        return []

    parts = re.split(r"[;,|\n\r]+", raw)
    return [part.strip() for part in parts if part and part.strip()]


def _read_cell(sheet, column, row):
    return sheet.range(f"{column}{row}").value


def _max_used_row(sheet):
    used_range = sheet.used_range
    if used_range is None:
        return 1
    try:
        return used_range.last_cell.row
    except Exception:
        return 1


def _get_sheet_by_name(workbook, target_name):
    for sheet in workbook.sheets:
        if sheet.name.lower() == target_name.lower():
            return sheet
    return None


def _append_triple(triples, subject, predicate, object_value, metadata=None):
    triple = {
        "subject": subject,
        "predicate": predicate,
        "object": object_value,
    }
    if metadata:
        triple["meta"] = metadata
    triples.append(triple)


def process_counters_sheet(sheet, file_stem, triples):
    max_row = _max_used_row(sheet)
    for row in range(2, max_row + 1):
        category = _normalize_cell(_read_cell(sheet, "A", row))
        name = _normalize_cell(_read_cell(sheet, "B", row))
        counter_id = _normalize_cell(_read_cell(sheet, "C", row))
        system_ids = _split_system_ids(_read_cell(sheet, "X", row))

        if not name or not counter_id:
            continue

        subject = name
        metadata = {
            "subject_properties": {
                "counter_id": counter_id,
            }
        }
        if category is not None:
            _append_triple(
                triples,
                subject=subject,
                predicate="HAS_CATEGORY",
                object_value=category,
                metadata=metadata,
            )

        for system_id in system_ids:
            _append_triple(
                triples,
                subject=subject,
                predicate="HAS_SYSTEM_ID",
                object_value=system_id,
                metadata=metadata,
            )


def process_parameters_sheet(sheet, file_stem, triples):
    max_row = _max_used_row(sheet)
    for row in range(2, max_row + 1):
        parameter = _normalize_cell(_read_cell(sheet, "B", row))
        hierarchy = _normalize_cell(_read_cell(sheet, "A", row))
        range_value = _normalize_cell(_read_cell(sheet, "J", row))
        default_value = _normalize_cell(_read_cell(sheet, "K", row))
        system_ids = _split_system_ids(_read_cell(sheet, "U", row))
        if not parameter or not system_ids:
            continue
        if hierarchy is None:
            continue

        metadata = {
            "subject_properties": {
                "hierarchy": hierarchy,
                "range": range_value,
                "default_value": default_value,
            }
        }

        for system_id in system_ids:
            _append_triple(
                triples,
                subject=parameter,
                predicate="HAS_SYSTEM_ID",
                object_value=system_id,
                metadata=metadata,
            )


def preprocess_excel(path, file_type="auto"):
    app = xw.App(visible=False, add_book=False)
    app.display_alerts = False
    app.screen_updating = False

    workbook = None
    try:
        workbook = app.books.open(str(path))
    except Exception:
        app.quit()
        raise

    file_stem = Path(path).stem
    triples = []

    try:
        if file_type in ("auto", "counters"):
            counter_sheet = _get_sheet_by_name(workbook, "Counter")
            if counter_sheet is not None:
                process_counters_sheet(counter_sheet, file_stem, triples)
            elif file_type == "counters":
                raise ValueError(
                    "Counter sheet not found. Expected sheet name: 'Counter'."
                )

        if file_type in ("auto", "parameters"):
            parameter_sheet = _get_sheet_by_name(workbook, "Parameter Description")
            if parameter_sheet is not None:
                process_parameters_sheet(
                    parameter_sheet,
                    file_stem,
                    triples,
                )
            elif file_type == "parameters":
                raise ValueError(
                    "Parameter sheet not found. Expected sheet name: 'Parameter Description'."
                )

        if file_type == "auto" and not triples:
            raise ValueError(
                "No supported sheet found. Expected 'Counter' or 'Parameter Description'."
            )
        return triples
    finally:
        if workbook is not None:
            workbook.close()
        app.quit()


def write_triples(triples, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for triple in triples:
            f.write(json.dumps(triple, ensure_ascii=False))
            f.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Excel files into Neo4j triple JSON.")
    parser.add_argument("file", help="Path to the Excel file")
    parser.add_argument(
        "--type",
        choices=["auto", "counters", "parameters"],
        default="auto",
        help="Target sheet type. Auto detects based on sheet names.",
    )
    parser.add_argument(
        "--output",
        help="Output path for triples (.jsonl recommended). Defaults to <filename>_triples.jsonl",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output or f"{Path(args.file).stem}_triples.jsonl"
    triples = preprocess_excel(args.file, file_type=args.type)
    write_triples(triples, output_path)
    print(f"Generated {len(triples)} triples -> {output_path}")


if __name__ == "__main__":
    main()
