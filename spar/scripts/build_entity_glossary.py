#!/usr/bin/env python3
"""Pass A: Excel Reference + Feature DOCX 스캔 → dictionary/samsung_entities.json 생성.

Usage:
    python scripts/build_entity_glossary.py \\
        --param-dir data/parameter_refs/ \\
        --counter-dir data/counter_refs/ \\
        --alarm-dir data/alarm_refs/ \\
        --feature-dir data/feature_descs/ \\
        --output dictionary/samsung_entities.json

    # 단일 파일 지정 (테스트/샘플용):
    python scripts/build_entity_glossary.py \\
        --param-file data/samples/parameter_ref_sample.xlsx \\
        --feature-file data/samples/feature_desc_sample.docx \\
        --output dictionary/samsung_entities.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from spar.parsers.alarm_ref_parser import parse_alarm_ref_excel
from spar.parsers.counter_ref_parser import parse_counter_ref_excel
from spar.parsers.docx_config import DocxParseConfig
from spar.parsers.docx_parser import DocxParser
from spar.parsers.parameter_ref_parser import parse_parameter_ref_excel

_FEATURE_ID_RE = re.compile(r"Feature ID:\s*(FGR-\w+)")

_NOISE = re.compile(r"^\d+$|^.{1,2}$")


def _clean(values: list[str]) -> list[str]:
    return sorted({v.strip() for v in values if v and not _NOISE.match(v.strip())})


def scan_parameter_refs(paths: list[Path]) -> dict[str, list[str]]:
    param_names = []
    for p in paths:
        result = parse_parameter_ref_excel(p)
        for r in result.records:
            param_names.append(r.param_name)
    return {"parameter_names": _clean(param_names)}


def scan_counter_refs(paths: list[Path]) -> dict[str, list[str]]:
    counter_names, counter_groups = [], []
    for p in paths:
        result = parse_counter_ref_excel(p)
        for r in result.records:
            counter_names.append(r.counter_name)
            if r.mid_group:
                counter_groups.append(r.mid_group)
            if r.large_group:
                counter_groups.append(r.large_group)
    return {
        "counter_names": _clean(counter_names),
        "counter_groups": _clean(counter_groups),
    }


def scan_feature_docs(paths: list[Path]) -> dict:
    _parser = DocxParser(DocxParseConfig())
    seen: dict[str, str] = {}
    for p in paths:
        result = _parser.parse(p)
        current_name: str | None = None
        for line in result.markdown.splitlines():
            if line.startswith("# "):
                current_name = line[2:].strip()
            m = _FEATURE_ID_RE.search(line)
            if m and current_name:
                seen[m.group(1)] = current_name
                current_name = None
    features = [{"id": fid, "name": name} for fid, name in sorted(seen.items())]
    return {"features": features}


def scan_alarm_refs(paths: list[Path]) -> dict:
    seen: dict[str, str] = {}
    for p in paths:
        result = parse_alarm_ref_excel(p)
        for r in result.records:
            aid = r.alarm_id.strip()
            aname = r.alarm_name.strip()
            if aid and not _NOISE.match(aid):
                seen[aid] = aname
    alarms = [{"id": aid, "name": name} for aid, name in sorted(seen.items())]
    return {"alarms": alarms}


def _merge_id_name_list(existing: list[dict], new: list[dict]) -> list[dict]:
    by_id = {e["id"]: e["name"] for e in existing}
    by_id.update({e["id"]: e["name"] for e in new})
    return [{"id": k, "name": v} for k, v in sorted(by_id.items())]


def build_and_write(
    param_paths: list[Path],
    counter_paths: list[Path],
    alarm_paths: list[Path],
    output_path: Path,
    feature_paths: list[Path] | None = None,
) -> dict:
    _DEPRECATED_KEYS = {"yang_paths", "feature_names", "alarm_ids", "alarm_names", "feature_ids"}
    _ID_NAME_KEYS = {"alarms", "features"}

    existing: dict = {}
    if output_path.exists():
        raw = json.loads(output_path.read_text())
        existing = {k: v for k, v in raw.items() if k not in _DEPRECATED_KEYS}

    new_entities: dict = {}
    new_entities.update(scan_parameter_refs(param_paths))
    new_entities.update(scan_counter_refs(counter_paths))
    new_entities.update(scan_alarm_refs(alarm_paths))
    if feature_paths:
        new_entities.update(scan_feature_docs(feature_paths))

    merged: dict = {}
    all_keys = existing.keys() | new_entities.keys()
    for key in all_keys:
        if key in _ID_NAME_KEYS:
            merged[key] = _merge_id_name_list(existing.get(key, []), new_entities.get(key, []))
        else:
            merged[key] = sorted(set(existing.get(key, [])) | set(new_entities.get(key, [])))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
    total = sum(len(v) for v in merged.values())
    print(f"Wrote {total} entities to {output_path}")
    return merged


def _collect_docx(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob("*.docx"))


def _collect_excel(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob("*.xlsx")) + sorted(directory.glob("*.xls"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--param-dir", type=Path, default=Path("data/parameter_refs"))
    parser.add_argument("--counter-dir", type=Path, default=Path("data/counter_refs"))
    parser.add_argument("--alarm-dir", type=Path, default=Path("data/alarm_refs"))
    parser.add_argument("--output", type=Path, default=Path("dictionary/samsung_entities.json"))
    parser.add_argument("--feature-dir", type=Path, default=Path("data/feature_descs"))
    parser.add_argument("--param-file", type=Path, action="append", default=[], dest="param_files")
    parser.add_argument("--counter-file", type=Path, action="append", default=[], dest="counter_files")
    parser.add_argument("--alarm-file", type=Path, action="append", default=[], dest="alarm_files")
    parser.add_argument("--feature-file", type=Path, action="append", default=[], dest="feature_files")
    args = parser.parse_args()

    param_paths = args.param_files or _collect_excel(args.param_dir)
    counter_paths = args.counter_files or _collect_excel(args.counter_dir)
    alarm_paths = args.alarm_files or _collect_excel(args.alarm_dir)
    feature_paths = args.feature_files or _collect_docx(args.feature_dir)

    build_and_write(param_paths, counter_paths, alarm_paths, args.output, feature_paths=feature_paths)
