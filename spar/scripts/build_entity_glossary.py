#!/usr/bin/env python3
"""Pass A: Excel Reference 파일 스캔 → dictionary/samsung_entities.json 생성.

Usage:
    python scripts/build_entity_glossary.py \\
        --param-dir data/parameter_refs/ \\
        --counter-dir data/counter_refs/ \\
        --alarm-dir data/alarm_refs/ \\
        --output dictionary/samsung_entities.json

    # 단일 파일 지정 (테스트/샘플용):
    python scripts/build_entity_glossary.py \\
        --param-file data/samples/parameter_ref_sample.xlsx \\
        --output dictionary/samsung_entities.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from spar.parsers.alarm_ref_parser import parse_alarm_ref_excel
from spar.parsers.counter_ref_parser import parse_counter_ref_excel
from spar.parsers.parameter_ref_parser import parse_parameter_ref_excel

_NOISE = re.compile(r"^\d+$|^.{1,2}$")


def _clean(values: list[str]) -> list[str]:
    return sorted({v.strip() for v in values if v and not _NOISE.match(v.strip())})


def scan_parameter_refs(paths: list[Path]) -> dict[str, list[str]]:
    param_names, yang_paths, feature_names = [], [], []
    for p in paths:
        result = parse_parameter_ref_excel(p)
        for r in result.records:
            param_names.append(r.param_name)
            if r.yang_path:
                yang_paths.append(r.yang_path)
            if r.feature_name:
                feature_names.append(r.feature_name)
    return {
        "parameter_names": _clean(param_names),
        "yang_paths": _clean(yang_paths),
        "feature_names": _clean(feature_names),
    }


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


def scan_alarm_refs(paths: list[Path]) -> dict[str, list[str]]:
    alarm_ids, alarm_names = [], []
    for p in paths:
        result = parse_alarm_ref_excel(p)
        for r in result.records:
            alarm_ids.append(r.alarm_id)
            alarm_names.append(r.alarm_name)
    return {
        "alarm_ids": _clean(alarm_ids),
        "alarm_names": _clean(alarm_names),
    }


def build_and_write(
    param_paths: list[Path],
    counter_paths: list[Path],
    alarm_paths: list[Path],
    output_path: Path,
) -> dict:
    entities: dict = {}
    entities.update(scan_parameter_refs(param_paths))
    entities.update(scan_counter_refs(counter_paths))
    entities.update(scan_alarm_refs(alarm_paths))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(entities, ensure_ascii=False, indent=2))
    total = sum(len(v) for v in entities.values())
    print(f"Wrote {total} entities to {output_path}")
    return entities


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
    parser.add_argument("--param-file", type=Path, action="append", default=[], dest="param_files")
    parser.add_argument("--counter-file", type=Path, action="append", default=[], dest="counter_files")
    parser.add_argument("--alarm-file", type=Path, action="append", default=[], dest="alarm_files")
    args = parser.parse_args()

    param_paths = args.param_files or _collect_excel(args.param_dir)
    counter_paths = args.counter_files or _collect_excel(args.counter_dir)
    alarm_paths = args.alarm_files or _collect_excel(args.alarm_dir)

    build_and_write(param_paths, counter_paths, alarm_paths, args.output)
