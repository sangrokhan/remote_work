"""Dump extracted param_names and alarm_codes to JSON/TXT for inspection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spar.parsers.parameter_ref_parser import parse_parameter_ref_excel
from spar.parsers.alarm_ref_parser import parse_alarm_ref_excel

SAMPLE_DIR = Path(__file__).parent.parent / "data" / "samples"


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump parsed keywords to file")
    parser.add_argument("--param-excel", default=str(SAMPLE_DIR / "parameter_ref_sample.xlsx"))
    parser.add_argument("--alarm-excel", default=str(SAMPLE_DIR / "alarm_excel_ref_sample.xlsx"))
    parser.add_argument("--out", default="output/keywords.json", help="Output file (.json or .txt)")
    args = parser.parse_args()

    param_result = parse_parameter_ref_excel(args.param_excel)
    alarm_result = parse_alarm_ref_excel(args.alarm_excel)

    param_names = sorted({r.param_name for r in param_result.records if r.param_name})
    alarm_codes = sorted({r.alarm_id for r in alarm_result.records if r.alarm_id})

    print(f"param_names: {len(param_names)}, alarm_codes: {len(alarm_codes)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix == ".txt":
        with open(out, "w") as f:
            f.write("=== PARAM NAMES ===\n")
            f.write("\n".join(param_names))
            f.write("\n\n=== ALARM CODES ===\n")
            f.write("\n".join(alarm_codes))
            f.write("\n")
    else:
        with open(out, "w") as f:
            json.dump({"param_names": param_names, "alarm_codes": alarm_codes}, f, indent=2)

    print(f"Written → {out}")


if __name__ == "__main__":
    main()
