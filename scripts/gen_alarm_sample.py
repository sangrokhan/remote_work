"""Reproducible generator for alarm_excel_ref_sample.xlsx.

Run: python scripts/gen_alarm_sample.py
Outputs: data/samples/alarm_excel_ref_sample.xlsx
"""
from __future__ import annotations

from pathlib import Path

import openpyxl

ROWS: list[tuple[str, str, str, str, str]] = [
    ("ALM-1001", "Cell Out of Service", "Critical", "Radio", "gNB-DU"),
    ("ALM-1002", "Link Down", "Critical", "Transport", "gNB-CU"),
    ("ALM-1003", "Cell Down", "Critical", "Radio", "gNB-DU"),
    ("ALM-1004", "High Temperature", "Major", "HW", "RU"),
    ("ALM-1005", "Fan Failure", "Major", "HW", "BBU"),
    ("ALM-1006", "Clock Sync Loss", "Major", "Transport", "gNB-DU"),
    ("ALM-1007", "License Expiring", "Minor", "SW", "OAM"),
    ("ALM-1008", "High CPU Usage", "Minor", "SW", "gNB-CU"),
    ("ALM-1009", "PRACH Anomaly", "Minor", "Radio", "gNB-DU"),
    ("ALM-1010", "Backup Failed", "Warning", "SW", "OAM"),
    ("ALM-1011", "Config Mismatch", "Warning", "SW", "gNB-CU"),
    ("ALM-1012", "Power Redundancy Lost", "Major", "HW", "BBU"),
]

HEADERS = ("Alarm ID", "Alarm Name", "Severity", "Category", "Module")


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "data" / "samples" / "alarm_excel_ref_sample.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "alarms"
    ws.append(HEADERS)
    for row in ROWS:
        ws.append(row)

    wb.save(out)
    print(f"wrote {out} ({len(ROWS)} rows)")


if __name__ == "__main__":
    main()
