"""In-memory index of AlarmRecord for alarm_id direct lookup."""
from __future__ import annotations

import os
from pathlib import Path

from spar.parsers.alarm_ref_parser import AlarmRecord, parse_alarm_ref_excel

_DEFAULT_SAMPLE = Path("data/samples/alarm_excel_ref_sample.xlsx")
_SINGLETON: "AlarmIndex | None" = None


class AlarmIndex:
    def __init__(self, records: list[AlarmRecord]) -> None:
        self._by_id: dict[str, AlarmRecord] = {}
        for r in records:
            self._by_id[r.alarm_id.upper()] = r
        self._records: list[AlarmRecord] = list(records)

    def __len__(self) -> int:
        return len(self._records)

    def lookup(self, alarm_id: str) -> AlarmRecord | None:
        if not alarm_id:
            return None
        return self._by_id.get(alarm_id.strip().upper())

    def search_by_name(self, query: str) -> list[AlarmRecord]:
        q = query.strip().lower()
        if not q:
            return []
        return [r for r in self._records if q in r.alarm_name.lower()]


def _resolve_path(path: str | Path | None) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get("SPAR_ALARM_REF_PATH")
    if env:
        return Path(env)
    return _DEFAULT_SAMPLE


def get_alarm_index(path: str | Path | None = None) -> AlarmIndex:
    """Return process-wide AlarmIndex singleton.

    First call loads from ``path`` > ``$SPAR_ALARM_REF_PATH`` > default sample.
    Subsequent calls ignore arguments and return the cached instance.
    """
    global _SINGLETON
    if _SINGLETON is None:
        resolved = _resolve_path(path)
        result = parse_alarm_ref_excel(resolved)
        _SINGLETON = AlarmIndex(result.records)
    return _SINGLETON
