"""Samsung RAN 파라미터 레퍼런스 Excel 파서.

지원 컬럼 (순서 무관, 헤더명으로 자동 탐색):
    Feature Name, YANG Path, Parameter Name,
    Type, Default, Min, Max, Description
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import openpyxl

# 헤더 후보명 → 정규 필드명 매핑 (소문자 비교)
_COLUMN_ALIASES: dict[str, str] = {
    "feature name": "feature_name",
    "feature": "feature_name",
    "yang path": "yang_path",
    "yang": "yang_path",
    "parameter name": "param_name",
    "param name": "param_name",
    "parameter": "param_name",
    "name": "param_name",
    "type": "type",
    "default": "default",
    "default value": "default",
    "min": "min",
    "minimum": "min",
    "max": "max",
    "maximum": "max",
    "description": "description",
    "desc": "description",
}

REQUIRED_FIELDS = {"param_name", "yang_path"}


@dataclass
class ParameterRecord:
    param_name: str
    yang_path: str
    feature_name: str = ""
    type: str = ""
    default: str = ""
    min: str = ""
    max: str = ""
    description: str = ""

    @property
    def mo_path(self) -> list[str]:
        """YANG Path를 '/' 구분 계층 리스트로 반환."""
        return [p for p in self.yang_path.split("/") if p]

    @property
    def leaf_mo(self) -> str:
        """YANG Path의 마지막 컨테이너명 (파라미터 직상위)."""
        parts = self.mo_path
        return parts[-1] if parts else ""

    def to_chunk_text(self) -> str:
        """Milvus ingest용 텍스트 표현."""
        lines = [
            f"Parameter: {self.param_name}",
            f"YANG Path: {self.yang_path}",
        ]
        if self.feature_name:
            lines.append(f"Feature: {self.feature_name}")
        if self.type:
            lines.append(f"Type: {self.type}")
        range_parts = []
        if self.default:
            range_parts.append(f"default={self.default}")
        if self.min:
            range_parts.append(f"min={self.min}")
        if self.max:
            range_parts.append(f"max={self.max}")
        if range_parts:
            lines.append("Value: " + ", ".join(range_parts))
        if self.description:
            lines.append(f"Description: {self.description}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "param_name": self.param_name,
            "yang_path": self.yang_path,
            "feature_name": self.feature_name,
            "type": self.type,
            "default": self.default,
            "min": self.min,
            "max": self.max,
            "description": self.description,
        }


@dataclass
class ParameterRefParseResult:
    records: list[ParameterRecord] = field(default_factory=list)
    skipped_rows: int = 0
    warnings: list[str] = field(default_factory=list)


def _resolve_header_row(ws: openpyxl.worksheet.worksheet.Worksheet) -> tuple[int, dict[str, int]]:
    """헤더 행 번호와 {필드명: 열 인덱스(0-based)} 반환."""
    for row_idx, row in enumerate(ws.iter_rows(max_row=10, values_only=True), start=1):
        col_map: dict[str, int] = {}
        for col_idx, cell in enumerate(row):
            if cell is None:
                continue
            alias = str(cell).strip().lower()
            field_name = _COLUMN_ALIASES.get(alias)
            if field_name:
                col_map[field_name] = col_idx
        if REQUIRED_FIELDS.issubset(col_map):
            return row_idx, col_map
    raise ValueError(
        f"헤더 행 탐색 실패 — 필수 컬럼 없음: {REQUIRED_FIELDS}. "
        "컬럼명 확인 필요 (Parameter Name, YANG Path)"
    )


def _cell_str(row: tuple, idx: int | None) -> str:
    if idx is None or idx >= len(row):
        return ""
    val = row[idx]
    if val is None:
        return ""
    return str(val).strip()


def parse_parameter_ref_excel(
    path: str | Path,
    sheet_name: str | None = None,
) -> ParameterRefParseResult:
    """Excel 파라미터 레퍼런스 파일을 파싱해 ParameterRefParseResult 반환."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"파일 없음: {path}")

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet_name] if sheet_name else wb.active

    header_row, col_map = _resolve_header_row(ws)

    result = ParameterRefParseResult()
    for row_idx, row in enumerate(ws.iter_rows(min_row=header_row + 1, values_only=True), start=header_row + 1):
        if all(v is None for v in row):
            continue

        param_name = _cell_str(row, col_map.get("param_name"))
        yang_path = _cell_str(row, col_map.get("yang_path"))

        if not param_name or not yang_path:
            result.skipped_rows += 1
            result.warnings.append(f"행 {row_idx}: param_name 또는 yang_path 비어있음 — 스킵")
            continue

        result.records.append(ParameterRecord(
            param_name=param_name,
            yang_path=yang_path,
            feature_name=_cell_str(row, col_map.get("feature_name")),
            type=_cell_str(row, col_map.get("type")),
            default=_cell_str(row, col_map.get("default")),
            min=_cell_str(row, col_map.get("min")),
            max=_cell_str(row, col_map.get("max")),
            description=_cell_str(row, col_map.get("description")),
        ))

    wb.close()
    return result
