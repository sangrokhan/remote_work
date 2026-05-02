# Word 문서 파서 (DocxParser) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `.docx` 파일을 파싱하여 Markdown(본문) + 개별 CSV(표) + 개별 이미지 파일로 추출하되, 각 파일에 소속 섹션 heading 컨텍스트를 기록한다.

**Architecture:** `python-docx`로 XML body child 순서대로 paragraph/table 순회. Heading style name(`"Heading 1"` ~ `"Heading N"`) 기반으로 섹션 컨텍스트 갱신. 표→CSV, 이미지→바이너리 저장 후 Markdown에 `<!-- TABLE/IMAGE: ... -->` placeholder 삽입.

**Tech Stack:** `python-docx>=1.0`, `pytest`, `dataclasses` (stdlib)

---

## 파일 구조

| 파일 | 역할 |
|------|------|
| `src/spar/parsers/__init__.py` | 생성 (빈 파일, `.gitkeep` 교체) |
| `src/spar/parsers/docx_config.py` | `DocxParseConfig` dataclass |
| `src/spar/parsers/docx_parser.py` | `DocxParser`, `ParseResult`, `ExtractedTable`, `ExtractedImage` |
| `scripts/parse_docx.py` | CLI 진입점 |
| `tests/parsers/__init__.py` | 생성 |
| `tests/parsers/test_docx_parser.py` | 전체 테스트 |
| `pyproject.toml` | `python-docx>=1.0` 의존성 추가 |

---

## Task 1: 의존성 추가 + 파일 scaffold

**Files:**
- Modify: `pyproject.toml`
- Create: `src/spar/parsers/__init__.py`
- Create: `tests/parsers/__init__.py`

- [ ] **Step 1: `python-docx` 의존성 추가**

`pyproject.toml`의 `dependencies` 리스트에 추가:
```toml
dependencies = [
    "langgraph>=0.2",
    "langchain-core>=0.3",
    "openpyxl>=3.1",
    "python-docx>=1.0",
]
```

- [ ] **Step 2: 패키지 설치 확인**

```bash
pip install python-docx
python -c "import docx; print(docx.__version__)"
```
Expected: 버전 출력 (에러 없음)

- [ ] **Step 3: `.gitkeep` 제거 + `__init__.py` 생성**

```bash
rm src/spar/parsers/.gitkeep
touch src/spar/parsers/__init__.py
mkdir -p tests/parsers
touch tests/parsers/__init__.py
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml src/spar/parsers/__init__.py tests/parsers/__init__.py
git commit -m "chore: add python-docx dep and scaffold parsers package"
```

---

## Task 2: DocxParseConfig

**Files:**
- Create: `src/spar/parsers/docx_config.py`
- Test: `tests/parsers/test_docx_parser.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/parsers/test_docx_parser.py` 생성:
```python
from __future__ import annotations

from pathlib import Path

import pytest
import docx as _docx

from spar.parsers.docx_config import DocxParseConfig


class TestDocxParseConfig:
    def test_defaults(self) -> None:
        cfg = DocxParseConfig()
        assert cfg.heading_depth == 2
        assert cfg.output_dir == Path("output")
        assert cfg.slugify_max_len == 30

    def test_custom_values(self) -> None:
        cfg = DocxParseConfig(heading_depth=3, output_dir=Path("/tmp/out"), slugify_max_len=20)
        assert cfg.heading_depth == 3
        assert cfg.output_dir == Path("/tmp/out")
        assert cfg.slugify_max_len == 20
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
pytest tests/parsers/test_docx_parser.py::TestDocxParseConfig -v
```
Expected: `ModuleNotFoundError: No module named 'spar.parsers.docx_config'`

- [ ] **Step 3: DocxParseConfig 구현**

`src/spar/parsers/docx_config.py` 생성:
```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DocxParseConfig:
    heading_depth: int = 2
    output_dir: Path = field(default_factory=lambda: Path("output"))
    slugify_max_len: int = 30
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/parsers/test_docx_parser.py::TestDocxParseConfig -v
```
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/spar/parsers/docx_config.py tests/parsers/test_docx_parser.py
git commit -m "feat(parsers): add DocxParseConfig dataclass"
```

---

## Task 3: slugify 유틸리티

**Files:**
- Create: `src/spar/parsers/docx_parser.py` (초기)
- Test: `tests/parsers/test_docx_parser.py` (추가)

- [ ] **Step 1: 실패하는 테스트 추가**

`tests/parsers/test_docx_parser.py` 끝에 추가:
```python
from spar.parsers.docx_parser import _slugify


class TestSlugify:
    def test_spaces_to_dash(self) -> None:
        assert _slugify("System Overview", 30) == "System-Overview"

    def test_truncates(self) -> None:
        result = _slugify("A Very Long Section Title Here", 15)
        assert len(result) <= 15

    def test_strips_special_chars(self) -> None:
        assert _slugify("Section (v2.0)!", 30) == "Section-v2.0"

    def test_empty_string(self) -> None:
        assert _slugify("", 30) == "unnamed"
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
pytest tests/parsers/test_docx_parser.py::TestSlugify -v
```
Expected: `ImportError`

- [ ] **Step 3: `_slugify` 구현**

`src/spar/parsers/docx_parser.py` 생성:
```python
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def _slugify(text: str, max_len: int) -> str:
    if not text.strip():
        return "unnamed"
    slug = re.sub(r"[^\w\s-]", "", text)
    slug = re.sub(r"[\s_]+", "-", slug).strip("-")
    return slug[:max_len] if slug else "unnamed"
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/parsers/test_docx_parser.py::TestSlugify -v
```
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/spar/parsers/docx_parser.py tests/parsers/test_docx_parser.py
git commit -m "feat(parsers): add _slugify utility"
```

---

## Task 4: ParseResult / ExtractedTable / ExtractedImage 데이터 클래스

**Files:**
- Modify: `src/spar/parsers/docx_parser.py`
- Test: `tests/parsers/test_docx_parser.py` (추가)

- [ ] **Step 1: 실패하는 테스트 추가**

```python
from spar.parsers.docx_parser import ParseResult, ExtractedTable, ExtractedImage


class TestParseResult:
    def test_parse_result_fields(self, tmp_path: Path) -> None:
        table = ExtractedTable(
            path=tmp_path / "Table_Intro_1.csv",
            section_path=["Introduction"],
            seq=1,
        )
        image = ExtractedImage(
            path=tmp_path / "Fig_Intro_1.png",
            section_path=["Introduction"],
            seq=1,
            ext="png",
        )
        result = ParseResult(markdown="# Hello", tables=[table], images=[image])
        assert result.markdown == "# Hello"
        assert len(result.tables) == 1
        assert len(result.images) == 1
        assert result.tables[0].seq == 1
        assert result.images[0].ext == "png"
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
pytest tests/parsers/test_docx_parser.py::TestParseResult -v
```
Expected: `ImportError`

- [ ] **Step 3: 데이터 클래스 추가**

`src/spar/parsers/docx_parser.py`에 추가 (기존 `_slugify` 아래):
```python
@dataclass
class ExtractedTable:
    path: Path
    section_path: list[str]
    seq: int


@dataclass
class ExtractedImage:
    path: Path
    section_path: list[str]
    seq: int
    ext: str


@dataclass
class ParseResult:
    markdown: str
    tables: list[ExtractedTable] = field(default_factory=list)
    images: list[ExtractedImage] = field(default_factory=list)
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/parsers/test_docx_parser.py::TestParseResult -v
```
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/spar/parsers/docx_parser.py tests/parsers/test_docx_parser.py
git commit -m "feat(parsers): add ParseResult, ExtractedTable, ExtractedImage dataclasses"
```

---

## Task 5: DocxParser — heading 파싱 + Markdown 본문 출력

**Files:**
- Modify: `src/spar/parsers/docx_parser.py`
- Test: `tests/parsers/test_docx_parser.py` (추가)

테스트 fixture는 `python-docx`로 프로그래매틱하게 `.docx` 생성.

- [ ] **Step 1: 실패하는 테스트 추가**

```python
import docx as _docx
from spar.parsers.docx_config import DocxParseConfig
from spar.parsers.docx_parser import DocxParser


def _make_docx(tmp_path: Path, setup_fn) -> Path:
    doc = _docx.Document()
    setup_fn(doc)
    path = tmp_path / "test.docx"
    doc.save(str(path))
    return path


class TestDocxParserHeadings:
    def test_heading1_becomes_h1(self, tmp_path: Path) -> None:
        def setup(doc):
            doc.add_heading("Introduction", level=1)
            doc.add_paragraph("Some text here.")

        path = _make_docx(tmp_path, setup)
        cfg = DocxParseConfig(output_dir=tmp_path)
        parser = DocxParser(cfg)
        result = parser.parse(path)
        assert "# Introduction" in result.markdown
        assert "Some text here." in result.markdown

    def test_heading2_within_depth_becomes_h2(self, tmp_path: Path) -> None:
        def setup(doc):
            doc.add_heading("Chapter", level=1)
            doc.add_heading("Sub Section", level=2)
            doc.add_paragraph("Body text.")

        path = _make_docx(tmp_path, setup)
        cfg = DocxParseConfig(heading_depth=2, output_dir=tmp_path)
        parser = DocxParser(cfg)
        result = parser.parse(path)
        assert "# Chapter" in result.markdown
        assert "## Sub Section" in result.markdown

    def test_heading_beyond_depth_still_rendered(self, tmp_path: Path) -> None:
        def setup(doc):
            doc.add_heading("Chapter", level=1)
            doc.add_heading("Deep", level=3)

        path = _make_docx(tmp_path, setup)
        cfg = DocxParseConfig(heading_depth=2, output_dir=tmp_path)
        parser = DocxParser(cfg)
        result = parser.parse(path)
        assert "### Deep" in result.markdown
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
pytest tests/parsers/test_docx_parser.py::TestDocxParserHeadings -v
```
Expected: `ImportError` or `AttributeError`

- [ ] **Step 3: DocxParser 핵심 구조 구현**

`src/spar/parsers/docx_parser.py`에 추가:
```python
import docx
from spar.parsers.docx_config import DocxParseConfig


class DocxParser:
    def __init__(self, config: DocxParseConfig) -> None:
        self._cfg = config

    def parse(self, docx_path: Path) -> ParseResult:
        doc = docx.Document(str(docx_path))
        out_dir = self._cfg.output_dir
        tables_dir = out_dir / "tables"
        images_dir = out_dir / "images"
        tables_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        section_path: list[str] = []
        table_seq: dict[str, int] = {}
        image_seq: dict[str, int] = {}
        lines: list[str] = []
        extracted_tables: list[ExtractedTable] = []
        extracted_images: list[ExtractedImage] = []

        for child in doc.element.body:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag

            if tag == "p":
                para = docx.text.paragraph.Paragraph(child, doc)
                self._handle_paragraph(
                    para, section_path, table_seq, image_seq,
                    lines, extracted_images, images_dir, doc,
                )
            elif tag == "tbl":
                table = docx.table.Table(child, doc)
                self._handle_table(
                    table, section_path, table_seq,
                    lines, extracted_tables, tables_dir, docx_path,
                )

        return ParseResult(
            markdown="\n".join(lines),
            tables=extracted_tables,
            images=extracted_images,
        )

    def _section_key(self, section_path: list[str]) -> str:
        return " > ".join(section_path) if section_path else "root"

    def _handle_paragraph(
        self,
        para,
        section_path: list[str],
        table_seq: dict[str, int],
        image_seq: dict[str, int],
        lines: list[str],
        extracted_images: list[ExtractedImage],
        images_dir: Path,
        doc,
    ) -> None:
        style_name: str = para.style.name if para.style else ""

        if style_name.startswith("Heading"):
            try:
                level = int(style_name.split()[-1])
            except (ValueError, IndexError):
                level = 1
            title = para.text.strip()
            if level <= self._cfg.heading_depth:
                # 섹션 컨텍스트 갱신
                if level == 1:
                    section_path.clear()
                    table_seq.clear()
                    image_seq.clear()
                elif len(section_path) >= level:
                    del section_path[level - 1:]
                    # seq는 섹션 변경 시 리셋하지 않음 — 동일 레벨 변경만 리셋
                section_path.append(title)
            lines.append(f"{'#' * level} {title}")
            return

        # 이미지 포함 여부 확인
        blips = child_blips(para)
        if blips:
            for rel_id in blips:
                self._save_image(
                    rel_id, doc, section_path, image_seq,
                    extracted_images, images_dir, lines,
                )
            return

        text = para.text.strip()
        if text:
            lines.append(text)

    def _handle_table(
        self,
        table,
        section_path: list[str],
        table_seq: dict[str, int],
        lines: list[str],
        extracted_tables: list[ExtractedTable],
        tables_dir: Path,
        source_path: Path,
    ) -> None:
        sec_key = self._section_key(section_path)
        table_seq[sec_key] = table_seq.get(sec_key, 0) + 1
        seq = table_seq[sec_key]
        slug = _slugify(section_path[-1] if section_path else "", self._cfg.slugify_max_len)
        filename = f"Table_{slug}_{seq}.csv"
        csv_path = tables_dir / filename

        rows = []
        for row in table.rows:
            rows.append([cell.text.strip() for cell in row.cells])

        with csv_path.open("w", encoding="utf-8", newline="") as f:
            import csv as _csv
            writer = _csv.writer(f)
            header_lines = [
                f"# section: {' > '.join(section_path) if section_path else 'root'}",
                f"# source: {source_path.name}",
            ]
            for h in header_lines:
                f.write(h + "\n")
            writer.writerows(rows)

        placeholder = f"<!-- TABLE: {filename[:-4]} -->"
        lines.append(placeholder)
        extracted_tables.append(ExtractedTable(path=csv_path, section_path=list(section_path), seq=seq))

    def _save_image(
        self,
        rel_id: str,
        doc,
        section_path: list[str],
        image_seq: dict[str, int],
        extracted_images: list[ExtractedImage],
        images_dir: Path,
        lines: list[str],
    ) -> None:
        try:
            image_part = doc.part.related_parts[rel_id]
        except KeyError:
            return
        ext = image_part.partname.split(".")[-1].lower()
        sec_key = self._section_key(section_path)
        image_seq[sec_key] = image_seq.get(sec_key, 0) + 1
        seq = image_seq[sec_key]
        slug = _slugify(section_path[-1] if section_path else "", self._cfg.slugify_max_len)
        filename = f"Fig_{slug}_{seq}.{ext}"
        img_path = images_dir / filename
        img_path.write_bytes(image_part.blob)

        meta_path = images_dir / f"{filename}.meta"
        meta_path.write_text(
            f"section: {' > '.join(section_path) if section_path else 'root'}\n"
            f"seq: {seq}\n",
            encoding="utf-8",
        )
        placeholder = f"<!-- IMAGE: {filename} -->"
        lines.append(placeholder)
        extracted_images.append(ExtractedImage(
            path=img_path, section_path=list(section_path), seq=seq, ext=ext,
        ))


def child_blips(para) -> list[str]:
    nsmap = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    blips = para._element.findall(".//a:blip", nsmap)
    return [b.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
            for b in blips if b.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")]
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/parsers/test_docx_parser.py::TestDocxParserHeadings -v
```
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/spar/parsers/docx_parser.py tests/parsers/test_docx_parser.py
git commit -m "feat(parsers): implement DocxParser heading and paragraph parsing"
```

---

## Task 6: 표 추출 테스트

**Files:**
- Test: `tests/parsers/test_docx_parser.py` (추가)

- [ ] **Step 1: 실패하는 테스트 추가**

```python
import csv


class TestDocxParserTables:
    def test_table_csv_created(self, tmp_path: Path) -> None:
        def setup(doc):
            doc.add_heading("Config", level=1)
            table = doc.add_table(rows=2, cols=2)
            table.cell(0, 0).text = "Param"
            table.cell(0, 1).text = "Value"
            table.cell(1, 0).text = "TTT"
            table.cell(1, 1).text = "100ms"

        path = _make_docx(tmp_path, setup)
        cfg = DocxParseConfig(output_dir=tmp_path)
        result = DocxParser(cfg).parse(path)

        assert len(result.tables) == 1
        csv_path = result.tables[0].path
        assert csv_path.exists()
        assert "Table_Config_1" in csv_path.name

    def test_table_csv_content(self, tmp_path: Path) -> None:
        def setup(doc):
            doc.add_heading("Parameters", level=1)
            table = doc.add_table(rows=2, cols=2)
            table.cell(0, 0).text = "Name"
            table.cell(0, 1).text = "Value"
            table.cell(1, 0).text = "Alpha"
            table.cell(1, 1).text = "1.0"

        path = _make_docx(tmp_path, setup)
        cfg = DocxParseConfig(output_dir=tmp_path)
        result = DocxParser(cfg).parse(path)

        csv_path = result.tables[0].path
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith("# section:")
        assert lines[1].startswith("# source:")
        data_lines = [l for l in lines if not l.startswith("#")]
        rows = list(csv.reader(data_lines))
        assert rows[0] == ["Name", "Value"]
        assert rows[1] == ["Alpha", "1.0"]

    def test_table_placeholder_in_markdown(self, tmp_path: Path) -> None:
        def setup(doc):
            doc.add_heading("Intro", level=1)
            table = doc.add_table(rows=1, cols=1)
            table.cell(0, 0).text = "X"

        path = _make_docx(tmp_path, setup)
        cfg = DocxParseConfig(output_dir=tmp_path)
        result = DocxParser(cfg).parse(path)
        assert "<!-- TABLE: Table_Intro_1 -->" in result.markdown

    def test_table_seq_resets_per_section(self, tmp_path: Path) -> None:
        def setup(doc):
            doc.add_heading("Sec A", level=1)
            t1 = doc.add_table(rows=1, cols=1)
            t1.cell(0, 0).text = "A1"
            doc.add_heading("Sec B", level=1)
            t2 = doc.add_table(rows=1, cols=1)
            t2.cell(0, 0).text = "B1"

        path = _make_docx(tmp_path, setup)
        cfg = DocxParseConfig(output_dir=tmp_path)
        result = DocxParser(cfg).parse(path)
        assert result.tables[0].seq == 1
        assert result.tables[1].seq == 1
        assert "Sec-A" in result.tables[0].path.name
        assert "Sec-B" in result.tables[1].path.name
```

- [ ] **Step 2: 테스트 실행**

```bash
pytest tests/parsers/test_docx_parser.py::TestDocxParserTables -v
```
Expected: 4 passed (Task 5 구현이 표 처리 포함)  
실패 시: `_handle_table` 로직 확인 — seq 리셋은 `section_path` 변경 시 `table_seq.clear()` 호출 필요.

- [ ] **Step 3: Commit**

```bash
git add tests/parsers/test_docx_parser.py
git commit -m "test(parsers): add table extraction tests"
```

---

## Task 7: 이미지 추출 테스트

**Files:**
- Test: `tests/parsers/test_docx_parser.py` (추가)

이미지 테스트는 실제 PNG 바이너리를 `docx`에 삽입해야 함.

- [ ] **Step 1: 실패하는 테스트 추가**

```python
import io
from PIL import Image as _PilImage


def _make_png_bytes() -> bytes:
    img = _PilImage.new("RGB", (10, 10), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestDocxParserImages:
    def test_image_file_created(self, tmp_path: Path) -> None:
        def setup(doc):
            doc.add_heading("Diagrams", level=1)
            png_bytes = _make_png_bytes()
            img_stream = io.BytesIO(png_bytes)
            doc.add_picture(img_stream)

        path = _make_docx(tmp_path, setup)
        cfg = DocxParseConfig(output_dir=tmp_path)
        result = DocxParser(cfg).parse(path)

        assert len(result.images) == 1
        img_path = result.images[0].path
        assert img_path.exists()
        assert "Fig_Diagrams_1" in img_path.name

    def test_image_meta_file_created(self, tmp_path: Path) -> None:
        def setup(doc):
            doc.add_heading("Diagrams", level=1)
            doc.add_picture(io.BytesIO(_make_png_bytes()))

        path = _make_docx(tmp_path, setup)
        cfg = DocxParseConfig(output_dir=tmp_path)
        result = DocxParser(cfg).parse(path)

        meta_path = Path(str(result.images[0].path) + ".meta")
        assert meta_path.exists()
        content = meta_path.read_text(encoding="utf-8")
        assert "section: Diagrams" in content
        assert "seq: 1" in content

    def test_image_placeholder_in_markdown(self, tmp_path: Path) -> None:
        def setup(doc):
            doc.add_heading("Diagrams", level=1)
            doc.add_picture(io.BytesIO(_make_png_bytes()))

        path = _make_docx(tmp_path, setup)
        cfg = DocxParseConfig(output_dir=tmp_path)
        result = DocxParser(cfg).parse(path)
        assert "<!-- IMAGE: Fig_Diagrams_1" in result.markdown
```

> **참고:** `Pillow` 없으면 `pip install Pillow`. 또는 실제 PNG bytes 하드코딩 가능:
> ```python
> _MINIMAL_PNG = bytes([
>     0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A,0x00,0x00,0x00,0x0D,0x49,0x48,0x44,0x52,
>     0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x01,0x08,0x02,0x00,0x00,0x00,0x90,0x77,0x53,
>     0xDE,0x00,0x00,0x00,0x0C,0x49,0x44,0x41,0x54,0x08,0xD7,0x63,0xF8,0xCF,0xC0,0x00,
>     0x00,0x00,0x02,0x00,0x01,0xE2,0x21,0xBC,0x33,0x00,0x00,0x00,0x00,0x49,0x45,0x4E,
>     0x44,0xAE,0x42,0x60,0x82,
> ])
> ```

- [ ] **Step 2: 테스트 실행**

```bash
pytest tests/parsers/test_docx_parser.py::TestDocxParserImages -v
```
Expected: 3 passed

- [ ] **Step 3: Commit**

```bash
git add tests/parsers/test_docx_parser.py
git commit -m "test(parsers): add image extraction tests"
```

---

## Task 8: CLI (parse_docx.py)

**Files:**
- Create: `scripts/parse_docx.py`

- [ ] **Step 1: CLI 작성**

`scripts/parse_docx.py` 생성:
```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from spar.parsers.docx_config import DocxParseConfig
from spar.parsers.docx_parser import DocxParser


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse .docx to Markdown + CSV/images")
    parser.add_argument("--file", required=True, type=Path, help=".docx 파일 경로")
    parser.add_argument("--output", default=Path("output"), type=Path, help="출력 디렉토리")
    parser.add_argument("--heading-depth", default=2, type=int, help="섹션 경계 heading 레벨 (1~3)")
    args = parser.parse_args()

    cfg = DocxParseConfig(
        heading_depth=args.heading_depth,
        output_dir=args.output,
    )
    result = DocxParser(cfg).parse(args.file)

    md_path = args.output / (args.file.stem + ".md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(result.markdown, encoding="utf-8")

    print(f"Parsed: {md_path}")
    print(f"Tables: {len(result.tables)} → {args.output / 'tables'}/")
    print(f"Images: {len(result.images)} → {args.output / 'images'}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: CLI 동작 확인 (실제 docx 없으므로 help만)**

```bash
python scripts/parse_docx.py --help
```
Expected: usage 메시지 출력, 에러 없음

- [ ] **Step 3: Commit**

```bash
git add scripts/parse_docx.py
git commit -m "feat(parsers): add parse_docx CLI script"
```

---

## Task 9: 전체 테스트 실행 + docs 업데이트

**Files:**
- Modify: `docs/prd.md`
- Modify: `AGENTS.md`
- Modify: `README.md`

- [ ] **Step 1: 전체 테스트 실행**

```bash
pytest tests/parsers/ -v
```
Expected: 전체 통과

- [ ] **Step 2: `docs/prd.md` 업데이트**

Task 1.1 항목에서 다음 체크박스 체크:
```
- [x] 각 유형별 PDF/문서 파서 개발 (Word .docx)
- [x] **산출물**: `parsers/` 디렉토리, `parsers/docx_parser.py`, `parsers/docx_config.py`
```

- [ ] **Step 3: `AGENTS.md` 디렉토리 맵 업데이트**

`parsers/` 항목 설명 갱신:
```
├── parsers/         # 문서 유형별 파서 — docx_parser.py (DocxParser), docx_config.py (DocxParseConfig) (Task 1.1 ✅ DOCX)
```

- [ ] **Step 4: Final commit**

```bash
git add docs/prd.md AGENTS.md README.md
git commit -m "docs: update Task 1.1 progress — docx parser complete"
```
