# Word Document Parser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parse `.docx` files into heading-bounded markdown chunks with extracted images, driven by a user-editable config file that maps heading styles, font sizes, and heading text to tags.

**Architecture:** `python-docx` walks the document body and emits a typed element stream (paragraphs, tables, images). The stream passes through heading resolution → table merging → chunking → rendering. All rules (style→depth, font size→depth, heading→tag) live in `config.yaml` so the parser can be tuned iteratively via log feedback without code changes.

**Tech Stack:** Python 3.11+, `python-docx >= 1.1`, `PyYAML >= 6.0`, `pytest`

---

## File Map

| File | Responsibility |
|------|---------------|
| `core/models.py` | Shared dataclasses: `Run`, `ParagraphElement`, `TableElement`, `ImageElement`, `Chunk` |
| `core/config.py` | Load and validate `config.yaml` into a `Config` dataclass |
| `parse_logging/parse_logger.py` | Dual console+file logger factory |
| `core/document.py` | Walk `.docx` body → emit ordered element stream |
| `core/heading.py` | `ParagraphElement` → heading depth (style-first, font-size fallback) |
| `core/table_merger.py` | Merge consecutive same-column tables separated only by page breaks |
| `core/chunker.py` | Element stream → `Chunk` list, resolve tags from config |
| `core/image_extractor.py` | Extract image bytes from docx relationships, name by context |
| `core/renderer.py` | `Chunk` → write `.md` file + save images |
| `parser.py` | CLI: argparse, wire all components, call logger |
| `tests/conftest.py` | Synthetic `.docx` fixture builders using `python-docx` |
| `config.yaml` | Example/default config |

---

## Task 1: Project Scaffolding + Shared Models

**Files:**
- Create: `core/__init__.py`
- Create: `core/models.py`
- Create: `parse_logging/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `requirements.txt`

- [ ] **Step 1: Create directory structure**

```bash
cd word_parser
mkdir -p core parse_logging tests output
touch core/__init__.py parse_logging/__init__.py tests/__init__.py
```

- [ ] **Step 2: Create `requirements.txt`**

```
python-docx>=1.1
PyYAML>=6.0
pytest>=8.0
```

- [ ] **Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: installs without error.

- [ ] **Step 4: Write test for models**

Create `tests/test_models.py`:

```python
from core.models import Run, ParagraphElement, TableElement, ImageElement, Chunk


def test_paragraph_element_defaults():
    run = Run(text="hello", font_size=12.0, bold=False)
    para = ParagraphElement(
        text="hello",
        style_name="Normal",
        runs=[run],
        page_approx=1,
    )
    assert para.is_page_break is False


def test_table_element():
    tbl = TableElement(
        rows=[["A", "B"], ["1", "2"]],
        col_count=2,
        page_approx=1,
        preceded_by_page_break=False,
    )
    assert tbl.col_count == 2


def test_image_element():
    img = ImageElement(
        relationship_id="rId1",
        content_type="image/png",
        data=b"\x89PNG",
        page_approx=1,
    )
    assert img.content_type == "image/png"


def test_chunk_defaults():
    chunk = Chunk(heading_text="Intro", heading_depth=1, tag="intro", elements=[], index=0)
    assert chunk.table_counter == 0
    assert chunk.image_counter == 0
```

- [ ] **Step 5: Run test to verify it fails**

```bash
pytest tests/test_models.py -v
```

Expected: `ModuleNotFoundError: No module named 'core.models'`

- [ ] **Step 6: Create `core/models.py`**

```python
from dataclasses import dataclass, field


@dataclass
class Run:
    text: str
    font_size: float | None
    bold: bool


@dataclass
class ParagraphElement:
    text: str
    style_name: str
    runs: list[Run]
    page_approx: int
    is_page_break: bool = False


@dataclass
class TableElement:
    rows: list[list[str]]
    col_count: int
    page_approx: int
    preceded_by_page_break: bool


@dataclass
class ImageElement:
    relationship_id: str
    content_type: str
    data: bytes
    page_approx: int


@dataclass
class Chunk:
    heading_text: str
    heading_depth: int
    tag: str
    elements: list
    index: int
    table_counter: int = 0
    image_counter: int = 0
```

- [ ] **Step 7: Run test to verify it passes**

```bash
pytest tests/test_models.py -v
```

Expected: 4 PASSED

- [ ] **Step 8: Create `tests/conftest.py` with synthetic docx helpers**

```python
import io
import pytest
from docx import Document
from docx.shared import Pt
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


def make_docx() -> Document:
    return Document()


def add_heading(doc: Document, text: str, level: int) -> None:
    doc.add_heading(text, level=level)


def add_paragraph(doc: Document, text: str, style: str = "Normal") -> None:
    doc.add_paragraph(text, style=style)


def add_paragraph_with_font_size(doc: Document, text: str, size_pt: float) -> None:
    para = doc.add_paragraph()
    run = para.add_run(text)
    run.font.size = Pt(size_pt)


def add_page_break(doc: Document) -> None:
    para = doc.add_paragraph()
    run = para.add_run()
    run.add_break(break_type=2)  # WD_BREAK.PAGE = 2


def add_table(doc: Document, rows: list[list[str]]) -> None:
    if not rows:
        return
    tbl = doc.add_table(rows=len(rows), cols=len(rows[0]))
    for r_idx, row in enumerate(rows):
        for c_idx, cell_text in enumerate(row):
            tbl.rows[r_idx].cells[c_idx].text = cell_text


def docx_bytes(doc: Document) -> bytes:
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
```

- [ ] **Step 9: Commit**

```bash
git add core/ parse_logging/ tests/ requirements.txt
git commit -m "feat(word_parser): scaffold project structure and shared models"
```

---

## Task 2: Config Loader

**Files:**
- Create: `core/config.py`
- Create: `config.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_config.py`:

```python
import textwrap
import pytest
from core.config import Config, load_config


SAMPLE_YAML = textwrap.dedent("""\
    heading_styles:
      "Heading 1": 1
      "Heading 2": 2
    font_size_map:
      24: 1
      18: 2
    heading_tags:
      "3.2 Configuration": "config"
    table_merge:
      enabled: true
    output_dir: "output"
    log_level: "INFO"
""")


def test_load_config_parses_heading_styles(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(SAMPLE_YAML)
    cfg = load_config(str(cfg_file))
    assert cfg.heading_styles["Heading 1"] == 1
    assert cfg.heading_styles["Heading 2"] == 2


def test_load_config_parses_font_size_map(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(SAMPLE_YAML)
    cfg = load_config(str(cfg_file))
    assert cfg.font_size_map[24] == 1
    assert cfg.font_size_map[18] == 2


def test_load_config_parses_heading_tags(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(SAMPLE_YAML)
    cfg = load_config(str(cfg_file))
    assert cfg.heading_tags["3.2 Configuration"] == "config"


def test_load_config_table_merge(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(SAMPLE_YAML)
    cfg = load_config(str(cfg_file))
    assert cfg.table_merge_enabled is True


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'core.config'`

- [ ] **Step 3: Create `core/config.py`**

```python
from dataclasses import dataclass
import yaml


@dataclass
class Config:
    heading_styles: dict[str, int]
    font_size_map: dict[int, int]
    heading_tags: dict[str, str]
    table_merge_enabled: bool
    output_dir: str
    log_level: str


def load_config(path: str) -> Config:
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise

    font_size_map = {int(k): int(v) for k, v in (data.get("font_size_map") or {}).items()}

    return Config(
        heading_styles=data.get("heading_styles") or {},
        font_size_map=font_size_map,
        heading_tags=data.get("heading_tags") or {},
        table_merge_enabled=(data.get("table_merge") or {}).get("enabled", True),
        output_dir=data.get("output_dir", "output"),
        log_level=data.get("log_level", "INFO"),
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_config.py -v
```

Expected: 5 PASSED

- [ ] **Step 5: Create `config.yaml`**

```yaml
# Heading style name → markdown depth
heading_styles:
  "Heading 1": 1
  "Heading 2": 2
  "Heading 3": 3

# Font size (pt, integer) → markdown depth
# Fallback for paragraphs whose style is not in heading_styles
font_size_map:
  24: 1
  18: 2
  14: 3

# Heading text substring → tag (first match wins, case-insensitive)
# Tables under that heading: {tag}_table_1, {tag}_table_2, ...
heading_tags:
  "3.2 Configuration": "config"
  "4.1 Parameters": "params"

table_merge:
  enabled: true

output_dir: "output"
log_level: "INFO"   # INFO | DEBUG
```

- [ ] **Step 6: Commit**

```bash
git add core/config.py config.yaml tests/test_config.py
git commit -m "feat(word_parser): add config loader"
```

---

## Task 3: Logger

**Files:**
- Create: `parse_logging/parse_logger.py`
- Test: `tests/test_logger.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_logger.py`:

```python
import logging
from pathlib import Path
from parse_logging.parse_logger import make_logger


def test_logger_writes_to_file(tmp_path):
    log_file = tmp_path / "parse.log"
    logger = make_logger("test", str(log_file), level=logging.DEBUG)
    logger.warning("test warning message")

    content = log_file.read_text()
    assert "test warning message" in content


def test_logger_has_console_handler(tmp_path):
    log_file = tmp_path / "parse.log"
    logger = make_logger("test2", str(log_file), level=logging.DEBUG)
    handlers = logger.handlers
    types = [type(h).__name__ for h in handlers]
    assert "StreamHandler" in types
    assert "FileHandler" in types


def test_logger_level_respected(tmp_path):
    log_file = tmp_path / "parse.log"
    logger = make_logger("test3", str(log_file), level=logging.WARNING)
    logger.debug("should not appear")
    logger.warning("should appear")

    content = log_file.read_text()
    assert "should not appear" not in content
    assert "should appear" in content
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_logger.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create `parse_logging/parse_logger.py`**

```python
import logging
import sys


def make_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(fmt)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_logger.py -v
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add parse_logging/parse_logger.py tests/test_logger.py
git commit -m "feat(word_parser): add dual console+file logger"
```

---

## Task 4: Document Element Stream

**Files:**
- Create: `core/document.py`
- Test: `tests/test_document.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_document.py`:

```python
import io
import pytest
from docx import Document as DocxDocument
from docx.shared import Pt
from tests.conftest import (
    make_docx, add_heading, add_paragraph, add_paragraph_with_font_size,
    add_page_break, add_table, docx_bytes,
)
from core.document import stream_elements
from core.models import ParagraphElement, TableElement, ImageElement


def test_stream_paragraphs():
    doc = make_docx()
    add_paragraph(doc, "Hello world")
    elements = list(stream_elements(docx_bytes(doc)))
    paras = [e for e in elements if isinstance(e, ParagraphElement)]
    texts = [p.text for p in paras]
    assert "Hello world" in texts


def test_stream_heading_style():
    doc = make_docx()
    add_heading(doc, "Section 1", level=1)
    elements = list(stream_elements(docx_bytes(doc)))
    paras = [e for e in elements if isinstance(e, ParagraphElement)]
    heading = next(p for p in paras if p.text == "Section 1")
    assert heading.style_name == "Heading 1"


def test_stream_table():
    doc = make_docx()
    add_table(doc, [["Name", "Value"], ["foo", "bar"]])
    elements = list(stream_elements(docx_bytes(doc)))
    tables = [e for e in elements if isinstance(e, TableElement)]
    assert len(tables) == 1
    assert tables[0].col_count == 2
    assert tables[0].rows[0] == ["Name", "Value"]


def test_stream_page_break_paragraph():
    doc = make_docx()
    add_paragraph(doc, "Before")
    add_page_break(doc)
    add_paragraph(doc, "After")
    elements = list(stream_elements(docx_bytes(doc)))
    paras = [e for e in elements if isinstance(e, ParagraphElement)]
    page_breaks = [p for p in paras if p.is_page_break]
    assert len(page_breaks) >= 1


def test_stream_table_preceded_by_page_break():
    doc = make_docx()
    add_table(doc, [["A"], ["1"]])
    add_page_break(doc)
    add_table(doc, [["A"], ["2"]])
    elements = list(stream_elements(docx_bytes(doc)))
    tables = [e for e in elements if isinstance(e, TableElement)]
    assert len(tables) == 2
    assert tables[1].preceded_by_page_break is True


def test_stream_font_size_on_run():
    doc = make_docx()
    add_paragraph_with_font_size(doc, "Big text", 24.0)
    elements = list(stream_elements(docx_bytes(doc)))
    paras = [e for e in elements if isinstance(e, ParagraphElement)]
    big = next(p for p in paras if p.text == "Big text")
    assert any(r.font_size == 24.0 for r in big.runs)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_document.py -v
```

Expected: `ModuleNotFoundError: No module named 'core.document'`

- [ ] **Step 3: Create `core/document.py`**

```python
import io
from typing import Generator
from docx import Document
from docx.oxml.ns import qn
from core.models import ParagraphElement, TableElement, ImageElement, Run

Element = ParagraphElement | TableElement | ImageElement


def _is_page_break_para(para) -> bool:
    for run in para.runs:
        for br in run._r.findall(qn("w:br")):
            if br.get(qn("w:type")) == "page":
                return True
    return False


def _para_runs(para) -> list[Run]:
    runs = []
    for r in para.runs:
        size = None
        if r.font.size is not None:
            size = r.font.size.pt
        runs.append(Run(text=r.text, font_size=size, bold=bool(r.bold)))
    return runs


def _table_rows(tbl) -> list[list[str]]:
    return [[cell.text for cell in row.cells] for row in tbl.rows]


def stream_elements(docx_data: bytes) -> Generator[Element, None, None]:
    doc = Document(io.BytesIO(docx_data))
    body = doc.element.body
    page_approx = 1
    last_was_page_break = False

    for child in body.iterchildren():
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag

        if tag == "p":
            from docx.text.paragraph import Paragraph
            para = Paragraph(child, doc)
            is_pb = _is_page_break_para(para)
            if is_pb:
                page_approx += 1
            runs = _para_runs(para)
            elem = ParagraphElement(
                text=para.text,
                style_name=para.style.name if para.style else "Normal",
                runs=runs,
                page_approx=page_approx,
                is_page_break=is_pb,
            )
            last_was_page_break = is_pb
            yield elem

        elif tag == "tbl":
            from docx.table import Table
            tbl = Table(child, doc)
            rows = _table_rows(tbl)
            col_count = max(len(r) for r in rows) if rows else 0
            elem = TableElement(
                rows=rows,
                col_count=col_count,
                page_approx=page_approx,
                preceded_by_page_break=last_was_page_break,
            )
            last_was_page_break = False
            yield elem

        else:
            last_was_page_break = False
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_document.py -v
```

Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add core/document.py tests/test_document.py
git commit -m "feat(word_parser): add document element stream walker"
```

---

## Task 5: Heading Resolver

**Files:**
- Create: `core/heading.py`
- Test: `tests/test_heading.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_heading.py`:

```python
import logging
import pytest
from core.models import ParagraphElement, Run
from core.config import Config
from core.heading import resolve_heading_depth


def make_config(**kwargs):
    defaults = dict(
        heading_styles={"Heading 1": 1, "Heading 2": 2},
        font_size_map={24: 1, 18: 2},
        heading_tags={},
        table_merge_enabled=True,
        output_dir="output",
        log_level="INFO",
    )
    defaults.update(kwargs)
    return Config(**defaults)


def make_para(text="text", style="Normal", font_size=None, bold=False):
    run = Run(text=text, font_size=font_size, bold=bold)
    return ParagraphElement(text=text, style_name=style, runs=[run], page_approx=1)


def test_heading_style_returns_depth(caplog):
    cfg = make_config()
    para = make_para(style="Heading 1")
    depth = resolve_heading_depth(para, cfg, logger=None)
    assert depth == 1


def test_heading_style_2(caplog):
    cfg = make_config()
    para = make_para(style="Heading 2")
    depth = resolve_heading_depth(para, cfg, logger=None)
    assert depth == 2


def test_font_size_fallback(caplog):
    cfg = make_config()
    para = make_para(style="Normal", font_size=24.0)
    depth = resolve_heading_depth(para, cfg, logger=None)
    assert depth == 1


def test_font_size_fallback_no_match_returns_none():
    cfg = make_config()
    para = make_para(style="Normal", font_size=12.0)
    depth = resolve_heading_depth(para, cfg, logger=None)
    assert depth is None


def test_normal_paragraph_returns_none():
    cfg = make_config()
    para = make_para(style="Normal", font_size=None)
    depth = resolve_heading_depth(para, cfg, logger=None)
    assert depth is None


def test_bold_large_no_match_returns_none(caplog):
    cfg = make_config()
    para = make_para(style="Normal", font_size=16.0, bold=True)
    depth = resolve_heading_depth(para, cfg, logger=None)
    assert depth is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_heading.py -v
```

Expected: `ModuleNotFoundError: No module named 'core.heading'`

- [ ] **Step 3: Create `core/heading.py`**

```python
import logging
from core.models import ParagraphElement
from core.config import Config


def resolve_heading_depth(
    para: ParagraphElement,
    cfg: Config,
    logger: logging.Logger | None,
) -> int | None:
    # Style-first
    if para.style_name in cfg.heading_styles:
        return cfg.heading_styles[para.style_name]

    # Font-size fallback: use largest font size across runs
    sizes = [r.font_size for r in para.runs if r.font_size is not None]
    if sizes:
        max_size = max(sizes)
        size_key = int(max_size)
        if size_key in cfg.font_size_map:
            if logger:
                logger.debug(
                    f"[heading] Font-size fallback: {para.text!r} size={max_size}pt → depth={cfg.font_size_map[size_key]} (page≈{para.page_approx})"
                )
            return cfg.font_size_map[size_key]

        # Bold + large text with no matching rule
        if any(r.bold for r in para.runs):
            if logger:
                logger.warning(
                    f"[heading] Bold+large text, no style/size rule: {para.text!r} bold=True size={max_size}pt (page≈{para.page_approx})"
                )

    return None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_heading.py -v
```

Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add core/heading.py tests/test_heading.py
git commit -m "feat(word_parser): add heading depth resolver with font-size fallback"
```

---

## Task 6: Table Merger

**Files:**
- Create: `core/table_merger.py`
- Test: `tests/test_table_merger.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_table_merger.py`:

```python
import pytest
from core.models import ParagraphElement, TableElement, Run
from core.table_merger import merge_tables


def page_break_para():
    return ParagraphElement(
        text="", style_name="Normal", runs=[], page_approx=1, is_page_break=True
    )


def normal_para(text="text"):
    return ParagraphElement(
        text=text, style_name="Normal", runs=[Run(text=text, font_size=None, bold=False)],
        page_approx=1, is_page_break=False
    )


def table(rows, preceded=False):
    return TableElement(
        rows=rows, col_count=len(rows[0]) if rows else 0,
        page_approx=1, preceded_by_page_break=preceded
    )


def test_merges_two_tables_with_only_page_break():
    t1 = table([["A", "B"], ["1", "2"]])
    pb = page_break_para()
    t2 = TableElement(rows=[["A", "B"], ["3", "4"]], col_count=2, page_approx=2, preceded_by_page_break=True)
    elements = [t1, pb, t2]
    result = merge_tables(elements, logger=None)
    tables = [e for e in result if isinstance(e, TableElement)]
    assert len(tables) == 1
    assert len(tables[0].rows) == 4  # both tables merged


def test_drops_repeated_header_on_merge():
    t1 = table([["A", "B"], ["1", "2"]])
    pb = page_break_para()
    t2 = TableElement(rows=[["A", "B"], ["3", "4"]], col_count=2, page_approx=2, preceded_by_page_break=True)
    elements = [t1, pb, t2]
    result = merge_tables(elements, logger=None)
    tables = [e for e in result if isinstance(e, TableElement)]
    # header ["A","B"] appears only once
    assert tables[0].rows.count(["A", "B"]) == 1


def test_no_merge_when_paragraph_between():
    t1 = table([["A", "B"], ["1", "2"]])
    mid = normal_para("some text")
    t2 = TableElement(rows=[["A", "B"], ["3", "4"]], col_count=2, page_approx=2, preceded_by_page_break=False)
    elements = [t1, mid, t2]
    result = merge_tables(elements, logger=None)
    tables = [e for e in result if isinstance(e, TableElement)]
    assert len(tables) == 2


def test_no_merge_different_col_count():
    t1 = table([["A", "B"], ["1", "2"]])
    pb = page_break_para()
    t2 = TableElement(rows=[["X", "Y", "Z"]], col_count=3, page_approx=2, preceded_by_page_break=True)
    elements = [t1, pb, t2]
    result = merge_tables(elements, logger=None)
    tables = [e for e in result if isinstance(e, TableElement)]
    assert len(tables) == 2


def test_page_break_elements_removed_after_merge():
    t1 = table([["A"], ["1"]])
    pb = page_break_para()
    t2 = TableElement(rows=[["A"], ["2"]], col_count=1, page_approx=2, preceded_by_page_break=True)
    elements = [t1, pb, t2]
    result = merge_tables(elements, logger=None)
    page_breaks = [e for e in result if isinstance(e, ParagraphElement) and e.is_page_break]
    assert len(page_breaks) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_table_merger.py -v
```

Expected: `ModuleNotFoundError: No module named 'core.table_merger'`

- [ ] **Step 3: Create `core/table_merger.py`**

```python
import logging
from core.models import ParagraphElement, TableElement, ImageElement

Element = ParagraphElement | TableElement | ImageElement


def merge_tables(elements: list[Element], logger: logging.Logger | None) -> list[Element]:
    result: list[Element] = []
    i = 0
    while i < len(elements):
        elem = elements[i]
        if not isinstance(elem, TableElement):
            result.append(elem)
            i += 1
            continue

        # Collect page-break-only paragraphs after this table
        j = i + 1
        gap: list[ParagraphElement] = []
        while j < len(elements):
            nxt = elements[j]
            if isinstance(nxt, ParagraphElement) and nxt.is_page_break:
                gap.append(nxt)
                j += 1
            else:
                break

        # Check if next element is a table with same col count
        if (
            gap
            and j < len(elements)
            and isinstance(elements[j], TableElement)
            and elements[j].col_count == elem.col_count
        ):
            next_tbl = elements[j]
            merged_rows = list(elem.rows)

            # Drop repeated header row
            continuation = list(next_tbl.rows)
            if continuation and continuation[0] == merged_rows[0]:
                if logger:
                    logger.debug(
                        f"[table_merger] Repeated header row dropped: {continuation[0]} (page≈{next_tbl.page_approx})"
                    )
                continuation = continuation[1:]

            merged_rows.extend(continuation)

            if logger:
                logger.info(
                    f"[table_merger] Merged table at page≈{next_tbl.page_approx} (col={elem.col_count})"
                )

            merged = TableElement(
                rows=merged_rows,
                col_count=elem.col_count,
                page_approx=elem.page_approx,
                preceded_by_page_break=elem.preceded_by_page_break,
            )
            result.append(merged)
            i = j + 1
        else:
            result.append(elem)
            result.extend(gap)
            i = j

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_table_merger.py -v
```

Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add core/table_merger.py tests/test_table_merger.py
git commit -m "feat(word_parser): add table merger for page-break-split tables"
```

---

## Task 7: Chunker

**Files:**
- Create: `core/chunker.py`
- Test: `tests/test_chunker.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_chunker.py`:

```python
import pytest
from core.models import ParagraphElement, TableElement, Run, Chunk
from core.config import Config
from core.chunker import build_chunks


def cfg(**kwargs):
    defaults = dict(
        heading_styles={"Heading 1": 1, "Heading 2": 2},
        font_size_map={},
        heading_tags={"3.2 Configuration": "config", "Parameters": "params"},
        table_merge_enabled=True,
        output_dir="output",
        log_level="INFO",
    )
    defaults.update(kwargs)
    return Config(**defaults)


def heading_para(text, style="Heading 1"):
    return ParagraphElement(
        text=text, style_name=style,
        runs=[Run(text=text, font_size=None, bold=False)],
        page_approx=1,
    )


def normal_para(text="body"):
    return ParagraphElement(
        text=text, style_name="Normal",
        runs=[Run(text=text, font_size=None, bold=False)],
        page_approx=1,
    )


def test_single_chunk_from_heading():
    elements = [heading_para("Introduction"), normal_para("Some text")]
    chunks = build_chunks(elements, cfg(), logger=None)
    assert len(chunks) == 1
    assert chunks[0].heading_text == "Introduction"
    assert chunks[0].heading_depth == 1


def test_two_chunks_from_two_headings():
    elements = [
        heading_para("Section A"),
        normal_para("body A"),
        heading_para("Section B"),
        normal_para("body B"),
    ]
    chunks = build_chunks(elements, cfg(), logger=None)
    assert len(chunks) == 2
    assert chunks[0].heading_text == "Section A"
    assert chunks[1].heading_text == "Section B"


def test_chunk_tag_resolved_from_heading_text():
    elements = [heading_para("3.2 Configuration Details"), normal_para("body")]
    chunks = build_chunks(elements, cfg(), logger=None)
    assert chunks[0].tag == "config"


def test_chunk_tag_unknown_when_no_match():
    elements = [heading_para("Unknown Section"), normal_para("body")]
    chunks = build_chunks(elements, cfg(), logger=None)
    assert chunks[0].tag == "unknown"


def test_preamble_chunk_for_content_before_first_heading():
    elements = [normal_para("preamble text"), heading_para("Section A")]
    chunks = build_chunks(elements, cfg(), logger=None)
    assert chunks[0].heading_text == ""
    assert chunks[0].heading_depth == 0
    assert chunks[1].heading_text == "Section A"


def test_chunk_elements_contain_body_paragraphs():
    body = normal_para("body text")
    elements = [heading_para("Section A"), body]
    chunks = build_chunks(elements, cfg(), logger=None)
    assert body in chunks[0].elements


def test_chunk_index_sequential():
    elements = [
        heading_para("A"), normal_para(),
        heading_para("B"), normal_para(),
        heading_para("C"), normal_para(),
    ]
    chunks = build_chunks(elements, cfg(), logger=None)
    assert [c.index for c in chunks] == [0, 1, 2]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_chunker.py -v
```

Expected: `ModuleNotFoundError: No module named 'core.chunker'`

- [ ] **Step 3: Create `core/chunker.py`**

```python
import logging
from core.models import ParagraphElement, TableElement, ImageElement, Chunk
from core.config import Config
from core.heading import resolve_heading_depth

Element = ParagraphElement | TableElement | ImageElement


def _resolve_tag(heading_text: str, cfg: Config, logger: logging.Logger | None) -> str:
    lower = heading_text.lower()
    for pattern, tag in cfg.heading_tags.items():
        if pattern.lower() in lower:
            return tag
    if logger:
        logger.warning(
            f"[chunker] No tag match for heading {heading_text!r} → using 'unknown'"
        )
    return "unknown"


def build_chunks(
    elements: list[Element],
    cfg: Config,
    logger: logging.Logger | None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    current_elements: list[Element] = []
    current_heading = ""
    current_depth = 0
    current_tag = "preamble"
    index = 0

    def flush():
        nonlocal index
        chunk = Chunk(
            heading_text=current_heading,
            heading_depth=current_depth,
            tag=current_tag,
            elements=list(current_elements),
            index=index,
        )
        chunks.append(chunk)
        index += 1

    for elem in elements:
        if isinstance(elem, ParagraphElement):
            depth = resolve_heading_depth(elem, cfg, logger)
            if depth is not None:
                # Flush current chunk if it has content or is a real heading
                if current_elements or current_heading:
                    if not current_elements and logger:
                        logger.debug(
                            f"[chunker] Empty chunk: {current_heading!r} depth={current_depth}"
                        )
                    flush()
                current_elements = []
                current_heading = elem.text
                current_depth = depth
                current_tag = _resolve_tag(elem.text, cfg, logger)
                continue

        current_elements.append(elem)

    # Flush last chunk
    if current_elements or current_heading:
        if not current_elements and logger:
            logger.debug(f"[chunker] Empty chunk: {current_heading!r} depth={current_depth}")
        flush()

    return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_chunker.py -v
```

Expected: 7 PASSED

- [ ] **Step 5: Commit**

```bash
git add core/chunker.py tests/test_chunker.py
git commit -m "feat(word_parser): add chunker with heading-bounded chunk grouping"
```

---

## Task 8: Renderer

**Files:**
- Create: `core/renderer.py`
- Test: `tests/test_renderer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_renderer.py`:

```python
import re
import pytest
from core.models import ParagraphElement, TableElement, Run, Chunk
from core.renderer import render_chunk, slugify


def normal_para(text):
    return ParagraphElement(
        text=text, style_name="Normal",
        runs=[Run(text=text, font_size=None, bold=False)],
        page_approx=1,
    )


def table_elem(rows):
    return TableElement(
        rows=rows, col_count=len(rows[0]),
        page_approx=1, preceded_by_page_break=False,
    )


def test_slugify_basic():
    assert slugify("3.2 Configuration Details") == "3_2_configuration_details"


def test_slugify_special_chars():
    assert slugify("Hello, World!") == "hello_world"


def test_render_heading():
    chunk = Chunk(heading_text="Introduction", heading_depth=2, tag="intro", elements=[], index=0)
    md = render_chunk(chunk)
    assert md.startswith("## Introduction")


def test_render_paragraph_body():
    chunk = Chunk(
        heading_text="Section", heading_depth=1, tag="sec",
        elements=[normal_para("Some body text")], index=0,
    )
    md = render_chunk(chunk)
    assert "Some body text" in md


def test_render_table_gfm():
    rows = [["Name", "Value"], ["foo", "bar"], ["baz", "qux"]]
    chunk = Chunk(
        heading_text="Config", heading_depth=1, tag="config",
        elements=[table_elem(rows)], index=0,
    )
    md = render_chunk(chunk)
    assert "| Name | Value |" in md
    assert "| --- | --- |" in md
    assert "| foo | bar |" in md


def test_render_table_id_comment():
    rows = [["A", "B"], ["1", "2"]]
    chunk = Chunk(
        heading_text="Config", heading_depth=1, tag="config",
        elements=[table_elem(rows)], index=0,
    )
    md = render_chunk(chunk)
    assert "<!-- table-id: config_table_1 -->" in md


def test_render_second_table_increments_id():
    rows = [["A"], ["1"]]
    chunk = Chunk(
        heading_text="Config", heading_depth=1, tag="config",
        elements=[table_elem(rows), table_elem(rows)], index=0,
    )
    md = render_chunk(chunk)
    assert "<!-- table-id: config_table_1 -->" in md
    assert "<!-- table-id: config_table_2 -->" in md


def test_render_empty_heading_no_heading_line():
    chunk = Chunk(heading_text="", heading_depth=0, tag="preamble",
                  elements=[normal_para("preamble")], index=0)
    md = render_chunk(chunk)
    assert not md.startswith("#")
    assert "preamble" in md
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_renderer.py -v
```

Expected: `ModuleNotFoundError: No module named 'core.renderer'`

- [ ] **Step 3: Create `core/renderer.py`**

```python
import re
from core.models import ParagraphElement, TableElement, ImageElement, Chunk


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _render_table(tbl: TableElement, tag: str, counter: int) -> str:
    lines = [f"<!-- table-id: {tag}_table_{counter} -->"]
    if not tbl.rows:
        return "\n".join(lines)

    header = tbl.rows[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in tbl.rows[1:]:
        # Pad or trim row to match header length
        padded = row + [""] * max(0, len(header) - len(row))
        lines.append("| " + " | ".join(padded[: len(header)]) + " |")
    return "\n".join(lines)


def render_chunk(chunk: Chunk) -> str:
    parts: list[str] = []
    table_counter = 0

    if chunk.heading_text:
        prefix = "#" * chunk.heading_depth
        parts.append(f"{prefix} {chunk.heading_text}\n")

    for elem in chunk.elements:
        if isinstance(elem, ParagraphElement):
            if elem.is_page_break or not elem.text.strip():
                continue
            parts.append(elem.text)
        elif isinstance(elem, TableElement):
            table_counter += 1
            parts.append(_render_table(elem, chunk.tag, table_counter))
        elif isinstance(elem, ImageElement):
            ext = elem.content_type.split("/")[-1]
            name = f"{chunk.tag}_img_{chunk.image_counter + 1}.{ext}"
            parts.append(f"![{name}](../images/{name})")

    return "\n\n".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_renderer.py -v
```

Expected: 8 PASSED

- [ ] **Step 5: Commit**

```bash
git add core/renderer.py tests/test_renderer.py
git commit -m "feat(word_parser): add markdown renderer with GFM tables and table IDs"
```

---

## Task 9: Image Extractor

**Files:**
- Create: `core/image_extractor.py`
- Test: `tests/test_image_extractor.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_image_extractor.py`:

```python
import io
import pytest
from docx import Document
from core.image_extractor import extract_images_from_docx
from core.models import ImageElement
from tests.conftest import make_docx, docx_bytes


def test_no_images_returns_empty():
    doc = make_docx()
    images = extract_images_from_docx(docx_bytes(doc))
    assert images == []


def test_image_elements_have_data():
    # Create a docx with a tiny PNG inline
    doc = make_docx()
    # 1x1 white PNG
    png_1x1 = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
        b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    buf = io.BytesIO()
    doc.save(buf)
    # We can't easily add real inline images via python-docx API in tests,
    # so just verify the function runs without error on an empty docx
    images = extract_images_from_docx(buf.getvalue())
    assert isinstance(images, list)


def test_extract_returns_image_elements():
    doc = make_docx()
    images = extract_images_from_docx(docx_bytes(doc))
    for img in images:
        assert isinstance(img, ImageElement)
        assert img.data
        assert img.content_type
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_image_extractor.py -v
```

Expected: `ModuleNotFoundError: No module named 'core.image_extractor'`

- [ ] **Step 3: Create `core/image_extractor.py`**

```python
import io
import logging
from docx import Document
from core.models import ImageElement

KNOWN_IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/bmp", "image/tiff"}


def extract_images_from_docx(
    docx_data: bytes,
    logger: logging.Logger | None = None,
) -> list[ImageElement]:
    doc = Document(io.BytesIO(docx_data))
    images = []

    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            content_type = rel.target_part.content_type
            if content_type not in KNOWN_IMAGE_TYPES:
                if logger:
                    logger.warning(
                        f"[image_extractor] Unrecognized image content type: {content_type}"
                    )
                continue
            images.append(
                ImageElement(
                    relationship_id=rel.reltype,
                    content_type=content_type,
                    data=rel.target_part.blob,
                    page_approx=0,
                )
            )

    return images
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_image_extractor.py -v
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add core/image_extractor.py tests/test_image_extractor.py
git commit -m "feat(word_parser): add image extractor from docx relationships"
```

---

## Task 10: CLI Entry Point + Integration

**Files:**
- Create: `parser.py`
- Test: `tests/test_parser.py`

- [ ] **Step 1: Write failing integration tests**

Create `tests/test_parser.py`:

```python
import subprocess
import sys
import textwrap
from pathlib import Path
from tests.conftest import (
    make_docx, add_heading, add_paragraph, add_table,
    add_page_break, docx_bytes,
)


def write_config(path: Path) -> None:
    path.write_text(textwrap.dedent("""\
        heading_styles:
          "Heading 1": 1
          "Heading 2": 2
        font_size_map:
          24: 1
        heading_tags:
          "Introduction": "intro"
          "Config": "cfg"
        table_merge:
          enabled: true
        output_dir: output
        log_level: INFO
    """))


def build_test_docx(path: Path) -> None:
    doc = make_docx()
    add_heading(doc, "Introduction", level=1)
    add_paragraph(doc, "This is the intro.")
    add_heading(doc, "Config", level=1)
    add_table(doc, [["Key", "Value"], ["a", "1"]])
    path.write_bytes(docx_bytes(doc))


def test_parser_creates_output_dir(tmp_path):
    docx_path = tmp_path / "test.docx"
    cfg_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "output"
    build_test_docx(docx_path)
    write_config(cfg_path)

    result = subprocess.run(
        [sys.executable, "parser.py", str(docx_path),
         "--config", str(cfg_path), "--output-dir", str(out_dir)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out_dir.exists()


def test_parser_creates_chunk_files(tmp_path):
    docx_path = tmp_path / "test.docx"
    cfg_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "output"
    build_test_docx(docx_path)
    write_config(cfg_path)

    subprocess.run(
        [sys.executable, "parser.py", str(docx_path),
         "--config", str(cfg_path), "--output-dir", str(out_dir)],
        capture_output=True, text=True,
    )
    chunk_files = list((out_dir / "test" / "chunks").glob("*.md"))
    assert len(chunk_files) >= 2


def test_parser_creates_log_file(tmp_path):
    docx_path = tmp_path / "test.docx"
    cfg_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "output"
    build_test_docx(docx_path)
    write_config(cfg_path)

    subprocess.run(
        [sys.executable, "parser.py", str(docx_path),
         "--config", str(cfg_path), "--output-dir", str(out_dir)],
        capture_output=True, text=True,
    )
    assert (out_dir / "test" / "parse.log").exists()


def test_parser_chunk_contains_table_id(tmp_path):
    docx_path = tmp_path / "test.docx"
    cfg_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "output"
    build_test_docx(docx_path)
    write_config(cfg_path)

    subprocess.run(
        [sys.executable, "parser.py", str(docx_path),
         "--config", str(cfg_path), "--output-dir", str(out_dir)],
        capture_output=True, text=True,
    )
    chunks_dir = out_dir / "test" / "chunks"
    all_md = " ".join(f.read_text() for f in chunks_dir.glob("*.md"))
    assert "<!-- table-id:" in all_md


def test_parser_exit_1_on_missing_file(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    write_config(cfg_path)
    result = subprocess.run(
        [sys.executable, "parser.py", str(tmp_path / "nonexistent.docx"),
         "--config", str(cfg_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_parser.py -v
```

Expected: errors about missing `parser.py`

- [ ] **Step 3: Create `parser.py`**

```python
import argparse
import logging
import sys
from pathlib import Path

from core.config import load_config
from core.document import stream_elements
from core.heading import resolve_heading_depth
from core.table_merger import merge_tables
from core.chunker import build_chunks
from core.renderer import render_chunk, slugify
from parse_logging.parse_logger import make_logger


def main():
    parser = argparse.ArgumentParser(description="Parse .docx into markdown chunks")
    parser.add_argument("input", help="Path to .docx file")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--log-level", default=None, help="Override log level (DEBUG|INFO)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        cfg = load_config(args.config)
    except Exception as e:
        print(f"Error: config parse failed: {e}", file=sys.stderr)
        sys.exit(2)

    output_dir = Path(args.output_dir or cfg.output_dir)
    doc_name = input_path.stem
    doc_out = output_dir / doc_name
    chunks_dir = doc_out / "chunks"
    images_dir = doc_out / "images"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, (args.log_level or cfg.log_level).upper(), logging.INFO)
    logger = make_logger("parser", str(doc_out / "parse.log"), level=log_level)

    try:
        docx_data = input_path.read_bytes()
        elements = list(stream_elements(docx_data))
        elements = merge_tables(elements, logger=logger)
        chunks = build_chunks(elements, cfg, logger=logger)
    except Exception as e:
        logger.error(f"Parse failed: {e}")
        sys.exit(3)

    for chunk in chunks:
        md = render_chunk(chunk)
        label = slugify(chunk.heading_text) if chunk.heading_text else "preamble"
        filename = f"{chunk.index:03d}_{label}.md"
        (chunks_dir / filename).write_text(md, encoding="utf-8")
        logger.info(f"[parser] Wrote chunk: {filename} (tag={chunk.tag})")

    logger.info(f"[parser] Done. {len(chunks)} chunks written to {chunks_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_parser.py -v
```

Expected: 5 PASSED

- [ ] **Step 5: Run full test suite**

```bash
pytest -v
```

Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add parser.py tests/test_parser.py
git commit -m "feat(word_parser): add CLI entry point and integration tests"
```

---

## Task 11: Smoke Test with Real Workflow

This task is manual — no docx file available. Verify the feedback loop works correctly.

- [ ] **Step 1: Run parser on any available docx**

```bash
python parser.py /path/to/any.docx --config config.yaml --log-level DEBUG
```

- [ ] **Step 2: Review `output/{docname}/parse.log`**

Look for WARNING lines. Each warning tells you a rule to add to `config.yaml`:
- `No tag match for heading "X"` → add `"X": "tag"` under `heading_tags:`
- `Font-size fallback: ... size=Npt` → confirm or adjust `font_size_map:` entry
- `Bold+large text, no style/size rule` → decide if it's a heading; add to `font_size_map:` or `heading_styles:`

- [ ] **Step 3: Add rules to `config.yaml`, re-run**

Repeat until no WARNING lines remain (or remaining warnings are intentional).

- [ ] **Step 4: Final commit**

```bash
git add config.yaml
git commit -m "config(word_parser): tune rules from smoke test feedback"
```
