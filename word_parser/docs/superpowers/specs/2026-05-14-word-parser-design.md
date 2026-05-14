# Word Document Parser — Design Spec

**Date:** 2026-05-14  
**Project:** `word_parser/`  
**Status:** Approved

---

## 1. Problem

Parse confidential Word (`.docx`) documents into document-aware markdown chunks with extracted images. Rules (heading levels, table tags) are provided iteratively via config — not hardcoded — because the real document cannot be shared. Logging fills the feedback loop: unknown conditions are surfaced with enough context to write the missing rule.

---

## 2. Goals

- Convert `.docx` → one `.md` file per heading-bounded chunk
- Extract embedded images, saved alongside chunks
- Map heading styles → markdown `#` depth (style-primary, font-size fallback)
- Detect and merge tables split across page breaks into a single table
- Assign unique IDs to tables via user-provided heading→tag config
- Log all unknown/ambiguous conditions to both console and file

---

## 3. Non-Goals

- No GUI or web interface
- No support for `.doc` (legacy binary format)
- No multi-document batch mode (single file per run, v1)
- No OCR for image-embedded text

---

## 4. Architecture

```
word_parser/
├── config.yaml              # all rules: heading styles, font sizes, heading→tag map
├── parser.py                # CLI entry point
├── core/
│   ├── document.py          # loads .docx, emits ordered element stream
│   ├── heading.py           # style-first + font-size fallback → heading depth
│   ├── table_merger.py      # detects & merges page-break-split tables
│   ├── chunker.py           # groups elements into heading-bounded chunks
│   ├── image_extractor.py   # extracts embedded images, assigns filenames
│   └── renderer.py          # chunk → markdown string
├── logging/
│   └── parse_logger.py      # dual console+file logger
└── output/                  # generated output (gitignored)
    └── {docname}/
        ├── chunks/
        │   ├── 001_heading_text.md
        │   └── ...
        ├── images/
        │   ├── {tag}_table_1.png
        │   └── ...
        └── parse.log
```

### Data Flow

```
.docx
  └─► document.py      (element stream: paragraphs, tables, images, breaks)
        └─► heading.py         (resolve heading level per paragraph)
        └─► table_merger.py    (merge split tables in stream)
        └─► chunker.py         (group into chunks by heading)
              └─► image_extractor.py  (extract images per chunk)
              └─► renderer.py         (emit .md per chunk)
```

---

## 5. Config Schema

`config.yaml`:

```yaml
# Heading style name → markdown depth
heading_styles:
  "Heading 1": 1
  "Heading 2": 2
  "Heading 3": 3

# Font size (pt, integer) → markdown depth
# Used as fallback when paragraph style is not in heading_styles
font_size_map:
  24: 1
  18: 2
  14: 3

# Heading text → tag (substring match, first match wins)
# Tables under matched heading get IDs: {tag}_table_1, {tag}_table_2, ...
heading_tags:
  "3.2 Configuration": "config"
  "4.1 Parameters":    "params"

# Table merge settings
table_merge:
  enabled: true
  # Merge condition: same column count AND only page break between tables

# Output
output_dir: "output"
log_level: "INFO"   # INFO | DEBUG
```

**Tag resolution rules:**
- Chunker resolves active tag when entering a heading paragraph
- Match: first `heading_tags` key that is a substring of the heading text (case-insensitive)
- No match → tag = `unknown`, log WARNING with heading text
- Tag resets when entering a new heading at same or higher level

**Image filenames:**
- Table images: `{tag}_table_{N}.{ext}` (N = 1-indexed within chunk)
- Inline images: `{tag}_img_{N}.{ext}`
- No tag context: `unnamed_img_{N}.{ext}`

---

## 6. Component Specs

### 6.1 `document.py` — Element Stream

Walks `docx.Document` body in document order. Emits typed elements:

```python
@dataclass
class ParagraphElement:
    text: str
    style_name: str
    runs: list[Run]       # for font size inspection
    page_approx: int      # estimated page number via page break counting

@dataclass
class TableElement:
    rows: list[list[str]]
    col_count: int
    page_approx: int
    preceded_by_page_break: bool

@dataclass
class ImageElement:
    relationship_id: str
    content_type: str     # e.g. "image/png"
    data: bytes
    page_approx: int
```

Page number is approximate — counted by accumulating `w:lastRenderedPageBreak` and `w:br type="page"` elements.

### 6.2 `heading.py` — Heading Level Resolution

```
Input:  ParagraphElement
Output: int (depth 1–6) | None

1. If style_name in heading_styles → return mapped depth
2. Else if any run font size in font_size_map → return mapped depth (log DEBUG)
3. If bold=True AND font_size not in map → log WARNING (bold+large text, no rule)
4. Return None (not a heading)
```

### 6.3 `table_merger.py` — Split Table Detection

Processes `TableElement` stream. Merge condition (both required):
- Previous element was `TableElement` with same `col_count`
- Only a page break (no paragraphs, no images) between them

On merge:
- If second table's first row matches first table's header row → drop it (log DEBUG)
- Append remaining rows to first table
- Log INFO with tag, col count, page approx

### 6.4 `chunker.py` — Chunk Grouping

- New chunk starts at each heading paragraph
- Chunk inherits heading text, depth, resolved tag
- Elements accumulate until next heading at same or higher depth
- Empty chunks (heading + no body) logged at DEBUG

### 6.5 `image_extractor.py` — Image Extraction

- Extracts image bytes from `.docx` relationship store
- Names by context: table image vs inline image, tag, index
- Unrecognized content type logged at WARNING
- Saves to `output/{docname}/images/`

### 6.6 `renderer.py` — Markdown Output

- Heading → `#{depth} {text}`
- Paragraph → plain text (preserves line breaks)
- Table → GFM pipe table with header row
  - Table ID as HTML comment anchor: `<!-- table-id: {tag}_table_{N} -->`
- Image → `![{name}](../images/{filename})`
- One `.md` file per chunk, zero-padded index prefix + slugified heading text: `001_configuration.md`, `002_parameters.md`, ... (slug = lowercase, spaces→underscores, non-alphanumeric stripped)

---

## 7. Logging

Dual output: console (stderr) + `output/{docname}/parse.log`.

### Log Events

| Condition | Level | Fields logged |
|-----------|-------|---------------|
| Style not in `heading_styles`, font size not in `font_size_map` | WARNING | text, style_name, font_size, page_approx |
| Font-size fallback used | DEBUG | text, font_size, resolved_depth |
| Bold+large text, no style/size match | WARNING | text, bold=True, font_size, page_approx |
| No tag match for heading | WARNING | heading_text, table_index, col_count, page_approx |
| Table merged across page break | INFO | tag, col_count, page_approx |
| Repeated header row dropped on merge | DEBUG | tag, row_content |
| Unrecognized image content type | WARNING | content_type, page_approx |
| Empty chunk | DEBUG | heading_text, depth |
| Unhandled XML element type | DEBUG | xml_tag, page_approx |

### Log Format

```
2026-05-14 10:23:01 WARNING  [table_merger] No tag for heading "5.1 Overview" → using unknown_table_1 (col=4, page≈12)
2026-05-14 10:23:01 DEBUG    [heading]      Font-size fallback: "Introduction" size=18pt → depth=2
2026-05-14 10:23:01 INFO     [table_merger] Merged table at page≈7 (col=6, tag=config)
```

---

## 8. CLI Interface

```bash
python parser.py <input.docx> [--config config.yaml] [--output-dir output] [--log-level DEBUG]
```

Exit codes:
- `0` — success
- `1` — input file not found
- `2` — config parse error
- `3` — unrecoverable parse error

---

## 9. Feedback Loop

Intended workflow:
1. Run parser on document
2. Review `parse.log` — WARNING entries indicate missing config rules
3. Add rules to `config.yaml` (heading styles, font sizes, heading tags)
4. Re-run until warnings resolve
5. Repeat for each new document type

---

## 10. Dependencies

```
python-docx >= 1.1
PyYAML >= 6.0
```

Python 3.11+. No other runtime dependencies.
