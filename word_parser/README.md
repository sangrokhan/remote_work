# word_parser

Parse `.docx` files into heading-bounded markdown chunks with extracted images.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
python parser.py <input.docx> [--config config.yaml] [--output-dir output] [--log-level DEBUG]
```

**Examples:**

```bash
# Basic run
python parser.py my_doc.docx

# Custom config and debug logging
python parser.py my_doc.docx --config config.yaml --log-level DEBUG

# Custom output directory
python parser.py my_doc.docx --output-dir /tmp/parsed
```

**Exit codes:** `0` success · `1` file not found · `2` config error · `3` parse error

## Output

```
output/{docname}/
├── chunks/
│   ├── 001_introduction.md
│   ├── 002_configuration.md
│   └── ...
├── images/
│   ├── config_table_1.png
│   └── ...
└── parse.log
```

Each chunk file covers one heading section. Images are referenced as `../images/{name}` in the markdown.

## Config (`config.yaml`)

```yaml
# Heading style name → markdown depth (#, ##, ###, ...)
heading_styles:
  "Heading 1": 1
  "Heading 2": 2
  "Heading 3": 3

# Font size (pt) → markdown depth (fallback when style is not in heading_styles)
font_size_map:
  24: 1
  18: 2
  14: 3

# Heading text substring → tag (first match wins, case-insensitive)
# Tables under that heading are named: {tag}_table_1, {tag}_table_2, ...
heading_tags:
  "3.2 Configuration": "config"
  "4.1 Parameters":    "params"

table_merge:
  enabled: true   # merge tables split across page breaks

output_dir: "output"
log_level: "INFO"   # INFO | DEBUG
```

## Iterative Tuning Workflow

1. Run parser on document
2. Review `output/{docname}/parse.log` — `WARNING` entries show missing rules
3. Add rules to `config.yaml` (heading styles, font sizes, heading tags)
4. Re-run until warnings resolve

**Common warnings and fixes:**

| Warning | Fix |
|---------|-----|
| `Style not in heading_styles` | Add the style name to `heading_styles` |
| `Bold+large text, no rule` | Add the font size to `font_size_map` |
| `No tag for heading` | Add a substring match to `heading_tags` |

## Tests

```bash
pytest tests/
```
