# Note/Table Root Cause Debugging Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Identify the root cause of note regions being emitted as tables in sample dump outputs before changing parser behavior.

**Architecture:** Use structure-first debugging on the existing parser flow. First collect page-level evidence from current outputs and region logs for `raw-38-50.dump` and `raw-93-114.dump`, then compare against `samples/*_table.md` and `samples/gold/*` to isolate whether the failure is caused by note ownership loss, table continuation logic, or output formatting only. No production fix should be attempted until a single root-cause hypothesis is supported by evidence and captured by failing tests.

**Tech Stack:** Python, `unittest`, existing parser debug outputs (`region_log`), sample dump fixtures.

---

### Task 1: Lock the Reproduction

**Files:**
- Use: `samples/raw-38-50.dump`
- Use: `samples/raw-93-114.dump`
- Use: `samples/raw-38-50_table.md`
- Use: `samples/raw-93-114_table.md`
- Use: `samples/gold/raw-38-50/md/FGR-BC0201.md`
- Use: `samples/gold/raw-93-114/md/FGR-BC0401.md`

- [ ] **Step 1: Run sample reproduction without debug page markers**

Run:
```bash
python3 -m unittest tests.test_samples
```

Expected:
- `raw-38-50.dump` fails because current parser emits 5 tables instead of sample's 4
- `raw-93-114.dump` fails because current parser emits 27 tables instead of sample's 22

- [ ] **Step 2: Confirm sample reference counts**

Run:
```bash
python3 - <<'PY'
from pathlib import Path
import re
for stem in ["raw-38-50", "raw-93-114"]:
    text = Path(f"samples/{stem}_table.md").read_text(encoding="utf-8")
    print(stem, len(re.findall(r"^\[[^\]]+_tables\.md - Table \d+\]$", text, flags=re.M)))
PY
```

Expected:
- `raw-38-50 4`
- `raw-93-114 22`

### Task 2: Collect Region Evidence

**Files:**
- Use: `extractor/pipeline.py`
- Use: `extractor/notes.py`
- Use: `extractor/tables.py`
- Create temp diagnostics under: `artifacts/manual/debug-note-table/`

- [ ] **Step 1: Generate region logs for the failing samples**

Run:
```bash
python3 - <<'PY'
from pathlib import Path
from extractor.pipeline import extract_pdf_to_outputs
base = Path("artifacts/manual/debug-note-table")
for raw in [Path("samples/raw-38-50.dump"), Path("samples/raw-93-114.dump")]:
    root = base / raw.stem
    extract_pdf_to_outputs(
        pdf_path=None,
        from_raw=raw,
        out_md_dir=root / "md",
        out_image_dir=root / "images",
        stem=raw.stem,
        region_log=root / "region_log.json",
    )
    print(root / "region_log.json")
PY
```

Expected:
- Region logs exist for both failing samples

- [ ] **Step 2: Compare note/table bbox overlap around extra tables**

Run:
```bash
python3 - <<'PY'
from pathlib import Path
import json
for stem in ["raw-38-50", "raw-93-114"]:
    region_log = Path(f"artifacts/manual/debug-note-table/{stem}/region_log.json")
    data = json.loads(region_log.read_text(encoding="utf-8"))
    print("==", stem, "==")
    for page_no, page in sorted(data["pages"].items(), key=lambda item: int(item[0])):
        notes = page.get("notes", [])
        tables = page.get("tables", [])
        if notes and tables:
            print("page", page_no, "notes", notes, "tables", tables)
PY
```

Expected:
- Evidence showing whether extra table candidates occupy the same region as detected notes

### Task 3: Form the Hypothesis

**Files:**
- Reference: `docs/prd.md`
- Use: `artifacts/manual/debug-note-table/raw-38-50/region_log.json`
- Use: `artifacts/manual/debug-note-table/raw-93-114/region_log.json`

- [ ] **Step 1: Write one explicit hypothesis**

Template:
```text
I think the extra table output is caused by <specific mechanism> because <specific evidence from region logs and current table markdown>.
```

- [ ] **Step 2: Reject alternatives explicitly**

Check and record:
- Not a page marker / formatting artifact
- Not a table continuation merge issue
- Not an image reference issue
- Caused by note-owned content being emitted as a table candidate, or prove otherwise

### Task 4: Add Failing Tests Before the Fix

**Files:**
- Modify: `tests/test_samples.py`
- Optionally create focused unit tests in `tests/test_pipeline.py`

- [ ] **Step 1: Keep the sample failures as the high-level regression**

Run:
```bash
python3 -m unittest tests.test_samples
```

Expected:
- Current failures remain red until the root cause is fixed

- [ ] **Step 2: Add one focused failing test for note/table ownership**

Behavior to lock:
- If a note region is already detected for a sample page, that same region must not be emitted as a table block

- [ ] **Step 3: Verify the focused test fails for the expected reason**

Run:
```bash
python3 -m unittest tests.test_pipeline -v
```

Expected:
- The new focused test fails because the current implementation still emits the extra table
