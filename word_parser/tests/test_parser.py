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
        cwd="/home/han/.openclaw/workspace/remote_work/.claude/worktrees/word-parser-impl/word_parser",
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
        cwd="/home/han/.openclaw/workspace/remote_work/.claude/worktrees/word-parser-impl/word_parser",
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
        cwd="/home/han/.openclaw/workspace/remote_work/.claude/worktrees/word-parser-impl/word_parser",
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
        cwd="/home/han/.openclaw/workspace/remote_work/.claude/worktrees/word-parser-impl/word_parser",
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
        cwd="/home/han/.openclaw/workspace/remote_work/.claude/worktrees/word-parser-impl/word_parser",
    )
    assert result.returncode == 1
