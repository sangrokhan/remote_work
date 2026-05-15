import argparse
import logging
import sys
import tempfile
from pathlib import Path

from core.config import load_config
from core.document import stream_elements
from core.table_merger import merge_tables
from core.chunker import build_chunks
from core.renderer import render_chunk, slugify
from parse_logging.parse_logger import make_logger


def _read_bytes_via_word(input_path: Path) -> bytes:
    """Open DRM-protected .docx via Word COM, save decrypted copy, return bytes."""
    try:
        import win32com.client
    except ImportError:
        raise RuntimeError("pywin32 not installed. Run: pip install pywin32")

    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    try:
        doc = word.Documents.Open(str(input_path.resolve()))
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp_path = tmp.name
        doc.SaveAs2(tmp_path, FileFormat=16)  # 16 = wdFormatXMLDocument
        doc.Close(False)
        return Path(tmp_path).read_bytes()
    finally:
        word.Quit()


def main():
    parser = argparse.ArgumentParser(description="Parse .docx into markdown chunks")
    parser.add_argument("input", help="Path to .docx file")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--log-level", default=None, help="Override log level (DEBUG|INFO)")
    parser.add_argument(
        "--use-word-com",
        action="store_true",
        help="Open via Word COM automation (Windows only, required for DRM-protected files)",
    )
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
    images_dir.mkdir(parents=True, exist_ok=True)  # reserved for extracted images (wired in future iteration)

    log_level = getattr(logging, (args.log_level or cfg.log_level).upper(), logging.INFO)
    logger = make_logger("parser", str(doc_out / "parse.log"), level=log_level)

    try:
        if args.use_word_com:
            logger.info("Opening via Word COM automation")
            docx_data = _read_bytes_via_word(input_path)
        else:
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
        logger.info(f"Wrote chunk: {filename} (tag={chunk.tag})")

    logger.info(f"Done. {len(chunks)} chunks written to {chunks_dir}")


if __name__ == "__main__":
    main()
