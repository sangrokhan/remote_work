import argparse
import logging
import sys
import tempfile
from pathlib import Path

from core.config import load_config
from core.document import stream_elements, attach_captions
from core.models import ImageElement
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
    doc_out.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, (args.log_level or cfg.log_level).upper(), logging.INFO)
    logger = make_logger("parser", str(doc_out / "parse.log"), level=log_level)

    try:
        if args.use_word_com:
            logger.info("Opening via Word COM automation")
            docx_data = _read_bytes_via_word(input_path)
        else:
            docx_data = input_path.read_bytes()
        elements = list(stream_elements(docx_data, logger=logger))
        from core.models import TableElement as _TE, ParagraphElement as _PE
        n_paras = sum(1 for e in elements if isinstance(e, _PE))
        n_tables = sum(1 for e in elements if isinstance(e, _TE))
        n_images = sum(1 for e in elements if isinstance(e, ImageElement))
        pages = max((e.page_approx for e in elements), default=1)
        logger.info(
            f"[document] Streamed {len(elements)} elements "
            f"({n_paras} paragraphs, {n_tables} tables, {n_images} images, ~{pages} pages)"
        )
        elements = merge_tables(elements, logger=logger)
        elements = attach_captions(elements)
        chunks = build_chunks(elements, cfg, logger=logger)
    except Exception as e:
        logger.error(f"Parse failed: {e}")
        sys.exit(3)

    for chunk in chunks:
        chunk_slug = slugify(chunk.heading_text) if chunk.heading_text else "preamble"
        filename_stem = f"{chunk.folder_index:03d}_{chunk_slug}"

        # Build folder path: doc_out / folder_slug_1 / folder_slug_2 / ...
        folder_path = doc_out
        for slug in chunk.folder_slugs:
            folder_path = folder_path / slug

        md_dir = folder_path / "md"
        md_dir.mkdir(parents=True, exist_ok=True)

        content_md, table_md = render_chunk(chunk)
        (md_dir / f"{filename_stem}.md").write_text(content_md, encoding="utf-8")
        if table_md.strip():
            (md_dir / f"{filename_stem}_table.md").write_text(table_md, encoding="utf-8")

        # Save inline images
        image_counter = 0
        for elem in chunk.elements:
            if isinstance(elem, ImageElement):
                image_counter += 1
                ext = elem.content_type.split("/")[-1]
                stem = slugify(elem.caption) if elem.caption else f"{chunk_slug}_img_{image_counter}"
                img_name = f"{stem}.{ext}"
                images_dir = folder_path / "images" / chunk_slug
                images_dir.mkdir(parents=True, exist_ok=True)
                (images_dir / img_name).write_bytes(elem.data)

        logger.info(f"Wrote chunk: {filename_stem}.md (folder={'/'.join(chunk.folder_slugs) or '.'})")

    logger.info(f"Done. {len(chunks)} chunks written to {doc_out}")


if __name__ == "__main__":
    main()
