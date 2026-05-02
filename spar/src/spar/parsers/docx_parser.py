from __future__ import annotations

import csv as _csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import docx
import docx.table
import docx.text.paragraph

from spar.parsers.docx_config import DocxParseConfig


def _slugify(text: str, max_len: int) -> str:
    if not text.strip():
        return "unnamed"
    slug = re.sub(r"[^\w\s.\-]", "", text)
    slug = re.sub(r"[\s_]+", "-", slug).strip("-")
    return slug[:max_len] if slug else "unnamed"


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


def child_blips(para) -> list[str]:
    """Return relationship IDs for embedded images in a paragraph."""
    nsmap = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    r_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    blips = para._element.findall(".//a:blip", nsmap)
    return [b.get(f"{{{r_ns}}}embed") for b in blips if b.get(f"{{{r_ns}}}embed")]


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
                    lines, extracted_images, images_dir, doc, docx_path,
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
        docx_path: Path,
    ) -> None:
        style_name: str = para.style.name if para.style else ""

        if style_name.startswith("Heading"):
            try:
                level = int(style_name.split()[-1])
            except (ValueError, IndexError):
                level = 1
            title = para.text.strip()
            if level <= self._cfg.heading_depth:
                if level == 1:
                    section_path.clear()
                    table_seq.clear()
                    image_seq.clear()
                else:
                    while len(section_path) >= level:
                        section_path.pop()
                section_path.append(title)
            lines.append(f"{'#' * level} {title}")
            return

        blips = child_blips(para)
        if blips:
            for rel_id in blips:
                self._save_image(
                    rel_id, doc, section_path, image_seq,
                    extracted_images, images_dir, lines, docx_path,
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

        with csv_path.open("w", encoding="utf-8", newline="") as f:
            f.write(f"# section: {self._section_key(section_path)}\n")
            f.write(f"# source: {source_path.name}\n")
            writer = _csv.writer(f)
            for row in table.rows:
                writer.writerow([cell.text.strip() for cell in row.cells])

        placeholder = f"<!-- TABLE: {filename[:-4]} -->"
        lines.append(placeholder)
        extracted_tables.append(
            ExtractedTable(path=csv_path, section_path=list(section_path), seq=seq)
        )

    def _save_image(
        self,
        rel_id: str,
        doc,
        section_path: list[str],
        image_seq: dict[str, int],
        extracted_images: list[ExtractedImage],
        images_dir: Path,
        lines: list[str],
        docx_path: Path,
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
            f"section: {self._section_key(section_path)}\nseq: {seq}\n",
            encoding="utf-8",
        )
        lines.append(f"<!-- IMAGE: {filename} -->")
        extracted_images.append(
            ExtractedImage(
                path=img_path,
                section_path=list(section_path),
                seq=seq,
                ext=ext,
            )
        )
