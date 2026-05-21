import re
from core.models import ParagraphElement, TableElement, ImageElement, Chunk

_EXT_MAP = {
    "image/x-emf": "emf",
    "image/emf": "emf",
    "image/x-wmf": "wmf",
    "image/wmf": "wmf",
    "image/svg+xml": "svg",
}


def _image_ext(content_type: str) -> str:
    return _EXT_MAP.get(content_type, content_type.split("/")[-1])


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _render_table(tbl: TableElement, filename_stem: str, counter: int) -> str:
    lines = [f"[{filename_stem}.md - Table {counter}]"]
    if not tbl.rows:
        return "\n".join(lines)

    def _cell(text: str) -> str:
        return text.replace("\n", "<br>")

    header = tbl.rows[0]
    lines.append("| " + " | ".join(_cell(c) for c in header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in tbl.rows[1:]:
        padded = row + [""] * max(0, len(header) - len(row))
        lines.append("| " + " | ".join(_cell(c) for c in padded[: len(header)]) + " |")
    return "\n".join(lines)


def render_chunk(chunk: Chunk, filename_stem: str) -> tuple[str, str]:
    """Return (content_md, table_md). content_md has heading+paragraphs+image refs.
    table_md has tables only (empty string if none)."""
    content_parts: list[str] = []
    table_parts: list[str] = []
    table_counter = 0
    image_counter = 0

    # Prepend parent folder headings (depth increases by 1 per level)
    for i, heading_text in enumerate(chunk.folder_headings):
        prefix = "#" * (i + 1)
        content_parts.append(f"{prefix} {heading_text}")

    if chunk.heading_text:
        depth = len(chunk.folder_headings) + 1 if chunk.folder_headings else chunk.heading_depth
        prefix = "#" * depth
        content_parts.append(f"{prefix} {chunk.heading_text}")

    for elem in chunk.elements:
        if isinstance(elem, ParagraphElement):
            if elem.is_page_break or not elem.text.strip():
                continue
            if elem.heading_depth is not None:
                prefix = "#" * elem.heading_depth
                content_parts.append(f"{prefix} {elem.text}")
            else:
                content_parts.append(elem.text)
        elif isinstance(elem, TableElement):
            table_counter += 1
            content_parts.append(f"[{filename_stem}.md - Table {table_counter}]")
            table_parts.append(_render_table(elem, filename_stem, table_counter))
        elif isinstance(elem, ImageElement):
            image_counter += 1
            ext = _image_ext(elem.content_type)
            stem = slugify(elem.caption) if elem.caption else f"img_{image_counter}"
            name = f"{stem}.{ext}"
            content_parts.append(f"![{name}](../images/{filename_stem}/{name})")

    return "\n\n".join(content_parts), "\n\n".join(table_parts)
