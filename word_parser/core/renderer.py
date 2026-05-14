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
    image_counter = 0

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
            image_counter += 1
            ext = elem.content_type.split("/")[-1]
            name = f"{chunk.tag}_img_{image_counter}.{ext}"
            parts.append(f"![{name}](../images/{name})")

    return "\n\n".join(parts)
