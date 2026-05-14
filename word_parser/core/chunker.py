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
