import logging
import re
from core.models import ParagraphElement, TableElement, ImageElement, Chunk
from core.config import Config
from core.heading import resolve_heading_depth

Element = ParagraphElement | TableElement | ImageElement


def _slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def build_chunks(
    elements: list[Element],
    cfg: Config,
    logger: logging.Logger | None,
) -> list[Chunk]:
    split_depth = cfg.chunk_split_depth
    chunks: list[Chunk] = []
    current_elements: list[Element] = []
    current_heading = ""
    current_depth = 0
    current_folder_slugs: list[str] = []
    folder_stack: list[tuple[int, str]] = []  # (depth, slug) for depths < split_depth
    index = 0

    def flush():
        nonlocal index
        chunk = Chunk(
            heading_text=current_heading,
            heading_depth=current_depth,
            elements=list(current_elements),
            index=index,
            folder_slugs=list(current_folder_slugs),
        )
        chunks.append(chunk)
        index += 1

    for elem in elements:
        if isinstance(elem, ParagraphElement):
            depth = resolve_heading_depth(elem, cfg, logger)
            if depth is not None:
                if split_depth > 0 and depth < split_depth:
                    # Folder-level heading: flush current chunk, update folder stack
                    if current_elements or current_heading:
                        if not current_elements and logger:
                            logger.debug(
                                f"[chunker] Empty chunk: {current_heading!r} depth={current_depth}"
                            )
                        flush()
                    current_elements = []
                    current_heading = ""
                    current_depth = 0
                    folder_stack = [(d, s) for d, s in folder_stack if d < depth]
                    folder_stack.append((depth, _slugify(elem.text)))
                    current_folder_slugs = [s for _, s in folder_stack]
                    continue

                if split_depth == 0 or depth == split_depth:
                    # File-level heading: start new chunk
                    if current_elements or current_heading:
                        if not current_elements and logger:
                            logger.debug(
                                f"[chunker] Empty chunk: {current_heading!r} depth={current_depth}"
                            )
                        if split_depth > 0 and not current_heading:
                            # Content before first file-level heading — skip, no chunk
                            if logger and current_elements:
                                logger.warning(
                                    f"[chunker] {len(current_elements)} elements before first "
                                    f"depth-{split_depth} heading — skipped (no matching heading)"
                                )
                        else:
                            flush()
                    current_elements = []
                    current_heading = elem.text
                    current_depth = depth
                    continue

                # depth > split_depth: sub-heading, treat as body content

        current_elements.append(elem)

    if current_elements or current_heading:
        if not current_elements and logger:
            logger.debug(f"[chunker] Empty chunk: {current_heading!r} depth={current_depth}")
        if split_depth > 0 and not current_heading and not current_elements:
            pass  # nothing to flush
        elif split_depth > 0 and not current_heading:
            if logger:
                logger.warning(
                    f"[chunker] {len(current_elements)} trailing elements with no "
                    f"depth-{split_depth} heading — skipped"
                )
        else:
            flush()

    return chunks
