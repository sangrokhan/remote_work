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
    trace: bool = False,
) -> list[Chunk]:
    split_depth = cfg.chunk_split_depth
    chunks: list[Chunk] = []
    current_elements: list[Element] = []
    current_heading = ""
    current_depth = 0
    current_folder_slugs: list[str] = []
    folder_stack: list[tuple[int, str, str]] = []  # (depth, slug, heading_text) for depths < split_depth
    folder_counters: dict[int, int] = {}      # global counter per depth level
    file_counter: int = 0                     # global counter for all file-level chunks
    index = 0

    def _tr(msg: str) -> None:
        if trace:
            print(msg, flush=True)

    def flush():
        nonlocal index, file_counter
        file_counter += 1
        effective_folder_slugs = list(current_folder_slugs) or (["000"] if split_depth > 0 else [])
        effective_folder_headings = [t for _, _, t in folder_stack] or []
        chunk = Chunk(
            heading_text=current_heading,
            heading_depth=current_depth,
            elements=list(current_elements),
            index=index,
            folder_slugs=effective_folder_slugs,
            folder_headings=effective_folder_headings,
            folder_index=file_counter,
        )
        chunks.append(chunk)
        index += 1

    def flush_cover():
        nonlocal index
        cover_folder_slugs = list(current_folder_slugs) or ["000"]
        cover_folder_headings = [t for _, _, t in folder_stack] or []
        chunk = Chunk(
            heading_text="",
            heading_depth=0,
            elements=list(current_elements),
            index=index,
            folder_slugs=cover_folder_slugs,
            folder_headings=cover_folder_headings,
            folder_index=0,
        )
        chunks.append(chunk)
        index += 1

    for elem in elements:
        if isinstance(elem, ParagraphElement):
            depth = resolve_heading_depth(elem, cfg, logger)
            text_preview = elem.text[:70].replace("\n", " ") if elem.text else ""
            depth_label = f"depth={depth}" if depth is not None else "depth=--"

            if depth is not None:
                if split_depth > 0 and depth < split_depth:
                    # Folder-level heading: flush current chunk, update folder stack
                    _tr(
                        f"[TRACE] p={elem.page_approx:4d}  para  {elem.style_name!r:30s}  {depth_label:9s}"
                        f"  → FOLDER  {text_preview!r}"
                    )
                    if current_elements or current_heading:
                        if not current_elements and logger:
                            logger.debug(
                                f"[chunker] Empty chunk: {current_heading!r} depth={current_depth}"
                            )
                        flush()
                    current_elements = []
                    current_heading = ""
                    current_depth = 0
                    folder_stack = [(d, s, t) for d, s, t in folder_stack if d < depth]
                    folder_counters[depth] = folder_counters.get(depth, 0) + 1
                    numbered_slug = f"{folder_counters[depth]:03d}"
                    folder_stack.append((depth, numbered_slug, elem.text))
                    current_folder_slugs = [s for _, s, _ in folder_stack]
                    continue

                if split_depth == 0 or depth == split_depth:
                    # File-level heading: start new chunk
                    _tr(
                        f"[TRACE] p={elem.page_approx:4d}  para  {elem.style_name!r:30s}  {depth_label:9s}"
                        f"  → NEW CHUNK[{index}]  {text_preview!r}"
                    )
                    if current_elements or current_heading:
                        if not current_elements and logger:
                            logger.debug(
                                f"[chunker] Empty chunk: {current_heading!r} depth={current_depth}"
                            )
                        if split_depth > 0 and not current_heading:
                            if current_elements:
                                if logger:
                                    logger.info(
                                        f"[chunker] {len(current_elements)} elements before first "
                                        f"depth-{split_depth} heading — emitted as cover (000)"
                                    )
                                flush_cover()
                        else:
                            flush()
                    current_elements = []
                    current_heading = elem.text
                    current_depth = depth
                    continue

                # depth > split_depth: sub-heading, treat as body content
                else:
                    _tr(
                        f"[TRACE] p={elem.page_approx:4d}  para  {elem.style_name!r:30s}  {depth_label:9s}"
                        f"  → sub-heading in chunk[{index}]  {text_preview!r}"
                    )
                    elem.heading_depth = depth
            else:
                _tr(
                    f"[TRACE] p={elem.page_approx:4d}  para  {elem.style_name!r:30s}  {depth_label:9s}"
                    f"  → body of chunk[{index}]  {text_preview!r}"
                )

        elif isinstance(elem, TableElement):
            _tr(
                f"[TRACE] p={elem.page_approx:4d}  table rows={len(elem.rows):3d} cols={elem.col_count:2d}"
                f"                                   → body of chunk[{index}]"
            )

        elif isinstance(elem, ImageElement):
            _tr(
                f"[TRACE] p={elem.page_approx:4d}  image rId={elem.relationship_id!r:15s}"
                f"  type={elem.content_type!r:20s}              → body of chunk[{index}]"
            )

        current_elements.append(elem)

    if current_elements or current_heading:
        if not current_elements and logger:
            logger.debug(f"[chunker] Empty chunk: {current_heading!r} depth={current_depth}")
        if split_depth > 0 and not current_heading and not current_elements:
            pass  # nothing to flush
        elif split_depth > 0 and not current_heading:
            if current_elements:
                if logger:
                    logger.info(
                        f"[chunker] {len(current_elements)} elements with no "
                        f"depth-{split_depth} heading — emitted as cover (000)"
                    )
                flush_cover()
        else:
            flush()

    return chunks
