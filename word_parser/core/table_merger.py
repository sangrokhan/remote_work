import logging
from core.models import ParagraphElement, TableElement, ImageElement

Element = ParagraphElement | TableElement | ImageElement


def merge_tables(elements: list[Element], logger: logging.Logger | None) -> list[Element]:
    result: list[Element] = []
    i = 0
    while i < len(elements):
        elem = elements[i]
        if not isinstance(elem, TableElement):
            result.append(elem)
            i += 1
            continue

        # Collect page-break-only paragraphs after this table
        j = i + 1
        gap: list[ParagraphElement] = []
        while j < len(elements):
            nxt = elements[j]
            if isinstance(nxt, ParagraphElement) and nxt.is_page_break:
                gap.append(nxt)
                j += 1
            else:
                break

        # Check if next element is a table with same col count
        if (
            gap
            and j < len(elements)
            and isinstance(elements[j], TableElement)
            and elements[j].col_count == elem.col_count
        ):
            next_tbl = elements[j]
            merged_rows = list(elem.rows)

            # Drop repeated header row
            continuation = list(next_tbl.rows)
            if continuation and continuation[0] == merged_rows[0]:
                if logger:
                    logger.debug(
                        f"[table_merger] Repeated header row dropped: {continuation[0]} (page≈{next_tbl.page_approx})"
                    )
                continuation = continuation[1:]

            merged_rows.extend(continuation)

            if logger:
                logger.info(
                    f"[table_merger] Merged table at page≈{next_tbl.page_approx} (col={elem.col_count})"
                )

            merged = TableElement(
                rows=merged_rows,
                col_count=elem.col_count,
                page_approx=elem.page_approx,
                preceded_by_page_break=elem.preceded_by_page_break,
            )
            result.append(merged)
            i = j + 1
        else:
            result.append(elem)
            result.extend(gap)
            i = j

    return result
