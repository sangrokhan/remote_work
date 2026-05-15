import logging
from core.models import ParagraphElement
from core.config import Config


def resolve_heading_depth(
    para: ParagraphElement,
    cfg: Config,
    logger: logging.Logger | None,
) -> int | None:
    # Style-first
    if para.style_name in cfg.heading_styles:
        depth = cfg.heading_styles[para.style_name]
        if logger:
            logger.debug(
                f"[heading] Style match: {para.text!r} style={para.style_name!r} → depth={depth} (page≈{para.page_approx})"
            )
        return depth

    if logger and para.style_name not in cfg.body_styles:
        logger.debug(f"[heading] No style match: {para.text[:60]!r} style={para.style_name!r}")

    if para.style_name in cfg.body_styles:
        return None

    # Font-size fallback: use largest font size across runs
    sizes = [r.font_size for r in para.runs if r.font_size is not None]
    if sizes:
        max_size = max(sizes)
        size_key = int(max_size)
        if size_key in cfg.font_size_map:
            if logger:
                logger.debug(
                    f"[heading] Font-size fallback: {para.text!r} size={max_size}pt → depth={cfg.font_size_map[size_key]} (page≈{para.page_approx})"
                )
            return cfg.font_size_map[size_key]

        # Bold + large text with no matching rule
        if any(r.bold for r in para.runs):
            if logger:
                logger.warning(
                    f"[heading] Bold+large text, no style/size rule: {para.text!r} bold=True size={max_size}pt (page≈{para.page_approx})"
                )

    return None
