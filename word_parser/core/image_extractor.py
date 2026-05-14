import io
import logging
from docx import Document
from core.models import ImageElement

KNOWN_IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/bmp", "image/tiff"}


def extract_images_from_docx(
    docx_data: bytes,
    logger: logging.Logger | None = None,
) -> list[ImageElement]:
    doc = Document(io.BytesIO(docx_data))
    images = []

    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            content_type = rel.target_part.content_type
            if content_type not in KNOWN_IMAGE_TYPES:
                if logger:
                    logger.warning(
                        f"[image_extractor] Unrecognized image content type: {content_type}"
                    )
                continue
            images.append(
                ImageElement(
                    relationship_id=rel.rId,
                    content_type=content_type,
                    data=rel.target_part.blob,
                    page_approx=0,
                )
            )

    return images
