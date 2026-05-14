import io
import pytest
from docx import Document
from core.image_extractor import extract_images_from_docx
from core.models import ImageElement
from tests.conftest import make_docx, docx_bytes


def test_no_images_returns_empty():
    doc = make_docx()
    images = extract_images_from_docx(docx_bytes(doc))
    assert images == []


def test_image_elements_have_data():
    doc = make_docx()
    buf = io.BytesIO()
    doc.save(buf)
    images = extract_images_from_docx(buf.getvalue())
    assert isinstance(images, list)


def test_extract_returns_image_elements():
    doc = make_docx()
    images = extract_images_from_docx(docx_bytes(doc))
    for img in images:
        assert isinstance(img, ImageElement)
        assert img.data
        assert img.content_type
