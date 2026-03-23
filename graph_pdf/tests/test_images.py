from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from extractor.images import _crop_page_region, _extract_embedded_images


class CropPageRegionTests(unittest.TestCase):
    def test_crop_page_region_uses_pdfplumber_top_coordinates(self) -> None:
        source = Image.new("RGB", (100, 100))
        for y in range(100):
            for x in range(100):
                source.putpixel((x, y), (y, y, y))
        page_image = SimpleNamespace(original=source)

        cropped = _crop_page_region(
            page_image=page_image,
            page_height=100.0,
            bbox=(20.0, 10.0, 80.0, 30.0),
            resolution=72.0,
        )

        self.assertEqual((60, 20), cropped.size)
        self.assertEqual((10, 10, 10), cropped.getpixel((0, 0)))
        self.assertEqual((29, 29, 29), cropped.getpixel((0, 19)))

    def test_extract_embedded_images_removes_watermark_from_drawing_crop(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        pdf_path = root / "watermark_only.pdf"

        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        c.saveState()
        c.setFillColor(colors.Color(0.92, 0.92, 0.92))
        c.setFont("Helvetica-Bold", 44)
        c.translate(letter[0] / 2.0, letter[1] / 2.0)
        c.rotate(55)
        c.drawCentredString(0, 0, "CONFIDENTIAL")
        c.restoreState()
        c.save()

        image_files = _extract_embedded_images(
            pdf_path=pdf_path,
            out_image_dir=root / "images",
            stem="watermark_only",
            drawing_regions_by_page={1: [(0.0, 0.0, float(letter[0]), float(letter[1]))]},
        )

        self.assertEqual(1, len(image_files))
        image = Image.open(image_files[0]).convert("L")
        self.assertEqual((255, 255), image.getextrema())
