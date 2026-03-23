from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageChops
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from extractor.images import _crop_page_region, _extract_embedded_images


def _build_flow_label_pdf(pdf_path: Path, include_watermark: bool) -> None:
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.setLineWidth(1.0)
    c.rect(140.0, 260.0, 300.0, 240.0)
    c.line(160.0, 280.0, 340.0, 480.0)
    c.line(160.0, 480.0, 420.0, 280.0)
    c.bezier(140.0, 390.0, 140.0, 250.0, 440.0, 470.0, 440.0, 270.0)
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(225.0, 388.0, "NODE")

    if include_watermark:
        c.saveState()
        c.setFillColor(colors.Color(0.92, 0.92, 0.92))
        c.setFont("Helvetica-Bold", 44)
        c.translate(300.0, 405.0)
        c.rotate(55)
        c.drawCentredString(0, 0, "CONFIDENTIAL")
        c.restoreState()

    c.save()


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

    def test_extract_embedded_images_preserves_label_text_while_excluding_watermark(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        clean_pdf_path = root / "label_only.pdf"
        watermark_pdf_path = root / "label_with_watermark.pdf"
        drawing_region = (140.0, 292.0, 440.0, 532.0)
        _build_flow_label_pdf(clean_pdf_path, include_watermark=False)
        _build_flow_label_pdf(watermark_pdf_path, include_watermark=True)

        clean_files = _extract_embedded_images(
            pdf_path=clean_pdf_path,
            out_image_dir=root / "images_clean",
            stem="label_only",
            drawing_regions_by_page={1: [drawing_region]},
        )
        watermark_files = _extract_embedded_images(
            pdf_path=watermark_pdf_path,
            out_image_dir=root / "images_watermark",
            stem="label_with_watermark",
            drawing_regions_by_page={1: [drawing_region]},
        )

        self.assertEqual(1, len(clean_files))
        self.assertEqual(1, len(watermark_files))
        clean_image = Image.open(clean_files[0]).convert("L")
        watermark_image = Image.open(watermark_files[0]).convert("L")
        self.assertEqual(clean_image.size, watermark_image.size)
        self.assertLess(watermark_image.getextrema()[0], 255)
        self.assertIsNone(ImageChops.difference(clean_image, watermark_image).getbbox())

    def test_extract_embedded_images_expands_curve_bbox_to_connected_line_bounds(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        pdf_path = root / "group_bounds.pdf"
        _build_flow_label_pdf(pdf_path, include_watermark=False)

        curve_region = (140.0, 402.0, 440.0, 522.0)
        full_group_region = (140.0, 292.0, 440.0, 532.0)

        curve_files = _extract_embedded_images(
            pdf_path=pdf_path,
            out_image_dir=root / "images_curve",
            stem="curve_region",
            drawing_regions_by_page={1: [curve_region]},
        )
        full_group_files = _extract_embedded_images(
            pdf_path=pdf_path,
            out_image_dir=root / "images_full",
            stem="full_region",
            drawing_regions_by_page={1: [full_group_region]},
        )

        self.assertEqual(1, len(curve_files))
        self.assertEqual(1, len(full_group_files))
        curve_image = Image.open(curve_files[0]).convert("L")
        full_group_image = Image.open(full_group_files[0]).convert("L")
        self.assertEqual(full_group_image.size, curve_image.size)
        self.assertIsNone(ImageChops.difference(curve_image, full_group_image).getbbox())
