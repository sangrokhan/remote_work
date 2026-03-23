from __future__ import annotations

import unittest
from types import SimpleNamespace

from PIL import Image

from extractor.images import _crop_page_region


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
