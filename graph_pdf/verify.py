from __future__ import annotations

from pathlib import Path
import sys

from sample_generator import create_demo_pdf
from extractor import extract_pdf_to_outputs


def run_checks() -> int:
    base = Path(__file__).resolve().parent
    root = base / "artifacts" / "verify"
    pdf_path = root / "sample_verify.pdf"
    md_dir = root / "md"
    image_dir = root / "images"

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    create_demo_pdf(pdf_path)

    result = extract_pdf_to_outputs(
        pdf_path=pdf_path,
        out_md_dir=md_dir,
        out_image_dir=image_dir,
        stem="verify",
    )

    markdown_text = result["markdown"]
    txt_text = result["text_file"].read_text(encoding="utf-8")

    if "CONFIDENTIAL" in markdown_text:
        raise AssertionError("watermark string remained in extracted text")
    if "Graph PDF Demo Header" in txt_text:
        raise AssertionError("header text was not removed")
    if "Graph PDF Demo Footer" in txt_text:
        raise AssertionError("footer text was not removed")

    if result["summary"]["table_count"] < 3:
        raise AssertionError("expected at least 3 tables including spanning continuation")

    # Body + table multi-line + merged-cell style checks.
    required_text = [
        "Chapter 1: Deep Structure Verification",
        "- 1st level bullet: layout and spacing checks",
        "- nested detail: line 2 confirms indentation",
        "- level 1: body copy, one of many lines",
        "Scenario matrix",
        "Docs",
        "- 1st-level continuation bullet",
    ]

    for required in required_text:
        if required not in txt_text:
            raise AssertionError(f"missing expected token: {required}")

    # The spanning table should be merged into one logical table across page 1+2.
    if txt_text.count("### Page 1 table") < 1:
        raise AssertionError("missing table extraction output")

    if "scenario matrix" in txt_text.lower() and "### page 1 table" not in txt_text.lower():
        raise AssertionError("table output marker changed after spanning merge")

    if len(result["image_files"]) < 2:
        raise AssertionError("expected page images for all pages")

    print("[verify] PASS")
    print("text_file:", result["text_file"])
    print("md_file:", result["md_file"])
    print("image count:", len(result["image_files"]))
    print("table count:", result["summary"]["table_count"])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run_checks())
    except Exception as e:
        print("[verify] FAIL:", e)
        sys.exit(1)
