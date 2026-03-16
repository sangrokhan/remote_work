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

    markdown_path = result["md_file"]
    md_text = markdown_path.read_text(encoding="utf-8")
    txt_text = result["text_file"].read_text(encoding="utf-8")

    if "CONFIDENTIAL" in md_text:
        raise AssertionError("watermark string remained in page text")
    if "Graph PDF Demo Header" in txt_text:
        raise AssertionError("header text was not removed")
    if "Graph PDF Demo Footer" in txt_text:
        raise AssertionError("footer text was not removed")

    # 테이블 2개 + 헤더/본문 추출 확인
    if result["summary"]["table_count"] < 2:
        raise AssertionError("table count is too low")

    for required in ["Widget", "Keyboard", "Alpha", "Beta", "PDF extraction sample"]:
        if required not in txt_text:
            raise AssertionError(f"missing expected token: {required}")

    if len(result["image_files"]) < 2:
        raise AssertionError("expected page images for all pages")

    print("[verify] PASS")
    print("text_file:", result["text_file"])
    print("md_file:", result["md_file"])
    print("image count:", len(result["image_files"]))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run_checks())
    except Exception as e:
        print("[verify] FAIL:", e)
        sys.exit(1)
