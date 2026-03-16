from __future__ import annotations

from pathlib import Path

from sample_generator import create_demo_pdf
from extractor import extract_pdf_to_outputs


def main() -> None:
    base = Path(__file__).resolve().parent
    output_root = base / "artifacts" / "run_demo"
    pdf_path = base / "sample.pdf"
    md_dir = output_root / "md"
    image_dir = output_root / "images"

    create_demo_pdf(pdf_path)
    result = extract_pdf_to_outputs(
        pdf_path=pdf_path,
        out_md_dir=md_dir,
        out_image_dir=image_dir,
        stem="demo",
    )

    print("[demo] PDF:", pdf_path)
    print("[demo] text output:", result["text_file"])
    print("[demo] markdown output:", result["md_file"])
    print("[demo] table markdown output:", result["table_md_file"])
    print("[demo] summary:", result["summary"])


if __name__ == "__main__":
    main()
