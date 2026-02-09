import os
import sys
import mammoth
from markdownify import markdownify as md

def convert_docx_to_markdown(docx_path, md_path):
    try:
        with open(docx_path, "rb") as docx_file:
            result = mammoth.convert_to_html(docx_file)
            html = result.value
            markdown = md(html, heading_style="ATX")
            with open(md_path, "w", encoding="utf-8") as md_file:
                md_file.write(markdown)
            return True
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python converter.py <input_docx> [output_dir]")
    else:
        in_file = sys.argv[1]
        out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(in_file), "../md")
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        out_file = os.path.join(out_dir, os.path.splitext(os.path.basename(in_file))[0] + ".md")
        if convert_docx_to_markdown(in_file, out_file):
            print(f"Successfully converted {in_file} to {out_file}")
