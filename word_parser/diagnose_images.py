"""
Diagnostic script: scan a .docx for all image-related XML tags and print their locations.
Usage: python3 diagnose_images.py <path-to.docx>
"""
import sys
import zipfile
from collections import defaultdict
from xml.etree import ElementTree as ET

# All known image/object tags to hunt for
IMAGE_TAGS = {
    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}drawing": "w:drawing",
    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pict": "w:pict",
    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}object": "w:object",
    "{http://schemas.openxmlformats.org/drawingml/2006/main}blip": "a:blip",
    "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}inline": "wp:inline",
    "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}anchor": "wp:anchor",
    "{urn:schemas-microsoft-com:vml}imagedata": "v:imagedata",
    "{urn:schemas-microsoft-com:vml}shape": "v:shape",
    "{urn:schemas-microsoft-com:office:office}OLEObject": "o:OLEObject",
}

R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
TAG_P   = f"{{{W_NS}}}p"
TAG_TBL = f"{{{W_NS}}}tbl"
TAG_TC  = f"{{{W_NS}}}tc"
TAG_TR  = f"{{{W_NS}}}tr"


def location_of(elem, root):
    """Return coarse location string for an element."""
    # Walk parents to determine context
    path = []
    parent_map = {c: p for p in root.iter() for c in p}

    node = elem
    while node in parent_map:
        node = parent_map[node]
        t = node.tag.split("}")[-1] if "}" in node.tag else node.tag
        path.append(t)

    if "tc" in path:
        return "table-cell"
    if "tbl" in path:
        return "table"
    if "p" in path:
        return "paragraph"
    return "other"


def scan_xml(xml_bytes, source_name):
    root = ET.fromstring(xml_bytes)
    counts = defaultdict(list)

    for full_tag, short_name in IMAGE_TAGS.items():
        for elem in root.iter(full_tag):
            loc = location_of(elem, root)
            # Grab r:embed / r:link / r:id if present
            r_embed = elem.get(f"{{{R_NS}}}embed")
            r_link  = elem.get(f"{{{R_NS}}}link")
            r_id    = elem.get(f"{{{R_NS}}}id")
            prog_id = elem.get("ProgID")
            attrs = []
            if r_embed: attrs.append(f"r:embed={r_embed}")
            if r_link:  attrs.append(f"r:link={r_link}")
            if r_id:    attrs.append(f"r:id={r_id}")
            if prog_id: attrs.append(f"ProgID={prog_id}")
            counts[short_name].append((loc, ", ".join(attrs) or "-"))

    if not counts:
        print(f"  [{source_name}] No image tags found.")
        return

    for tag, hits in sorted(counts.items()):
        print(f"  [{source_name}] {tag}: {len(hits)} hit(s)")
        for loc, attrs in hits[:10]:
            print(f"      location={loc}  attrs={attrs}")
        if len(hits) > 10:
            print(f"      ... and {len(hits)-10} more")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 diagnose_images.py <path-to.docx>")
        sys.exit(1)

    docx_path = sys.argv[1]
    print(f"Scanning: {docx_path}\n")

    with zipfile.ZipFile(docx_path) as z:
        names = z.namelist()

        # Report media files
        media = [n for n in names if n.startswith("word/media/")]
        print(f"=== Media files in docx: {len(media)} ===")
        for m in media:
            print(f"  {m}  ({z.getinfo(m).file_size} bytes)")

        print()

        # Scan XML parts
        xml_parts = [n for n in names if n.endswith(".xml") or n.endswith(".rels")]
        print(f"=== Image tag scan across {len(xml_parts)} XML parts ===")
        for part in sorted(xml_parts):
            try:
                data = z.read(part)
                scan_xml(data, part)
            except ET.ParseError:
                pass  # .rels files or non-XML

    print("\nDone.")


if __name__ == "__main__":
    main()
