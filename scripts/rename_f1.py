"""Write F1 as F with a subscript one, throughout a .docx.

Usage: python scripts/rename_f1.py IN.docx OUT.docx
"""
import re
import shutil
import sys
import zipfile

import lxml.etree as ET

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
PATTERN = re.compile(r"F1")


def main():
    src, dst = sys.argv[1], sys.argv[2]
    shutil.copy(src, dst)
    with zipfile.ZipFile(src) as z:
        names = z.namelist()
        blobs = {n: z.read(n) for n in names}

    changed = 0
    for part in [n for n in names if n.startswith("word/") and n.endswith(".xml")]:
        root = ET.fromstring(blobs[part])
        touched = False
        for node in root.iter(f"{{{W}}}t", f"{{{W}}}delText"):
            text = node.text or ""
            if "F1" in text:
                node.text = PATTERN.sub("F₁", text)
                changed += text.count("F1")
                touched = True
        if touched:
            blobs[part] = ET.tostring(root, xml_declaration=True,
                                      encoding="UTF-8", standalone=True)

    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, blobs[n])
    print(f"{changed} occurrence(s) rewritten -> {dst}")


if __name__ == "__main__":
    main()
