"""Accept every tracked change in a .docx.

Usage: python scripts/docx_accept_all.py IN.docx OUT.docx
"""
import sys
import zipfile

import lxml.etree as ET

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
KEEP_CONTENT = {"ins", "moveTo"}                 # unwrap, keep children
DROP_WHOLE = {"del", "moveFrom"}                 # remove element and children
DROP_RECORD = {"rPrChange", "pPrChange", "tblPrChange", "trPrChange",
               "tcPrChange", "sectPrChange", "numberingChange",
               "moveFromRangeStart", "moveFromRangeEnd",
               "moveToRangeStart", "moveToRangeEnd",
               "customXmlInsRangeStart", "customXmlInsRangeEnd",
               "customXmlDelRangeStart", "customXmlDelRangeEnd"}


def q(tag):
    return f"{{{W}}}{tag}"


def local(el):
    tag = el.tag
    return tag.split("}", 1)[1] if isinstance(tag, str) and "}" in tag else None


def accept(root):
    counts = {"ins": 0, "del": 0, "moveTo": 0, "moveFrom": 0, "prChange": 0,
              "delText": 0}

    # Deletions first: their content disappears, so nothing inside needs fixing.
    for el in list(root.iter()):
        name = local(el)
        if name in DROP_WHOLE:
            counts[name] = counts.get(name, 0) + 1
            el.getparent().remove(el)

    for el in list(root.iter()):
        if local(el) in DROP_RECORD:
            counts["prChange"] += 1
            el.getparent().remove(el)

    # Insertions: splice the children into the parent where the wrapper stood,
    # and carry the wrapper's tail so no whitespace is lost.
    changed = True
    while changed:
        changed = False
        for el in list(root.iter()):
            if local(el) not in KEEP_CONTENT:
                continue
            parent = el.getparent()
            if parent is None:
                continue
            counts[local(el)] = counts.get(local(el), 0) + 1
            idx = list(parent).index(el)
            children = list(el)
            for off, child in enumerate(children):
                parent.insert(idx + off, child)
            if el.tail:
                if children:
                    last = children[-1]
                    last.tail = (last.tail or "") + el.tail
                elif idx > 0:
                    prev = list(parent)[idx - 1]
                    prev.tail = (prev.tail or "") + el.tail
            parent.remove(el)
            changed = True

    # A run that was inside a deletion is gone; any stray <w:delText> left in an
    # accepted run is ordinary text once the revision is accepted.
    for el in list(root.iter()):
        if local(el) == "delText":
            counts["delText"] += 1
            el.tag = q("t")
            el.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")

    return counts


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: docx_accept_all.py IN.docx OUT.docx")
    src, dst = sys.argv[1], sys.argv[2]

    with zipfile.ZipFile(src) as z:
        names = z.namelist()
        blobs = {n: z.read(n) for n in names}

    total = {}
    for part in [n for n in names if n.endswith(".xml") and
                 ("document" in n or "footnotes" in n or "endnotes" in n or
                  "header" in n or "footer" in n)]:
        root = ET.fromstring(blobs[part])
        counts = accept(root)
        if any(counts.values()):
            blobs[part] = ET.tostring(root, xml_declaration=True,
                                      encoding="UTF-8", standalone=True)
            for k, v in counts.items():
                total[k] = total.get(k, 0) + v
            print(f"  {part}: " + ", ".join(f"{k}={v}" for k, v in counts.items() if v))

    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, blobs[n])
    print(f"accepted: " + ", ".join(f"{k}={v}" for k, v in total.items() if v)
          + f"  ->  {dst}")


if __name__ == "__main__":
    main()
