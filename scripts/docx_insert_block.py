"""Insert paragraphs and a table into a .docx as tracked insertions.

    {"author": "...", "anchor": "text to insert after",
     "blocks": [{"type": "paragraph", "template": "...", "text": "..."},
                {"type": "table", "template": "...", "rows": [[...], [...]]}]}

Usage: python scripts/docx_insert_block.py BLOCK.json IN.docx OUT.docx
"""
import copy
import json
import sys
import zipfile
from datetime import datetime, timezone

import lxml.etree as ET

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def q(tag):
    return f"{{{W}}}{tag}"


def text_of(el):
    return "".join(t.text or "" for t in el.iter(q("t")))


class Ids:
    def __init__(self, start=20000):
        self.n = start

    def next(self):
        self.n += 1
        return str(self.n)


def ins_attrs(author, when, ids):
    return {q("id"): ids.next(), q("author"): author, q("date"): when}


def mark_paragraph_inserted(p, author, when, ids):
    """Wrap every run in <w:ins> and mark the paragraph mark as inserted."""
    for r in list(p.findall(q("r"))):
        idx = list(p).index(r)
        ins = ET.Element(q("ins"), ins_attrs(author, when, ids))
        p.remove(r)
        ins.append(r)
        p.insert(idx, ins)
    pPr = p.find(q("pPr"))
    if pPr is None:
        pPr = ET.Element(q("pPr"))
        p.insert(0, pPr)
    rPr = pPr.find(q("rPr"))
    if rPr is None:
        rPr = ET.SubElement(pPr, q("rPr"))
    if rPr.find(q("ins")) is None:
        el = ET.Element(q("ins"), ins_attrs(author, when, ids))
        rPr.insert(0, el)


def build_paragraph(template, text, author, when, ids):
    """A copy of `template` carrying `text` in a single run, marked inserted."""
    p = copy.deepcopy(template)
    runs = p.findall(q("r"))
    if not runs:
        raise SystemExit("template paragraph has no run to copy")
    keep = runs[0]
    for extra in runs[1:]:
        p.remove(extra)
    for t in list(keep.findall(q("t"))):
        keep.remove(t)
    t = ET.SubElement(keep, q("t"))
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    mark_paragraph_inserted(p, author, when, ids)
    return p


def set_cell_text(tc, text):
    ps = tc.findall(q("p"))
    keep = ps[0]
    for extra in ps[1:]:
        tc.remove(extra)
    runs = keep.findall(q("r"))
    if not runs:
        r = ET.SubElement(keep, q("r"))
        runs = [r]
    first = runs[0]
    for extra in runs[1:]:
        keep.remove(extra)
    for t in list(first.findall(q("t"))):
        first.remove(t)
    t = ET.SubElement(first, q("t"))
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return keep


def build_table(template, rows, author, when, ids):
    tbl = copy.deepcopy(template)
    trs = tbl.findall(q("tr"))
    if len(trs) < len(rows):
        # extend by cloning the last body row
        while len(trs) < len(rows):
            clone = copy.deepcopy(trs[-1])
            tbl.append(clone)
            trs = tbl.findall(q("tr"))
    for extra in trs[len(rows):]:
        tbl.remove(extra)
    trs = tbl.findall(q("tr"))

    for tr, values in zip(trs, rows):
        tcs = tr.findall(q("tc"))
        if len(tcs) != len(values):
            raise SystemExit(
                f"template row has {len(tcs)} cells, block gives {len(values)}")
        for tc, value in zip(tcs, values):
            p = set_cell_text(tc, value)
            mark_paragraph_inserted(p, author, when, ids)
        trPr = tr.find(q("trPr"))
        if trPr is None:
            trPr = ET.Element(q("trPr"))
            tr.insert(0, trPr)
        if trPr.find(q("ins")) is None:
            trPr.append(ET.Element(q("ins"), ins_attrs(author, when, ids)))
    return tbl


def main():
    block_path, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    spec = json.load(open(block_path))
    author = spec.get("author", "Revision")
    when = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ids = Ids()

    with zipfile.ZipFile(src) as z:
        names = z.namelist()
        blobs = {n: z.read(n) for n in names}
    root = ET.fromstring(blobs["word/document.xml"])
    body = root.find(q("body"))

    paras = body.findall(q("p"))
    anchors = [p for p in paras if spec["anchor"] in text_of(p)]
    if len(anchors) != 1:
        raise SystemExit(f"anchor matched {len(anchors)} paragraphs")
    anchor = anchors[0]
    at = list(body).index(anchor)

    made = []
    for b in spec["blocks"]:
        if b["type"] == "paragraph":
            tpl = [p for p in body.iter(q("p")) if b["template"] in text_of(p)]
            if not tpl:
                raise SystemExit(f"no template paragraph matching {b['template'][:40]!r}")
            made.append(build_paragraph(tpl[0], b["text"], author, when, ids))
        elif b["type"] == "table":
            tpl = [t for t in body.iter(q("tbl")) if b["template"] in text_of(t)]
            if not tpl:
                raise SystemExit(f"no template table matching {b['template'][:40]!r}")
            made.append(build_table(tpl[0], b["rows"], author, when, ids))
        else:
            raise SystemExit(f"unknown block type {b['type']}")

    for off, el in enumerate(made):
        body.insert(at + 1 + off, el)

    blobs["word/document.xml"] = ET.tostring(
        root, xml_declaration=True, encoding="UTF-8", standalone=True)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, blobs[n])
    kinds = ", ".join(b["type"] for b in spec["blocks"])
    print(f"inserted {len(made)} block(s) ({kinds}) after the anchor -> {dst}")


if __name__ == "__main__":
    main()
