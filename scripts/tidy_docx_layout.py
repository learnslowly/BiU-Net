"""Bring figures and captions in a .docx to one layout.

  figures    centred, with no indent, so a wide panel sits on the page middle
             instead of inheriting the body text's first-line indent
  captions   no indent and two points smaller than body text, the size the
             supplement already uses, applied to the main text as well
  inserted   a run added by a tracked edit inherits its size from the style
  runs       rather than from the paragraph it joined, which shows up as a
             different typeface; the size of the paragraph's other runs is
             copied onto it

Usage: python scripts/tidy_docx_layout.py IN.docx OUT.docx
"""
import re
import shutil
import sys
import zipfile
from collections import Counter

import lxml.etree as ET

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
CAPTION = re.compile(r"^(Figure|Table)\s+S?\d+[.\s]")
# CT_PPr child order, enough of it to place jc and ind correctly
PPR_ORDER = """pStyle keepNext keepLines pageBreakBefore framePr widowControl
numPr suppressLineNumbers pBdr shd tabs suppressAutoHyphens kinsoku wordWrap
overflowPunct topLinePunct autoSpaceDE autoSpaceDN bidi adjustRightInd
snapToGrid spacing ind contextualSpacing mirrorIndents suppressOverlap jc
textDirection textAlignment textboxTightWrap outlineLvl divId cnfStyle rPr
sectPr pPrChange""".split()
RPR_ORDER = """ins del moveFrom moveTo rStyle rFonts b bCs i iCs caps smallCaps
strike dstrike outline shadow emboss imprint noProof snapToGrid vanish
webHidden color spacing w kern position sz szCs highlight u effect bdr shd
fitText vertAlign rtl cs em lang eastAsianLayout specVanish oMath
rPrChange""".split()


def q(tag):
    return f"{{{W}}}{tag}"


def ordered_insert(parent, child, order):
    """Put a child where the schema expects it rather than at the end."""
    rank = order.index(ET.QName(child).localname)
    for existing in parent:
        name = ET.QName(existing).localname
        if name in order and order.index(name) > rank:
            existing.addprevious(child)
            return
    parent.append(child)


def get_or_make(parent, tag, order):
    node = parent.find(q(tag))
    if node is None:
        node = parent.makeelement(q(tag), {})
        if tag in ("pPr", "rPr") and ET.QName(parent).localname in ("p", "r"):
            # a property bag is always the first child of the run or paragraph
            parent.insert(0, node)
        else:
            ordered_insert(parent, node, order)
    return node


def para_text(p):
    return "".join(t.text or "" for t in p.iter(q("t")))


def has_picture(p):
    return p.find(f".//{q('drawing')}") is not None or p.find(f".//{q('pict')}") is not None


def body_size(body):
    """The size the body text is set at.

    Counting every run would let a document that is mostly captions, as the
    supplement is, report the caption size as the body size. Only paragraphs of
    running prose are counted: long, not a caption, and not holding a figure.
    """
    sizes = Counter()
    for p in body.findall(q("p")):
        text = para_text(p).strip()
        if has_picture(p) or CAPTION.match(text) or len(text) < 120:
            continue
        for r in p.findall(q("r")):
            rPr = r.find(q("rPr"))
            sz = rPr.find(q("sz")) if rPr is not None else None
            if sz is not None:
                sizes[sz.get(q("val"))] += 1
    return sizes.most_common(1)[0][0] if sizes else "24"


def centre_and_unindent(p):
    pPr = get_or_make(p, "pPr", PPR_ORDER)
    get_or_make(pPr, "jc", PPR_ORDER).set(q("val"), "center")
    ind = get_or_make(pPr, "ind", PPR_ORDER)
    for attr in ("firstLine", "left", "start", "hanging"):
        ind.set(q(attr), "0")


def unindent(p):
    pPr = get_or_make(p, "pPr", PPR_ORDER)
    ind = get_or_make(pPr, "ind", PPR_ORDER)
    for attr in ("firstLine", "left", "start", "hanging"):
        ind.set(q(attr), "0")


def set_size(p, half_points):
    """Force every run of a paragraph, and its mark, to one size."""
    pPr = get_or_make(p, "pPr", PPR_ORDER)
    mark = get_or_make(pPr, "rPr", PPR_ORDER)
    for tag in ("sz", "szCs"):
        get_or_make(mark, tag, RPR_ORDER).set(q("val"), half_points)
    for r in p.iter(q("r")):
        rPr = get_or_make(r, "rPr", RPR_ORDER)
        for tag in ("sz", "szCs"):
            get_or_make(rPr, tag, RPR_ORDER).set(q("val"), half_points)


def fix_inserted_runs(p, size):
    """Give a tracked insertion the size its neighbours carry."""
    fixed = 0
    for ins in p.findall(q("ins")):
        for r in ins.findall(q("r")):
            rPr = get_or_make(r, "rPr", RPR_ORDER)
            if rPr.find(q("sz")) is None:
                for tag in ("sz", "szCs"):
                    get_or_make(rPr, tag, RPR_ORDER).set(q("val"), size)
                fixed += 1
    return fixed


def main():
    src, dst = sys.argv[1], sys.argv[2]
    with zipfile.ZipFile(src) as z:
        names = z.namelist()
        blobs = {n: z.read(n) for n in names}
    root = ET.fromstring(blobs["word/document.xml"])
    body = root.find(q("body"))

    size = body_size(body)
    caption_size = str(max(int(size) - 4, 2))  # two points, in half-points
    figures = captions = runs = 0
    # iter rather than findall: a caption can sit inside a table cell, and one
    # left at the body size while its neighbours shrink is the visible fault
    for p in body.iter(q("p")):
        text = para_text(p).strip()
        is_caption = bool(CAPTION.match(text)) and len(text) > 20
        if has_picture(p):
            centre_and_unindent(p)
            figures += 1
        if is_caption:
            # a paragraph can hold both the figure and its caption; it is
            # centred as a figure and still set at the caption size
            unindent(p)
            set_size(p, caption_size)
            captions += 1
        elif not has_picture(p):
            runs += fix_inserted_runs(p, size)
    # captions inside table cells, and footnotes, follow the same size
    for part in [n for n in names if re.match(r"word/(foot|end)notes\.xml", n)]:
        sub = ET.fromstring(blobs[part])
        for p in sub.iter(q("p")):
            set_size(p, caption_size)
        blobs[part] = ET.tostring(sub, xml_declaration=True, encoding="UTF-8",
                                  standalone=True)

    print(f"  body text {int(size)/2:g} pt, captions and notes set to "
          f"{int(caption_size)/2:g} pt")
    print(f"  {figures} figure paragraph(s) centred, {captions} caption(s) "
          f"resized, {runs} inserted run(s) given the body size")

    blobs["word/document.xml"] = ET.tostring(root, xml_declaration=True,
                                             encoding="UTF-8", standalone=True)
    shutil.copy(src, dst)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, blobs[n])
    print(f"  written -> {dst}")


if __name__ == "__main__":
    main()
