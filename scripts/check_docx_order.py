"""Check that a .docx keeps the child order OOXML requires.

Usage: python scripts/check_docx_order.py FILE.docx [FILE.docx ...]
Exits non-zero if any element is out of order.
"""
import sys
import zipfile

import lxml.etree as ET

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

# CT_RPr and CT_PPr, in the order the schema fixes for them.
RPR = """ins del moveFrom moveTo rStyle rFonts b bCs i iCs caps smallCaps strike
dstrike outline shadow emboss imprint noProof snapToGrid vanish webHidden color
spacing w kern position sz szCs highlight u effect bdr shd fitText vertAlign rtl
cs em lang eastAsianLayout specVanish oMath rPrChange""".split()
PPR = """pStyle keepNext keepLines pageBreakBefore framePr widowControl numPr
suppressLineNumbers pBdr shd tabs suppressAutoHyphens kinsoku wordWrap
overflowPunct topLinePunct autoSpaceDE autoSpaceDN bidi adjustRightInd
snapToGrid spacing ind contextualSpacing mirrorIndents suppressOverlap jc
textDirection textAlignment textboxTightWrap outlineLvl divId cnfStyle rPr
sectPr pPrChange""".split()
ORDERS = {"rPr": RPR, "pPr": PPR}


W14 = "http://schemas.microsoft.com/office/word/2010/wordml"


def duplicate_para_ids(root):
    """How many times each repeated paragraph id appears.

    Word writes repeated ids itself in large tables and reads those back
    without complaint, so a count that matches the unedited file is not a
    fault. What breaks a document is an edit that introduces new repeats,
    which is why this returns counts for comparison rather than a verdict.
    """
    counts = {}
    for p in root.iter(f"{{{W}}}p"):
        pid = p.get(f"{{{W14}}}paraId")
        if pid is not None:
            counts[pid] = counts.get(pid, 0) + 1
    return {k: v for k, v in counts.items() if v > 1}


def check(path):
    with zipfile.ZipFile(path) as z:
        parts = [n for n in z.namelist()
                 if n.startswith("word/") and n.endswith(".xml")]
        blobs = {n: z.read(n) for n in parts}
    problems, repeats = [], {}
    for part, blob in blobs.items():
        root = ET.fromstring(blob)
        for tag, order in ORDERS.items():
            for el in root.iter(f"{{{W}}}{tag}"):
                seen = [ET.QName(k).localname for k in el]
                ranks = [order.index(k) for k in seen if k in order]
                if ranks != sorted(ranks):
                    problems.append(f"{part}: <w:{tag}> children out of order: "
                                    + ", ".join(seen))
        dup = duplicate_para_ids(root)
        if dup:
            repeats[part] = sum(dup.values()) - len(dup)
    return problems, repeats


def main():
    """First path is the reference; the rest are compared against its repeats."""
    failed = False
    baseline = None
    for path in sys.argv[1:]:
        problems, repeats = check(path)
        name = path.rsplit("/", 1)[-1]
        total = sum(repeats.values())
        if baseline is None:
            baseline = total
            extra = 0
        else:
            extra = total - baseline
        if problems:
            failed = True
            print(f"{name}: {len(problems)} element(s) out of order")
            for line in problems[:5]:
                print("   " + line)
        else:
            print(f"{name}: child order is sound")
        if extra > 0:
            failed = True
            print(f"   {extra} paragraph id(s) repeated beyond the reference file")
        elif total:
            print(f"   {total} repeated paragraph id(s), as in the reference file")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
