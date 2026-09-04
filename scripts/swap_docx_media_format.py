"""Replace an embedded image and change its format at the same time.

Usage: python scripts/swap_docx_media_format.py IN.docx OUT.docx old=new [old=new ...]
  old   the media part in the document, such as image2.png
  new   the replacement on disk, whose extension decides the new part name
"""
import re
import shutil
import sys
import zipfile

RELS = "word/_rels/document.xml.rels"
TYPES = "[Content_Types].xml"


def main():
    src, dst = sys.argv[1], sys.argv[2]
    swaps = dict(a.split("=", 1) for a in sys.argv[3:])
    if not swaps:
        raise SystemExit("give at least one old=new pair")

    with zipfile.ZipFile(src) as z:
        names = z.namelist()
        blobs = {n: z.read(n) for n in names}

    rels = blobs[RELS].decode("utf-8")
    order = []
    for old, path in swaps.items():
        ext = path.rsplit(".", 1)[-1].lower()
        new = old.rsplit(".", 1)[0] + "." + ext
        old_key, new_key = f"word/media/{old}", f"word/media/{new}"
        if old_key not in blobs:
            raise SystemExit(f"{src} has no {old_key}")
        if f'Target="media/{old}"' not in rels:
            raise SystemExit(f"no relationship points at media/{old}")
        del blobs[old_key]
        blobs[new_key] = open(path, "rb").read()
        rels = rels.replace(f'Target="media/{old}"', f'Target="media/{new}"')
        order.append((old_key, new_key))
        print(f"  {old} -> {new} ({len(blobs[new_key])/1024:.0f} KB)")

    blobs[RELS] = rels.encode("utf-8")
    types = blobs[TYPES].decode("utf-8")
    for ext in {p.rsplit(".", 1)[-1].lower() for p in swaps.values()}:
        if f'Extension="{ext}"' not in types:
            raise SystemExit(f"{TYPES} declares no content type for .{ext}")

    # keep the archive's part order, substituting each renamed part in place
    renamed = dict(order)
    names = [renamed.get(n, n) for n in names]

    shutil.copy(src, dst)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, blobs[n])
    print(f"{len(swaps)} image(s) replaced -> {dst}")


if __name__ == "__main__":
    main()
