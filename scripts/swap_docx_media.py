"""Replace the images embedded in a .docx, leaving the document itself alone.

Usage: python scripts/swap_docx_media.py IN.docx OUT.docx part=file [part=file ...]
  part   the media name inside the document, such as image3.emf
  file   the replacement on disk
"""
import shutil
import sys
import zipfile


def main():
    src, dst = sys.argv[1], sys.argv[2]
    swaps = dict(a.split("=", 1) for a in sys.argv[3:])
    if not swaps:
        raise SystemExit("give at least one part=file pair")

    with zipfile.ZipFile(src) as z:
        names = z.namelist()
        blobs = {n: z.read(n) for n in names}

    for part, path in swaps.items():
        key = f"word/media/{part}"
        if key not in blobs:
            raise SystemExit(f"{src} has no {key}")
        before = len(blobs[key])
        blobs[key] = open(path, "rb").read()
        print(f"  {part}: {before/1024:.0f} KB -> {len(blobs[key])/1024:.0f} KB "
              f"({path.rsplit('/', 1)[-1]})")

    shutil.copy(src, dst)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, blobs[n])
    print(f"{len(swaps)} image(s) replaced -> {dst}")


if __name__ == "__main__":
    main()
