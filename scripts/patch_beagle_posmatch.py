"""Build a Beagle variant whose reference-target join matches on position alone.

Usage: python scripts/patch_beagle_posmatch.py SRC_DIR OUT_DIR
"""
import os
import shutil
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "$REMOTE_ROOT/beagle/src/src"
OUT = sys.argv[2] if len(sys.argv) > 2 else "$REMOTE_ROOT/beagle_posmatch/src"

TARGET = os.path.join("vcf", "RefTargSlidingWindow.java")

# The inner loop advances the reference iterator past same-position records that
# are not equal; with position-only matching it must stop at the position.
OLD_ADVANCE = ("|| (nextRefRec.marker().pos() == targPos "
               "&& targMarker.equals(nextRefRec.marker()) == false)")
NEW_ADVANCE = "/* POSMATCH: same position is a match, do not advance past it */"

OLD_ACCEPT = "if (nextRefRec!=null && nextRefRec.marker().equals(targMarker)) {"
NEW_ACCEPT = ("if (nextRefRec!=null "
              "&& nextRefRec.marker().chromIndex()==targMarker.chromIndex() "
              "&& nextRefRec.marker().pos()==targMarker.pos()) {  // POSMATCH")


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    shutil.copytree(SRC, OUT)
    path = os.path.join(OUT, TARGET)
    text = open(path).read()
    for old in (OLD_ADVANCE, OLD_ACCEPT):
        if old not in text:
            sys.exit(f"pattern not found, source differs from the expected release:\n{old}")
    text = text.replace(OLD_ADVANCE, NEW_ADVANCE, 1)
    text = text.replace(OLD_ACCEPT, NEW_ACCEPT, 1)
    open(path, "w").write(text)
    print(f"patched {path}")
    for i, line in enumerate(open(path), 1):
        if "POSMATCH" in line:
            print(f"  {i}: {line.rstrip()}")


if __name__ == "__main__":
    main()
