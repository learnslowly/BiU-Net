"""Locate the submitted supplementary DOCX, wherever the script is running.
"""
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATTERN = "*upplementary*.docx"

CANDIDATES = tuple(
    os.path.join(d, PATTERN)
    for d in (os.environ.get("SUPPLEMENTARY_DOCX", ""),
              os.path.join(ROOT, "manuscript"), "manuscript")
    if d
)


def find(explicit=None):
    """Path of the supplementary DOCX, or None when no copy is reachable."""
    env = os.environ.get("SUPPLEMENTARY_DOCX", "")
    if explicit and os.path.exists(explicit):
        return explicit
    if env and os.path.isfile(env):
        return env
    for pattern in CANDIDATES:
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    return None
