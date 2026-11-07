"""Small text helpers shared by the trainer, the dataset builder and the inference engine (no heavy imports)."""
from __future__ import annotations

import re

WORD_RE = re.compile(r"^[^\W\d_]+(?:['\-][^\W\d_]+)*$")


def normalize_def(text: str, max_chars: int = 220) -> str:
    """Lowercase, collapse whitespace, drop trailing punctuation, cut at a word boundary."""
    t = re.sub(r"\s+", " ", str(text)).strip().lower()
    t = t.rstrip(" .;:,")
    if len(t) > max_chars:
        t = t[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return t


def valid_word(w: str) -> bool:
    return 2 <= len(w) <= 32 and w == w.lower() and bool(WORD_RE.match(w))
