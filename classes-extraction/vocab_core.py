"""
vocab_core.py - Extract distinct words + counts from a CSV column,
let the user curate (delete/add) the list, and save it back out.

No UI imports here - reused by both vocab_cli.py and vocab_app.py.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

RE_NON_ALNUM = re.compile(r"[^A-Za-z0-9 ]+")
RE_WS = re.compile(r"\s+")


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    s = RE_NON_ALNUM.sub(" ", text.upper())
    return [t for t in RE_WS.split(s) if len(t) > 1]


def word_frequencies(df: pd.DataFrame, column: str) -> Counter:
    """One Counter of every distinct word in `column` across all rows."""
    if column not in df.columns:
        raise KeyError(f"Column {column!r} not in CSV. Available: {list(df.columns)}")
    c = Counter()
    for val in df[column].dropna():
        c.update(tokenize(val))
    return c


def to_table(counts: Counter) -> pd.DataFrame:
    """word/count table, most frequent first, for display or CSV export."""
    return pd.DataFrame(counts.most_common(), columns=["word", "count"])


# ----------------------------------------------------------------------------
# Curated-list persistence
# ----------------------------------------------------------------------------

def load_curated(path: str | Path) -> set[str]:
    """Read a previously saved keyword list (JSON list or one-word-per-line)."""
    p = Path(path)
    if not p.exists():
        return set()
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    if text.lstrip().startswith("["):
        return {w.upper() for w in json.loads(text)}
    return {ln.strip().upper() for ln in text.splitlines() if ln.strip()}


def save_curated(words: set[str] | list[str], path: str | Path,
                 fmt: str = "json") -> Path:
    """Save the curated keyword list. fmt: 'json' or 'txt'."""
    p = Path(path)
    ordered = sorted(set(w.upper() for w in words))
    if fmt == "json":
        p.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        p.write_text("\n".join(ordered) + "\n", encoding="utf-8")
    return p


def apply_edits(counts: Counter, remove: set[str] = frozenset(),
                add: set[str] = frozenset()) -> Counter:
    """Return a new Counter with `remove` dropped and `add` inserted (count=0
    if not already present, so newly added words are visibly distinct)."""
    out = Counter(counts)
    for w in remove:
        out.pop(w.upper(), None)
    for w in add:
        w = w.upper()
        if w not in out:
            out[w] = 0
    return out