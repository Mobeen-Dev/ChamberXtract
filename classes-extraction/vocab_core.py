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
                add: set[str] = frozenset(),
                merges: dict[str, str] | None = None) -> Counter:
    """Return a new Counter after removing, adding, and merging words.

    merges: {source_word: target_word}. Every source's count is folded into
    target (target need not already exist). A word can be both a merge
    source and separately in `remove`/`add`; merges are applied first, then
    remove, then add, so `add` always wins and survives a same-named removal.
    """
    out = Counter(counts)
    for src, tgt in (merges or {}).items():
        src, tgt = src.upper(), tgt.upper()
        if src == tgt or src not in out:
            continue
        out[tgt] = out.get(tgt, 0) + out.pop(src)
    for w in remove:
        out.pop(w.upper(), None)
    for w in add:
        w = w.upper()
        if w not in out:
            out[w] = 0
    return out


def merge_words(counts: Counter, source: str, target: str) -> Counter:
    """Merge `source` into `target`, keeping `target` as the canonical name
    and summing both counts. `source` disappears from the result."""
    return apply_edits(counts, merges={source: target})


def filter_min_count(counts: Counter, min_count: int) -> Counter:
    """Keep only words occurring at least `min_count` times."""
    return Counter({w: n for w, n in counts.items() if n >= min_count})


# ----------------------------------------------------------------------------
# Edit-log persistence (so curation survives a re-extraction of the source CSV)
# ----------------------------------------------------------------------------

def save_edit_log(path: str | Path, remove: set[str] = frozenset(),
                  add: set[str] = frozenset(),
                  merges: dict[str, str] | None = None,
                  min_count: int = 1) -> Path:
    """Save the edit operations themselves (not just the resulting word
    list), so the same curation can be re-applied after re-extracting words
    from an updated CSV."""
    p = Path(path)
    payload = {
        "remove": sorted(w.upper() for w in remove),
        "add": sorted(w.upper() for w in add),
        "merges": {k.upper(): v.upper() for k, v in (merges or {}).items()},
        "min_count": min_count,
    }
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def load_edit_log(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"remove": [], "add": [], "merges": {}, "min_count": 1}
    return json.loads(p.read_text(encoding="utf-8"))


def apply_edit_log(counts: Counter, log: dict) -> Counter:
    """Apply a saved edit log (merges -> remove -> add -> min_count filter)
    to a freshly extracted Counter."""
    out = apply_edits(counts, remove=set(log.get("remove", [])),
                      add=set(log.get("add", [])),
                      merges=log.get("merges", {}))
    return filter_min_count(out, log.get("min_count", 1))