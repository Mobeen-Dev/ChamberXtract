"""
vocab_app.py - Upload a CSV, pick a column, see every distinct word with
its occurrence count. Delete words, add new ones, merge two words into
one (choosing which name survives), filter by a minimum count, then
save the curated list.

Run:  streamlit run vocab_app.py
"""

from __future__ import annotations

import io
import json
from collections import Counter

import pandas as pd
import streamlit as st

from vocab_core import apply_edits, filter_min_count, to_table, word_frequencies

st.set_page_config(page_title="Keyword curation", layout="wide")
st.title("Word frequency - keyword curation")

# ---------------------------------------------------------------------------
# 1. Upload + column
# ---------------------------------------------------------------------------

mode = st.radio(
    "What is this file?",
    ["Raw data - tokenize a text column", "Already word,count (from a previous export)"],
    horizontal=True,
    help="Pick the second option when reloading a file this app (or "
         "vocab_cli.py extract) already produced - it has one row per word "
         "and should NOT be re-tokenized, or every count collapses to 1.",
)

up = st.file_uploader("CSV file", type=["csv"])
if up is None:
    st.info("Upload a CSV to begin.")
    st.stop()


@st.cache_data(show_spinner=False)
def load(buf: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(buf), dtype=str, keep_default_na=False)


df = load(up.getvalue())

if mode.startswith("Already"):
    cols = list(df.columns)
    wcol = st.selectbox("Word column", cols,
                        index=cols.index("word") if "word" in cols else 0)
    ccol = st.selectbox("Count column", cols,
                        index=cols.index("count") if "count" in cols
                        else min(1, len(cols) - 1))
    column = f"__wc__:{wcol}:{ccol}"   # cache-key discriminator only
else:
    column = st.selectbox("Text column to tokenize", df.columns)

# Reset curation state only when the source file, mode, or column changes.
key = (up.name, up.size, mode, column)
if st.session_state.get("_key") != key:
    st.session_state["_key"] = key
    if mode.startswith("Already"):
        counts_raw = pd.to_numeric(df[ccol], errors="coerce").fillna(0).astype(int)
        st.session_state["base_counts"] = {
            str(w).upper(): int(n) for w, n in zip(df[wcol], counts_raw)
        }
    else:
        st.session_state["base_counts"] = dict(word_frequencies(df, column))
    st.session_state["removed"] = set()
    st.session_state["added"] = set()
    st.session_state["merges"] = {}       # source -> target

base_counts = st.session_state["base_counts"]
removed = st.session_state["removed"]
added = st.session_state["added"]
merges = st.session_state["merges"]

# Everything downstream is computed from base_counts + the three edit sets,
# so merges/removes/adds compose the same way every rerun.
working = apply_edits(Counter(base_counts), remove=removed, add=added, merges=merges)

# ---------------------------------------------------------------------------
# 2. Merge two words
# ---------------------------------------------------------------------------

st.subheader("Merge two words")
current_words = sorted(working.keys())
mc1, mc2, mc3 = st.columns([2, 2, 1])
w1 = mc1.selectbox("Word A", current_words, key="merge_a")
w2 = mc2.selectbox("Word B", current_words, key="merge_b",
                   index=min(1, len(current_words) - 1))

# A stable key with changing options: reset the stored selection whenever
# it no longer matches one of the two current choices, so Streamlit doesn't
# try to resolve a value that dropped out of `options`.
if st.session_state.get("merge_keep") not in (w1, w2):
    st.session_state["merge_keep"] = w1
keep = mc3.radio("Keep", [w1, w2], key="merge_keep")

if st.button("Merge", disabled=(w1 == w2)):
    other = w2 if keep == w1 else w1
    st.session_state["merges"][other] = keep
    # if the losing side was itself a manually-added word, drop it from
    # `added` so it doesn't get re-created by apply_edits' add step
    st.session_state["added"].discard(other)
    st.rerun()

if merges:
    st.caption("Active merges: " + ", ".join(f"{s} -> {t}" for s, t in merges.items()))
    if st.button("Clear all merges"):
        st.session_state["merges"] = {}
        st.rerun()

# ---------------------------------------------------------------------------
# 3. Minimum-count filter
# ---------------------------------------------------------------------------

st.subheader("Show words with count greater than or equal to")
max_count = max(working.values(), default=1)
min_count = st.slider("Minimum count", 0, int(max_count), 0)

# ---------------------------------------------------------------------------
# 4. Word table with remove checkboxes
# ---------------------------------------------------------------------------

visible = filter_min_count(working, min_count) if min_count else working
st.subheader(f"{len(visible)} of {len(working)} words shown "
            f"(count >= {min_count})")

table = to_table(visible)
search = st.text_input("Filter (contains)", "")
view = table[table["word"].str.contains(search.upper(), na=False)] if search else table

edited = st.data_editor(
    view.assign(remove=False),
    column_config={
        "word": st.column_config.TextColumn(disabled=True),
        "count": st.column_config.NumberColumn(disabled=True),
        "remove": st.column_config.CheckboxColumn(help="Check to delete this word"),
    },
    hide_index=True,
    width="stretch",
    height=min(600, 40 + 35 * len(view)),
    key=f"editor_{min_count}_{search}",
)

to_remove_now = set(edited.loc[edited["remove"], "word"])
if to_remove_now:
    if st.button(f"Delete {len(to_remove_now)} checked word(s)"):
        st.session_state["removed"] |= to_remove_now
        st.rerun()

# ---------------------------------------------------------------------------
# 5. Add new words
# ---------------------------------------------------------------------------

st.subheader("Add words")
new_words_raw = st.text_input("Comma-separated words to add (not from the data)", "")
if st.button("Add") and new_words_raw.strip():
    st.session_state["added"] |= {w.strip().upper() for w in new_words_raw.split(",")
                                  if w.strip()}
    st.rerun()

# ---------------------------------------------------------------------------
# 6. Preview + save
# ---------------------------------------------------------------------------

final = filter_min_count(working, min_count) if min_count else working
st.subheader(f"Curated list preview - {len(final)} words")
st.dataframe(to_table(final), width="stretch", hide_index=True, height=250)

c1, c2, c3 = st.columns([1, 1, 2])
fname = c1.text_input("Filename", "keywords.json")
fmt = c2.radio("Format", ["json", "txt"], horizontal=True)

if fmt == "json":
    payload = json.dumps(sorted(final.keys()), indent=2, ensure_ascii=False).encode()
    mime = "application/json"
else:
    payload = ("\n".join(sorted(final.keys())) + "\n").encode()
    mime = "text/plain"

c3.download_button("Save curated list", payload, fname, mime, type="primary")

# Edit log, so the same curation can be reapplied after re-extracting from
# an updated / larger CSV via `vocab_cli.py reapply`.
log_payload = json.dumps({
    "remove": sorted(st.session_state["removed"]),
    "add": sorted(st.session_state["added"]),
    "merges": st.session_state["merges"],
    "min_count": min_count,
}, indent=2, ensure_ascii=False).encode()
st.download_button("Save edit log (for reapplying later)", log_payload,
                   fname.rsplit(".", 1)[0] + ".edits.json", "application/json")