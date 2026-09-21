"""
vocab_app.py - Upload a CSV, pick a column, see every distinct word with
its occurrence count, delete/add words, then save the curated list.

Run:  streamlit run vocab_app.py
"""

from __future__ import annotations

import io
from collections import Counter

import pandas as pd
import streamlit as st

from vocab_core import apply_edits, save_curated, to_table, word_frequencies

st.set_page_config(page_title="Keyword curation", layout="wide")
st.title("Word frequency - keyword curation")

# ---------------------------------------------------------------------------
# 1. Upload + column
# ---------------------------------------------------------------------------

up = st.file_uploader("CSV file", type=["csv"])
if up is None:
    st.info("Upload a CSV, then pick the text column to pull words from.")
    st.stop()


@st.cache_data(show_spinner=False)
def load(buf: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(buf), dtype=str, keep_default_na=False)


df = load(up.getvalue())
column = st.selectbox("Column", df.columns)

# Recompute only when the source file or column changes.
key = (up.name, up.size, column)
if st.session_state.get("_key") != key:
    counts = word_frequencies(df, column)
    st.session_state["_key"] = key
    st.session_state["counts"] = dict(counts)
    st.session_state["removed"] = set()

counts = st.session_state["counts"]
removed = st.session_state["removed"]

# ---------------------------------------------------------------------------
# 2. Current table: word, count, remove-checkbox
# ---------------------------------------------------------------------------

st.subheader(f"{len(counts)} distinct words in '{column}'")

table = to_table(Counter(counts))
table["remove"] = table["word"].isin(removed)

search = st.text_input("Filter (contains)", "")
view = table[table["word"].str.contains(search.upper(), na=False)] if search else table

edited = st.data_editor(
    view,
    column_config={
        "word": st.column_config.TextColumn(disabled=True),
        "count": st.column_config.NumberColumn(disabled=True),
        "remove": st.column_config.CheckboxColumn(help="Check to delete this word"),
    },
    hide_index=True,
    width="stretch",
    height=min(600, 40 + 35 * len(view)),
)

# Sync checkbox state back into the removed set (only for currently visible rows).
newly_checked = set(edited.loc[edited["remove"], "word"])
newly_unchecked = set(view["word"]) - newly_checked
st.session_state["removed"] = (removed - newly_unchecked) | newly_checked
removed = st.session_state["removed"]

if removed:
    st.caption(f"Marked for removal ({len(removed)}): " + ", ".join(sorted(removed)))

# ---------------------------------------------------------------------------
# 3. Add new words
# ---------------------------------------------------------------------------

st.subheader("Add words")
new_words = st.text_input("Comma-separated words to add (not from the data)", "")
add_set = {w.strip().upper() for w in new_words.split(",") if w.strip()}
if add_set:
    st.caption(f"Will add: {', '.join(sorted(add_set))}")

# ---------------------------------------------------------------------------
# 4. Preview + save
# ---------------------------------------------------------------------------

curated = apply_edits(Counter(counts), remove=removed, add=add_set)
st.subheader(f"Curated list preview - {len(curated)} words")
st.dataframe(to_table(curated), width="stretch", hide_index=True, height=250)

c1, c2, c3 = st.columns([1, 1, 2])
fname = c1.text_input("Filename", "keywords.json")
fmt = c2.radio("Format", ["json", "txt"], horizontal=True)

if fmt == "json":
    import json
    payload = json.dumps(sorted(curated.keys()), indent=2, ensure_ascii=False).encode()
    mime = "application/json"
else:
    payload = ("\n".join(sorted(curated.keys())) + "\n").encode()
    mime = "text/plain"

c3.download_button("Save curated list", payload, fname, mime, type="primary")