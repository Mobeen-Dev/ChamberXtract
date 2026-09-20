"""
app.py - VoterParse clustering & word-analysis dashboard.

Run:  streamlit run app.py
"""

from __future__ import annotations

import io

import altair as alt
import pandas as pd
import streamlit as st

from cluster_core import (DEFAULT_STOPWORDS, ClusterConfig, cluster,
                          cluster_summary, cluster_word_table, word_counts)

st.set_page_config(page_title="Company-name clustering", layout="wide")
st.title("Company-name clustering & word analysis")


# ---------------------------------------------------------------------------
# 1. Upload + column selection
# ---------------------------------------------------------------------------

up = st.file_uploader("CSV file", type=["csv"])
if up is None:
    st.info("Upload any CSV, then pick the column that holds company names.")
    st.stop()


@st.cache_data(show_spinner=False)
def load(buf: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(buf), dtype=str, keep_default_na=False)


df = load(up.getvalue())
st.caption(f"{len(df):,} rows x {len(df.columns)} columns")

guess = (next((c for c in df.columns if "company" in c.lower()), None)
         or next((c for c in df.columns if "name" in c.lower()), None)
         or df.columns[0])
column = st.selectbox("Company-name column", df.columns,
                      index=list(df.columns).index(guess))
st.dataframe(df[[column]].head(5), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# 2. Parameters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Clustering parameters")
    method = st.radio("Algorithm", ["agglomerative", "kmeans"],
                      help="Agglomerative = no k needed, O(n^2) memory. "
                           "KMeans = scalable, needs k.")
    analyzer = st.radio("Token unit", ["char_wb", "word"], index=0,
                        help="char_wb tolerates spelling variants; "
                             "word groups by shared whole words.")
    ng_lo, ng_hi = st.slider("N-gram range", 1, 6, (1, 3))

    if method == "agglomerative":
        thr = st.slider("Distance threshold (cosine)", 0.10, 0.99, 0.75, 0.01,
                        help="Lower = tighter, more clusters.")
        k = st.number_input("Or fix cluster count (0 = use threshold)",
                            0, 500, 0, step=1)
    else:
        thr, k = 0.75, st.number_input("k (0 = heuristic)", 0, 500, 0, step=1)

    drop_legal = st.checkbox("Strip legal forms (PVT, LTD, CO...)", value=True)
    extra_sw = st.text_input("Extra stopwords (comma-separated)",
                             ", ".join(sorted(DEFAULT_STOPWORDS)))
    top_n = st.slider("Words per chart", 5, 25, 10)

cfg = ClusterConfig(
    analyzer=analyzer, ngram_min=ng_lo, ngram_max=ng_hi,
    method=method, distance_threshold=thr,
    n_clusters=int(k) or None, drop_legal=drop_legal,
    stopwords=tuple(w.strip().upper() for w in extra_sw.split(",") if w.strip()),
)

if not st.button("Run clustering", type="primary"):
    st.stop()


@st.cache_data(show_spinner="Clustering...")
def run(_df: pd.DataFrame, col: str, cfg_key: tuple, top_n: int):
    res = cluster(_df, col, ClusterConfig(*cfg_key))
    return (res.df, res.method_used, res.n_clusters, res.silhouette, res.notes,
            cluster_summary(res, col), cluster_word_table(res, top_n))


cfg_key = tuple(cfg.__dict__.values())
out_df, used, n_clusters, sil, notes, summary, words = run(df, column, cfg_key, top_n)

# ---------------------------------------------------------------------------
# 3. Results
# ---------------------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows clustered", f"{(out_df.cluster_id >= 0).sum():,}")
c2.metric("Clusters", n_clusters)
c3.metric("Algorithm", used)
c4.metric("Silhouette", f"{sil:.3f}" if sil is not None else "n/a")
for n in notes:
    st.warning(n)


def hbar(data: pd.DataFrame, title: str):
    return (alt.Chart(data, title=title)
            .mark_bar()
            .encode(x=alt.X("count:Q", title="Frequency"),
                    y=alt.Y("word:N", sort="-x", title=None),
                    tooltip=["word", "count"])
            .properties(height=28 * max(len(data), 1)))


st.subheader("Overall dataset - top words")
overall = word_counts(out_df["_normalized"], top_n)
st.altair_chart(hbar(overall, f"All {len(out_df):,} names"), width="stretch")

st.subheader("Clusters")
st.dataframe(summary, width="stretch", hide_index=True)

view = st.radio("Cluster view", ["Grid of charts", "Single cluster detail"],
                horizontal=True)

if view == "Grid of charts":
    max_show = st.slider("Clusters to chart (largest first)", 1,
                         min(60, max(1, n_clusters)), min(12, max(1, n_clusters)))
    ids = summary.cluster_id.head(max_show).tolist()
    cols = st.columns(2)
    for i, cid in enumerate(ids):
        d = words[words.cluster_id == cid]
        lab = summary.loc[summary.cluster_id == cid, "label"].iloc[0]
        size = summary.loc[summary.cluster_id == cid, "size"].iloc[0]
        with cols[i % 2]:
            st.altair_chart(hbar(d, f"Cluster {cid:02d} - {lab} (n={size})"),
                            width="stretch")
else:
    cid = st.selectbox("Cluster", summary.cluster_id.tolist(),
                       format_func=lambda c: f"{c:02d} - "
                       f"{summary.loc[summary.cluster_id == c, 'label'].iloc[0]}")
    st.altair_chart(hbar(words[words.cluster_id == cid], f"Cluster {cid:02d}"),
                    width="stretch")
    st.dataframe(out_df[out_df.cluster_id == cid].drop(columns=["_normalized"]),
                 width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# 4. Downloads
# ---------------------------------------------------------------------------

st.subheader("Download")
d1, d2, d3 = st.columns(3)
d1.download_button("Rows + cluster_id",
                   out_df.drop(columns=["_normalized"]).to_csv(index=False).encode(),
                   "clustered.csv", "text/csv")
d2.download_button("Cluster summary", summary.to_csv(index=False).encode(),
                   "cluster_summary.csv", "text/csv")
d3.download_button("Word frequencies", words.to_csv(index=False).encode(),
                   "cluster_word_counts.csv", "text/csv")