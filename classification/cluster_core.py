"""
cluster_core.py - UI-independent engine for company-name clustering
and word-frequency analysis. No Streamlit imports here on purpose:
the same module is used by app.py and by the CLI.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

# Legal-form and boilerplate tokens: removed before clustering so that
# "X (PVT) LTD" and "X INDUSTRIES" are not pulled together by their suffix.
LEGAL_TOKENS = {
    ")",
    "(",
    "-",
    "AL-",
    "AL ",
    "(PVT.)",
    "(SMC-PVT)",
    "(PVT)",
    "PVT",
    "PVT.",
    "PRIVATE",
    "LTD.",
    "LTD",
    "LIMITED",
    "LIMTED",
    "SMC",
    "CO",
    "COMPANY",
    "INC",
    "CORP",
    "CORPORATION",
    "LLC",
    "LLP",
    "PLC",
    "AND",
    "THE",
    "OF",
}
# Tokens that are location noise in this dataset; editable from the UI.
DEFAULT_STOPWORDS = {"LAHORE", "PAKISTAN", "PUNJAB"}

RE_NON_ALNUM = re.compile(r"[^A-Z0-9 ]+")
RE_WS = re.compile(r"\s+")


def normalize(name: str, stopwords: set[str], drop_legal: bool = True) -> str:
    """Upper-case, strip punctuation, drop legal forms and stopwords."""
    if not isinstance(name, str):
        return ""
    s = RE_NON_ALNUM.sub(" ", name.upper())
    toks = [t for t in RE_WS.split(s) if t]
    drop = set(stopwords) | (LEGAL_TOKENS if drop_legal else set())
    toks = [t for t in toks if t not in drop and len(t) > 1]
    return " ".join(toks)


def word_counts(series: pd.Series, top_n: int = 10) -> pd.DataFrame:
    c = Counter()
    for s in series.dropna():
        c.update(t for t in str(s).split() if t)
    return pd.DataFrame(c.most_common(top_n), columns=["word", "count"])


def word_counts_min_freq(series: pd.Series, min_count: int = 20) -> pd.DataFrame:
    """All words with frequency > min_count, sorted descending. No fixed cap."""
    c = Counter()
    for s in series.dropna():
        c.update(t for t in str(s).split() if t)
    items = [(w, n) for w, n in c.items() if n > min_count]
    items.sort(key=lambda x: x[1], reverse=True)
    return pd.DataFrame(items, columns=["word", "count"])


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


@dataclass
class ClusterConfig:
    analyzer: str = "char_wb"  # "char_wb" (typo-tolerant) or "word"
    ngram_min: int = 3
    ngram_max: int = 4
    min_df: int = 1
    max_features: int = 50_000
    method: str = "agglomerative"  # "agglomerative" | "kmeans"
    distance_threshold: float = 0.75  # agglomerative, cosine distance 0..1
    n_clusters: int | None = None  # kmeans, or agglomerative override
    drop_legal: bool = True
    stopwords: tuple = tuple(sorted(DEFAULT_STOPWORDS))
    auto_switch_rows: int = 8_000  # above this, agglomerative -> kmeans


@dataclass
class ClusterResult:
    df: pd.DataFrame  # original rows + _normalized + cluster_id
    labels: np.ndarray
    vectorizer: TfidfVectorizer
    matrix: object  # sparse TF-IDF matrix
    method_used: str
    n_clusters: int
    silhouette: float | None
    notes: list


def vectorize(texts, cfg: ClusterConfig):
    vec = TfidfVectorizer(
        analyzer=cfg.analyzer,
        ngram_range=(cfg.ngram_min, cfg.ngram_max),
        min_df=cfg.min_df,
        max_features=cfg.max_features,
        sublinear_tf=True,
    )
    return vec, vec.fit_transform(texts)


def cluster(df: pd.DataFrame, column: str, cfg: ClusterConfig) -> ClusterResult:
    notes = []
    work = df.copy()
    sw = set(cfg.stopwords)
    work["_normalized"] = work[column].map(lambda v: normalize(v, sw, cfg.drop_legal))

    usable = work["_normalized"].str.len() > 0
    if (~usable).any():
        notes.append(
            f"{(~usable).sum()} row(s) empty after normalisation -> cluster_id = -1"
        )

    texts = work.loc[usable, "_normalized"].tolist()
    if not texts:
        raise ValueError(f"Column {column!r} produced no usable text.")

    vec, X = vectorize(texts, cfg)

    method = cfg.method
    if method == "agglomerative" and len(texts) > cfg.auto_switch_rows:
        method = "kmeans"
        notes.append(
            f"{len(texts)} rows exceeds auto_switch_rows={cfg.auto_switch_rows}; "
            "switched to MiniBatchKMeans (agglomerative is O(n^2) in memory)."
        )

    if method == "agglomerative":
        model = AgglomerativeClustering(
            n_clusters=cfg.n_clusters,
            distance_threshold=None if cfg.n_clusters else cfg.distance_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = model.fit_predict(X.toarray())
    else:
        k = cfg.n_clusters or max(2, min(60, int(np.sqrt(len(texts) / 2))))
        model = MiniBatchKMeans(n_clusters=k, n_init=5, random_state=0, batch_size=1024)
        labels = model.fit_predict(X)
        if not cfg.n_clusters:
            notes.append(f"k not given; used heuristic k={k}.")

    sil = None
    n_found = len(set(labels))
    if 1 < n_found < len(texts):
        idx = np.arange(len(texts))
        if len(idx) > 5000:  # subsample: silhouette is O(n^2)
            idx = np.random.default_rng(0).choice(idx, 5000, replace=False)
        sil = float(silhouette_score(X[idx], labels[idx], metric="cosine"))

    work["cluster_id"] = -1
    work.loc[usable, "cluster_id"] = labels

    # relabel so cluster 0 is the largest group (stable, readable output)
    order = work.loc[usable, "cluster_id"].value_counts().index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    work.loc[usable, "cluster_id"] = work.loc[usable, "cluster_id"].map(remap)

    return ClusterResult(work, labels, vec, X, method, n_found, sil, notes)


# ---------------------------------------------------------------------------
# Word frequency
# ---------------------------------------------------------------------------


def word_counts(series: pd.Series, top_n: int = 10) -> pd.DataFrame:
    c = Counter()
    for s in series.dropna():
        c.update(t for t in str(s).split() if t)
    return pd.DataFrame(c.most_common(top_n), columns=["word", "count"])


def cluster_word_table(res: ClusterResult, top_n: int = 10) -> pd.DataFrame:
    """Long-form table: cluster_id, size, rank, word, count. Chart-ready."""
    rows = []
    for cid, grp in res.df[res.df.cluster_id >= 0].groupby("cluster_id"):
        wc = word_counts(grp["_normalized"], top_n)
        for rank, (w, n) in enumerate(zip(wc["word"], wc["count"]), start=1):
            rows.append(
                {
                    "cluster_id": cid,
                    "size": len(grp),
                    "rank": rank,
                    "word": w,
                    "count": int(n),
                }
            )
    return pd.DataFrame(rows)


def cluster_summary(
    res: ClusterResult, column: str, label_words: int = 3
) -> pd.DataFrame:
    """One row per cluster: size, auto label from its top words, 3 examples."""
    rows = []
    for cid, grp in res.df[res.df.cluster_id >= 0].groupby("cluster_id"):
        top = word_counts(grp["_normalized"], label_words)["word"].tolist()
        rows.append(
            {
                "cluster_id": cid,
                "size": len(grp),
                "label": " / ".join(top) if top else "(unlabelled)",
                "examples": " | ".join(grp[column].astype(str).head(3)),
            }
        )
    return pd.DataFrame(rows).sort_values("size", ascending=False, ignore_index=True)
