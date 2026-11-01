"""Small text statistics for the Data Explorer (pure functions, easy to test)."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z']+\b"


def top_words(texts: Sequence[str], labels: Sequence[str], label_names: Sequence[str], n: int = 12, min_df: int = 5) -> Dict[str, dict]:
    """For each label: the most frequent words and the words most typical of that label.

    Words are counted once per review (a review that says 'love' five times still counts as one review).
    English stop words are removed here, for exploration only. The models themselves keep them.
    'Typical' = how many times more often the word appears in this label's reviews than in all reviews
    (only words seen in at least 15 reviews of the label are considered, to avoid rare accidents).
    """
    texts = list(texts)
    labels = np.asarray(list(labels))
    vec = CountVectorizer(binary=True, stop_words="english", token_pattern=TOKEN_PATTERN, min_df=min_df, lowercase=True)
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())
    overall_share = np.asarray(X.sum(axis=0)).ravel() / max(1, X.shape[0])
    out: Dict[str, dict] = {}
    for name in label_names:
        mask = labels == name
        n_docs = int(mask.sum())
        if n_docs == 0:
            out[name] = {"frequent": [], "typical": []}
            continue
        freq = np.asarray(X[mask].sum(axis=0)).ravel()
        share = freq / n_docs
        lift = share / np.maximum(overall_share, 1e-12)
        eligible = freq >= 15
        lift_masked = np.where(eligible, lift, -np.inf)
        frequent_idx = np.argsort(freq)[::-1][:n]
        typical_idx = [i for i in np.argsort(lift_masked)[::-1][:n] if np.isfinite(lift_masked[i])]
        out[name] = {
            "frequent": [(str(vocab[i]), int(freq[i])) for i in frequent_idx],
            "typical": [(str(vocab[i]), float(lift[i]), int(freq[i])) for i in typical_idx],
            "n_reviews": n_docs,
        }
    return out


def length_stats(words: Sequence[int], labels: Sequence[str], label_names: Sequence[str]) -> list:
    """Median, mean and 90th percentile review length (in words) for each label."""
    words = np.asarray(list(words), dtype=float)
    labels = np.asarray(list(labels))
    rows = []
    for name in label_names:
        w = words[labels == name]
        if len(w) == 0:
            continue
        rows.append({"label": name, "reviews": int(len(w)), "median": float(np.median(w)),
                     "mean": float(w.mean()), "p90": float(np.percentile(w, 90))})
    return rows
