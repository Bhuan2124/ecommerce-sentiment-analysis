"""Cached access to the result files and datasets.

The cache key includes the file's modification time, so when a new run rewrites a file the page
picks up the new numbers on the next interaction (no restart needed).
"""
from __future__ import annotations

import streamlit as st

from webapp import artifacts as A

# friendly name -> (loader function in artifacts.py, artifact whose file-time decides when to refresh)
_LOADERS = {
    "dataset_summary": ("dataset_summary", "dataset_summary"),
    "classical_val": ("classical_val", "classical_val"),
    "classical_summary": ("classical_summary", "classical_summary"),
    "classical_tuning": ("classical_tuning", "classical_tuning"),
    "distilbert_summary": ("distilbert_summary", "distilbert_summary"),
    "distilbert_search": ("distilbert_search", "distilbert_search"),
    "distilbert_history": ("distilbert_history", "distilbert_history"),
    "training_log": ("training_log", "training_log"),
    "test_results": ("test_results", "test_results"),
}


@st.cache_data(show_spinner=False)
def _call(loader_name: str, stamp: int):
    return getattr(A, loader_name)()


def get(name: str):
    """Return the named result (a dict, a table or text), or None if it does not exist yet."""
    loader, artifact = _LOADERS[name]
    return _call(loader, A.mtime_key(artifact))


@st.cache_data(show_spinner=False)
def _split(name: str, stamp: int):
    return A.load_split(name)


def split(name: str):
    """'train', 'val' or 'test' as a table, or None."""
    return _split(name, A.mtime_key(name))


@st.cache_data(show_spinner=False)
def _raw(stamp: int):
    return A.load_raw()


def raw():
    """The original CSV from Kaggle (unchanged), or None."""
    p = A.find_raw_csv()
    stamp = p.stat().st_mtime_ns if p else 0
    return _raw(stamp)
