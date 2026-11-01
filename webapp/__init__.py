"""Streamlit dashboard for the e-commerce sentiment analysis project (Phase 5).

Package layout
    theme.py       colours, CSS and small HTML helpers (one look for every page)
    compat.py      tiny wrappers so charts/tables work across Streamlit versions
    artifacts.py   reads the files the earlier phases wrote (never trains anything)
    results.py     turns those files into the tables the pages show
    inference.py   loads saved models and predicts (used by the Sentiment Analyzer)
    history.py     remembers recent predictions in a small CSV file
    content.py     the plain-language teaching text
    views/         one file per page
"""
