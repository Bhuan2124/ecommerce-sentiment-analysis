"""About Project: dataset, pipeline, technology, file status and how to run."""
from __future__ import annotations

import platform
import sys
from importlib import metadata

import pandas as pd
import streamlit as st

from src import config as C
from webapp import artifacts as A
from webapp import content
from webapp import theme as T
from webapp.compat import show_df

PIPELINE = [
    ("1. Raw data", "Women's E-Commerce Clothing Reviews (Kaggle CSV)"),
    ("2. Clean", "Drop empty text, map stars to labels, remove duplicates"),
    ("3. Split", "70% train, 15% validation, 15% test (fixed seed, stratified)"),
    ("4. Train", "TF-IDF + 3 classical models, and fine-tuned DistilBERT"),
    ("5. Evaluate", "Score every model once on the test split"),
    ("6. Dashboard", "This Streamlit app loads saved models and results"),
]


def _version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "not installed"


def render() -> None:
    T.page_header("About This Project", "What was built, with which data and tools, and how to run every part of it.")

    st.markdown(
        '<div class="esa-card"><h4>Sentiment Analysis of E-Commerce Reviews: Classical ML vs. Transformer (DistilBERT)</h4>'
        "<p>Online shops get thousands of reviews. This project builds four sentiment classifiers that label a review as "
        "negative, neutral or positive, and compares them on accuracy and on <b>cost</b>: training time, speed, size and parameters. "
        "Three are classical (TF-IDF with Logistic Regression, Multinomial Naive Bayes and Linear SVM). One is a fine-tuned DistilBERT transformer.</p></div>",
        unsafe_allow_html=True,
    )

    T.section("Pipeline")
    steps = "".join(f'<div class="esa-step"><b>{T.esc(a)}</b><span>{T.esc(b)}</span></div>' for a, b in PIPELINE)
    st.markdown(f'<div class="esa-flow">{steps}</div>', unsafe_allow_html=True)

    T.section("Dataset")
    a, b = st.columns([1.3, 1])
    with a:
        T.card(
            "Women's E-Commerce Clothing Reviews",
            "<p>Source: Kaggle, uploaded by <b>nicapotato</b> (<code>nicapotato/womens-ecommerce-clothing-reviews</code>). "
            "Licence: CC0 Public Domain. About 23,000 anonymised customer reviews written in English, "
            "each with a title, text, star rating and some product information.</p>"
            "<p><b>Labels are derived from the star rating</b> (see the table). This is a common shortcut, but it has limits: "
            "a 3-star review can be mostly positive, a 4-star review can contain complaints, and people use stars differently. "
            "So some of the model's 'mistakes' may really be label noise.</p>",
        )
    with b:
        show_df(pd.DataFrame(
            [("1 or 2 stars", "Negative"), ("3 stars", "Neutral"), ("4 or 5 stars", "Positive")], columns=["Star rating", "Label"]))

    T.section("Technology")
    show_df(pd.DataFrame(content.TECH_STACK, columns=["Part", "Tool"]))

    T.section("Project files", "Which results and models exist right now. A missing item simply means that step has not been run yet.")
    inv = A.inventory()
    found = int((inv["found"] == "yes").sum())
    st.caption(f"{found} of {len(inv)} items found.")
    show_df(inv)

    T.section("Software versions on this computer")
    libs = ["streamlit", "plotly", "pandas", "numpy", "scikit-learn", "torch", "transformers"]
    rows = [("Python", platform.python_version())] + [(p, _version(p)) for p in libs]
    show_df(pd.DataFrame(rows, columns=["Package", "Version"]))

    T.section("How to run it", "Open the project folder in VS Code, then use these commands in the terminal.")
    st.code(
        "python -m venv .venv\n"
        ".venv\\Scripts\\activate\n"
        "pip install -r requirements-app.txt\n"
        "\n"
        "# 1. (once) score every saved model on the test set\n"
        "python scripts/evaluate_models.py\n"
        "\n"
        "# 2. run the checks\n"
        "python -m pytest\n"
        "\n"
        "# 3. start the dashboard\n"
        "python -m streamlit run app.py",
        language="bash",
    )
