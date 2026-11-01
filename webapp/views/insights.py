"""Model Insights: a plain-language guide to the ideas behind the project."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from webapp import content, data_access as D
from webapp import inference
from webapp import results as R
from webapp import theme as T
from webapp.compat import show_df, show_plot

TFIDF_DEMO = (
    "The dress is lovely and fits well\n"
    "The dress is too small and the fabric is itchy\n"
    "Lovely colour but the dress is too long"
)


@st.cache_resource(show_spinner=False)
def _logreg_top_words(n: int = 12):
    """The words with the biggest weights in the saved Logistic Regression (None if the model is missing)."""
    ok, _ = inference.availability("logreg")
    if not ok:
        return None
    import joblib

    from src import classical_models as cm

    pipe = joblib.load(C.CLASSICAL_MODELS_DIR / "logreg.joblib")
    return cm.top_features(pipe.named_steps["tfidf"], pipe.named_steps["clf"], n=n)


def _concepts_tab() -> None:
    for item in content.CONCEPTS:
        with st.expander(item["title"], expanded=item["title"] == "Sentiment analysis"):
            st.markdown(item["simple"])
            st.markdown(f"**Everyday comparison:** {item['analogy']}")
            st.markdown(f"**In this project:** {item['project']}")

    T.section("Try TF-IDF yourself", "A live demo of the idea. It uses the sentences below, not our dataset.")
    text = st.text_area("One short review per line", value=TFIDF_DEMO, height=110, key="ins_tfidf_text")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:6]
    if len(lines) < 2:
        T.callout("info", "Add at least two lines", "TF-IDF compares words across several reviews.")
        return
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(lowercase=True)
    X = vec.fit_transform(lines).toarray()
    vocab = np.array(vec.get_feature_names_out())
    rows = []
    for i, line in enumerate(lines):
        order = np.argsort(X[i])[::-1][:4]
        top = ", ".join(f"{vocab[j]} ({X[i, j]:.2f})" for j in order if X[i, j] > 0)
        rows.append({"Review": line, "Highest TF-IDF words (score)": top})
    show_df(pd.DataFrame(rows))
    st.caption("Words that appear in every line (like 'the') get low scores. Words that appear in only one line (like 'itchy') score higher.")


def _metrics_tab() -> None:
    cols = st.columns(2)
    for i, m in enumerate(content.METRICS):
        with cols[i % 2]:
            T.card(m["title"], f'<p><code>{T.esc(m["formula"])}</code></p><p>{T.inline_md(m["simple"])}</p>')
            st.write("")

    T.section("Practise the formulas", "These numbers are made up to practise with. They are NOT results from our models.")
    cm = np.array([[30, 5, 5], [8, 20, 12], [4, 6, 90]], dtype=float)
    tp = np.diag(cm)
    precision, recall = tp / cm.sum(axis=0), tp / cm.sum(axis=1)
    f1 = 2 * precision * recall / (precision + recall)
    a, b = st.columns([1, 1.2])
    with a:
        df = pd.DataFrame(cm.astype(int), index=[f"True {n}" for n in C.LABEL_NAMES], columns=[f"Pred. {n}" for n in C.LABEL_NAMES])
        st.markdown("**Example confusion matrix (made-up numbers)**")
        show_df(df.reset_index().rename(columns={"index": ""}))
    with b:
        out = pd.DataFrame({"Class": [n.capitalize() for n in C.LABEL_NAMES], "Precision": precision.round(3),
                            "Recall": recall.round(3), "F1": f1.round(3)})
        show_df(out)
        st.markdown(f"Accuracy = {tp.sum():.0f} / {cm.sum():.0f} = **{tp.sum() / cm.sum():.3f}**  \n"
                    f"Macro F1 = ({f1[0]:.3f} + {f1[1]:.3f} + {f1[2]:.3f}) / 3 = **{f1.mean():.3f}**")
    st.caption("Notice how accuracy looks good while the neutral class is weaker. That is why we use macro F1.")


def _compare_tab() -> None:
    rows = pd.DataFrame(content.COMPARISON_ROWS, columns=["", "Classical ML (TF-IDF + linear model)", "Transformer (DistilBERT)"])
    show_df(rows)
    test = D.get("test_results")
    split, table = R.score_table(test, None, None)
    if split == "test":
        T.section("Measured in this project", "From results/test_results.json.")
        cost = R.cost_table(test)
        merged = table[["key", "model", "accuracy", "macro_f1"]].merge(cost[["key", "train_seconds", "inference_ms_per_review", "model_size_mb"]], on="key", how="left")
        merged["Accuracy"] = merged["accuracy"].map(lambda v: f"{v:.3f}")
        merged["Macro F1"] = merged["macro_f1"].map(lambda v: f"{v:.3f}")
        merged["Training time"] = merged["train_seconds"].map(lambda v: R.human_time(None if pd.isna(v) else v))
        merged["Per review"] = merged["inference_ms_per_review"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.3g} ms")
        merged["Size"] = merged["model_size_mb"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.3g} MB")
        show_df(merged[["model", "Accuracy", "Macro F1", "Training time", "Per review", "Size"]].rename(columns={"model": "Model"}))
    else:
        T.callout("pending", "Measured numbers will appear here", "They come from the test-set evaluation (see the Model Comparison page).")


def _learned_tab() -> None:
    T.section("What Logistic Regression learned", "The words with the largest weights for each class, read from the saved model.")
    try:
        top = _logreg_top_words()
    except Exception as exc:  # corrupt or incompatible file: say so, keep the page alive
        top = None
        T.callout("warn", "The saved Logistic Regression could not be read", T.esc(f"{type(exc).__name__}: {exc}"))
    if top is None:
        T.callout("pending", "The Logistic Regression model is not available", "Run notebooks/02_classical_models.ipynb to create it.")
    else:
        cols = st.columns(3)
        for col, name in zip(cols, C.LABEL_NAMES):
            with col:
                chips = " ".join(f'<span class="esa-badge" style="background:{T.SENTIMENT_TINT[name]};color:{T.SENTIMENT_TEXT[name]};border-color:{T.SENTIMENT_COLORS[name]};margin:2px 2px 2px 0">{T.esc(w)}</span>' for w in top[name])
                st.markdown(f'<div class="esa-card"><h4>{name.capitalize()}</h4>{chips}</div>', unsafe_allow_html=True)
        st.caption("This is the benefit of a simple model: you can see which words push a review towards each class.")

    T.section("How DistilBERT training went", "Validation macro F1 after each epoch, for each setting that was tried.")
    hist = D.get("distilbert_history")
    if hist is None or hist.empty:
        log = D.get("training_log")
        extra = f" Latest log line: <code>{T.esc([l for l in log.splitlines() if l.strip()][-1])}</code>" if log and log.strip() else ""
        T.callout("pending", "No DistilBERT training history yet", "It is written when notebooks/03_distilbert.ipynb finishes." + extra)
    else:
        fig = go.Figure()
        for run, part in hist.groupby("run"):
            name = f"Run {run}: lr={part['learning_rate'].iloc[0]}, class weights={part['class_weighted'].iloc[0]}"
            fig.add_trace(go.Scatter(x=part["epoch"], y=part["macro_f1"], mode="lines+markers", name=name))
        fig.update_layout(**T.chart_layout("Validation macro F1 by epoch", height=360))
        fig.update_xaxes(title_text="Epoch", dtick=1)
        fig.update_yaxes(title_text="Macro F1")
        T.style_axes(fig)
        show_plot(fig, key="ins_curves")
        search = D.get("distilbert_search")
        if search is not None and not search.empty:
            show_df(search)

    csum = D.get("classical_summary")
    if csum and csum.get("best_params"):
        T.section("Settings chosen for the classical models", "Chosen on the validation split only.")
        rows = [{"Model": C.CLASSICAL_MODEL_NAMES.get(k, k), "Chosen settings": ", ".join(f"{a} = {b}" for a, b in v.items())}
                for k, v in csum["best_params"].items()]
        show_df(pd.DataFrame(rows))


def _brief_tab() -> None:
    T.section("Problem statement")
    st.markdown(content.PROBLEM_STATEMENT)
    T.section("Objectives")
    st.markdown("\n".join(f"{i}. {o}" for i, o in enumerate(content.OBJECTIVES, 1)))
    T.section("Methodology")
    show_df(pd.DataFrame(content.METHODOLOGY, columns=["Step", "What we did"]))
    T.section("Limitations", "Be ready to explain these in the viva. Honest limits make a project stronger.")
    st.markdown("\n".join(f"- {x}" for x in content.LIMITATIONS))
    T.section("Future scope")
    st.markdown("\n".join(f"- {x}" for x in content.FUTURE_SCOPE))


def render() -> None:
    T.page_header(
        "Model Insights",
        "A plain-language guide to every idea used in this project, so you can explain it to anyone, including in the viva.",
    )
    tabs = st.tabs(["Concepts", "Metrics", "Classical vs transformer", "What the models learned", "Project brief"])
    with tabs[0]:
        _concepts_tab()
    with tabs[1]:
        _metrics_tab()
    with tabs[2]:
        _compare_tab()
    with tabs[3]:
        _learned_tab()
    with tabs[4]:
        _brief_tab()
