"""Overview Dashboard: the landing page."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src import config as C
from webapp import data_access as D
from webapp import history, inference
from webapp import results as R
from webapp import theme as T
from webapp.compat import show_df, show_plot


def goto(page: str) -> None:
    """Button callback: switch the sidebar to another page."""
    st.session_state["nav"] = page


def _distribution_chart(dist, total):
    fig = go.Figure(
        go.Pie(
            labels=[n.capitalize() for n in dist["label"]],
            values=dist["count"],
            hole=0.64,
            sort=False,
            marker=dict(colors=[T.SENTIMENT_COLORS[n] for n in dist["label"]], line=dict(color="#FFFFFF", width=3)),
            textinfo="percent",
            textfont=dict(size=13, color="#FFFFFF"),
            hovertemplate="%{label}: %{value:,} reviews (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(**T.chart_layout("Sentiment distribution", height=330, legend=True))
    fig.add_annotation(text=f"<b>{total:,}</b><br>reviews", x=0.5, y=0.5, showarrow=False, font=dict(size=15, color=T.TEXT))
    return fig


def _performance_chart(table, split, baseline):
    fig = go.Figure(
        go.Bar(
            x=table["model"],
            y=table["macro_f1"],
            marker_color=[T.MODEL_COLORS[k] for k in table["key"]],
            text=[f"{v:.3f}" for v in table["macro_f1"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>Macro F1: %{y:.3f}<extra></extra>",
        )
    )
    title = "Macro F1-score by model (" + ("test set" if split == "test" else "validation set") + ")"
    fig.update_layout(**T.chart_layout(title, height=330, legend=False))
    fig.update_yaxes(range=[0, 1.05], title_text="Macro F1")
    T.style_axes(fig)
    if baseline is not None:
        fig.add_hline(y=baseline, line_dash="dash", line_color=T.MUTED, annotation_text="Always guess 'positive'",
                      annotation_position="top left", annotation_font_color=T.MUTED)
    return fig


def _baseline(test, val_df, split):
    if split == "test" and test and test.get("baseline"):
        return test["baseline"].get("macro_f1")
    if val_df is not None and "model" in val_df.columns:
        row = val_df[val_df["model"] == R.BASELINE_NAME]
        if not row.empty:
            return float(row.iloc[0]["macro_f1"])
    return None


def render() -> None:
    dsum = D.get("dataset_summary")
    test = D.get("test_results")
    val_df = D.get("classical_val")
    db_sum = D.get("distilbert_summary")
    dist = R.class_distribution(dsum)
    total = R.total_reviews(dsum)
    split, table = R.score_table(test, val_df, db_sum)
    best = R.best_model(table)

    # ---- hero
    st.markdown(
        '<div class="esa-hero"><div class="esa-eyebrow">College project</div>'
        '<div class="esa-title">Sentiment Analysis of E-Commerce Reviews</div>'
        '<div class="esa-subtitle" style="margin-bottom:0">Classical machine learning (TF-IDF with Logistic Regression, '
        "Naive Bayes and Linear SVM) compared with a fine-tuned DistilBERT transformer. "
        "The system reads a customer review and decides whether it is negative, neutral or positive.</div></div>",
        unsafe_allow_html=True,
    )
    b1, b2, _ = st.columns([1.2, 1.2, 3])
    with b1:
        st.button("Open the Sentiment Analyzer", type="primary", on_click=goto, args=("Sentiment Analyzer",), key="ov_go_analyzer")
    with b2:
        st.button("See model comparison", on_click=goto, args=("Model Comparison",), key="ov_go_compare")

    # ---- dataset numbers
    T.section("Dataset", "Counts come from results/dataset_summary.json (written when the data was prepared).")
    if dist is None or total is None:
        T.callout(
            "warn", "The dataset summary was not found",
            "Run <code>notebooks/01_data_exploration.ipynb</code> first. It cleans the Kaggle file and writes "
            "<code>results/dataset_summary.json</code> and the train / validation / test files.",
        )
    else:
        pct = {r.label: r.percent for r in dist.itertuples()}
        T.kpi_row(
            [
                ("Total reviews", f"{total:,}", "after cleaning and removing duplicates", T.ACCENT),
                ("Positive", f"{pct['positive']:.1f}%", "4-5 star reviews", T.POSITIVE),
                ("Neutral", f"{pct['neutral']:.1f}%", "3 star reviews", T.NEUTRAL),
                ("Negative", f"{pct['negative']:.1f}%", "1-2 star reviews", T.NEGATIVE),
            ]
        )

    # ---- model numbers
    T.section("Models", "Ranked by macro F1-score, the fairest single number when the classes are unbalanced.")
    n_saved = sum(inference.available_models().values())
    if best is None:
        split_note, best_name, best_f1 = "", "Pending", "n/a"
    else:
        best_name, best_f1 = best["model"], f"{best['macro_f1']:.3f}"
        split_note = "final test set" if split == "test" else "validation set (test evaluation not run yet)"
    T.kpi_row(
        [
            ("Best model", best_name, split_note or "no results yet", T.ACCENT),
            ("Macro F1-score", best_f1, split_note or "no results yet", T.ACCENT),
            ("Models with results", f"{len(table)} of 4", "Logistic Regression, Naive Bayes, Linear SVM, DistilBERT", T.MUTED),
            ("Saved models ready", f"{n_saved} of 4", "available in the Sentiment Analyzer", T.MUTED),
        ]
    )

    if split == "validation":
        T.callout(
            "pending", "Showing validation scores for now",
            "The final comparison on the untouched test set has not been run yet. Run <code>scripts/evaluate_models.py</code> "
            "(VS Code task: <em>ESA: 3 Evaluate models on test set</em>). These numbers will then be replaced automatically.",
        )
    if "distilbert" not in set(table["key"]):
        log = D.get("training_log")
        last = [ln for ln in (log or "").splitlines() if ln.strip()][-1:] or []
        extra = f" Latest training log line: <code>{T.esc(last[0])}</code>" if last else ""
        T.callout(
            "pending", "DistilBERT results are pending",
            "The DistilBERT model has not finished training, or its results file is not written yet. "
            "It will appear here automatically once <code>results/distilbert_summary.json</code> exists." + extra,
        )

    # ---- charts
    left, right = st.columns([1, 1.35])
    with left:
        if dist is not None and total is not None:
            show_plot(_distribution_chart(dist, total), key="ov_dist")
        else:
            T.callout("pending", "No distribution to show yet", "The dataset summary file is missing.")
    with right:
        if best is not None:
            show_plot(_performance_chart(table, split, _baseline(test, val_df, split)), key="ov_perf")
        else:
            T.callout("pending", "No model results yet", "Run notebooks 02 and 03 (training), then the evaluation script.")

    # ---- recent predictions
    T.section("Recent predictions", "The last reviews analysed on the Sentiment Analyzer page.")
    recent = history.recent(8)
    if recent.empty:
        T.callout("info", "No predictions yet", "Open the Sentiment Analyzer, type a review and press Analyze.")
    else:
        view = recent.copy()
        view["confidence"] = view["confidence"].map(lambda v: "" if v in ("", None) else f"{float(v) * 100:.1f}%")
        view["prediction"] = view["prediction"].str.capitalize()
        show_df(view.rename(columns={"time": "Time", "model": "Model", "prediction": "Prediction",
                                     "confidence": "Confidence", "review": "Review"}))
