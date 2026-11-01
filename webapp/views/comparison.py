"""Model Comparison: all models scored on the same untouched test set (Phase 4 results)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from webapp import content, data_access as D
from webapp import results as R
from webapp import theme as T
from webapp.compat import show_df, show_plot


def _f3(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def _bar(table: pd.DataFrame, col: str, title: str, y_title: str, log: bool = False, fmt=None, key=None) -> None:
    table = table[table[col].notna()]
    if table.empty:
        T.callout("pending", title, "Not measured yet.")
        return
    fmt = fmt or (lambda v: f"{v:.3f}")
    fig = go.Figure(
        go.Bar(
            x=table["model"], y=table[col], marker_color=[T.MODEL_COLORS[k] for k in table["key"]],
            text=[fmt(v) for v in table[col]], textposition="outside", cliponaxis=False,
            hovertemplate="%{x}<br>" + y_title + ": %{text}<extra></extra>",
        )
    )
    fig.update_layout(**T.chart_layout(title, height=320, legend=False))
    T.style_axes(fig)
    if log:
        fig.update_yaxes(type="log", title_text=y_title + " (log scale)")
    else:
        fig.update_yaxes(title_text=y_title, range=[0, 1.08] if table[col].max() <= 1.0 else None)
    show_plot(fig, key=key)


def _use_log(series: pd.Series) -> bool:
    s = series.dropna()
    s = s[s > 0]
    return len(s) > 1 and s.max() / s.min() > 20


def _pending() -> None:
    T.callout(
        "pending", "The final test-set comparison has not been run yet",
        "This page shows the results of scoring every model <b>once</b> on the untouched test split. To create them, run "
        "<code>python scripts/evaluate_models.py</code> (VS Code task: <em>ESA: 3 Evaluate models on test set</em>). "
        "It loads the saved models, so it needs the DistilBERT training to have finished for DistilBERT to be included.",
    )
    split, table = R.score_table(None, D.get("classical_val"), D.get("distilbert_summary"))
    if split == "validation":
        T.section("Preview: validation scores", "Validation scores were used to choose settings. They are shown only until the test run exists.")
        view = table[["model", "accuracy", "macro_f1"]].rename(columns={"model": "Model", "accuracy": "Accuracy", "macro_f1": "Macro F1"})
        view["Accuracy"] = view["Accuracy"].map(_f3)
        view["Macro F1"] = view["Macro F1"].map(_f3)
        show_df(view)
    else:
        T.callout("info", "No model results exist yet", "Train the models in notebooks 02 and 03 first.")


def _confusion_figs(cm: np.ndarray, name: str):
    labels = [n.capitalize() for n in C.LABEL_NAMES]
    pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
    scale = [[0, "#FFFFFF"], [1, T.ACCENT]]
    figs = []
    for z, text, title in ((cm, cm.astype(int).astype(str), f"{name}: number of reviews"),
                           (pct, np.vectorize(lambda v: f"{v:.1f}%")(pct), f"{name}: % of each true class")):
        fig = go.Figure(
            go.Heatmap(z=z, x=[f"Pred. {l}" for l in labels], y=[f"True {l}" for l in labels], colorscale=scale,
                       showscale=False, text=text, texttemplate="%{text}", textfont=dict(size=15),
                       hovertemplate="%{y}, %{x}: %{text}<extra></extra>", xgap=3, ygap=3)
        )
        fig.update_layout(**T.chart_layout(title, height=360, legend=False))
        fig.update_yaxes(autorange="reversed")
        figs.append(fig)
    return figs


def _report_table(entry: dict) -> pd.DataFrame:
    rows = []
    for label in C.LABEL_NAMES:
        pc = entry["per_class"][label]
        rows.append({"": label.capitalize(), "Precision": _f3(pc["precision"]), "Recall": _f3(pc["recall"]),
                     "F1-score": _f3(pc["f1"]), "Reviews": int(pc["support"])})
    n = int(sum(entry["per_class"][l]["support"] for l in C.LABEL_NAMES))
    m = entry["metrics"]
    rows.append({"": "Macro average", "Precision": _f3(m.get("macro_precision")), "Recall": _f3(m.get("macro_recall")),
                 "F1-score": _f3(m.get("macro_f1")), "Reviews": n})
    w = entry.get("weighted")
    if w:
        rows.append({"": "Weighted average", "Precision": _f3(w.get("precision")), "Recall": _f3(w.get("recall")),
                     "F1-score": _f3(w.get("f1")), "Reviews": n})
    rows.append({"": "Accuracy", "Precision": "", "Recall": "", "F1-score": _f3(m["accuracy"]), "Reviews": n})
    return pd.DataFrame(rows)


def _tradeoff_summary(table: pd.DataFrame, cost: pd.DataFrame) -> str:
    cl = table[table["family"] == "classical"]
    tr = table[table["family"] == "transformer"]
    if cl.empty or tr.empty:
        return ""
    best_cl = cl.sort_values("macro_f1", ascending=False).iloc[0]
    t = tr.iloc[0]
    diff = (t["macro_f1"] - best_cl["macro_f1"]) * 100
    text = (f"On this test set DistilBERT's macro F1 is <b>{t['macro_f1']:.3f}</b> and the best classical model "
            f"({T.esc(best_cl['model'])}) scores <b>{best_cl['macro_f1']:.3f}</b>, a difference of {diff:+.1f} points. ")
    c_cost = cost[cost["key"] == best_cl["key"]]
    t_cost = cost[cost["key"] == t["key"]]
    if not c_cost.empty and not t_cost.empty:
        c, d = c_cost.iloc[0], t_cost.iloc[0]
        if pd.notna(c["train_seconds"]) and pd.notna(d["train_seconds"]):
            text += f"Training took {R.human_time(d['train_seconds'])} for DistilBERT versus {R.human_time(c['train_seconds'])}. "
        if pd.notna(c["model_size_mb"]) and pd.notna(d["model_size_mb"]):
            text += f"The saved model is {d['model_size_mb']:.0f} MB versus {c['model_size_mb']:.1f} MB. "
        if pd.notna(c["inference_ms_per_review"]) and pd.notna(d["inference_ms_per_review"]):
            text += (f"Answering one review takes about {d['inference_ms_per_review']:.2f} ms versus "
                     f"{c['inference_ms_per_review']:.3f} ms ({T.esc(str(d['inference_device']))} versus {T.esc(str(c['inference_device']))}). ")
    text += "With a single test split, differences of a point or two may be noise."
    return text


def render() -> None:
    T.page_header(
        "Model Comparison",
        "All models were scored on the same held-out test reviews. Every number on this page was measured in this project.",
    )
    test = D.get("test_results")
    if test is None:
        _pending()
        return

    ev = R.evaluated_models(test)
    split, table = R.score_table(test, None, None)
    cost = R.cost_table(test)
    st.caption(
        f"Test set: {int(test['n_test']):,} reviews. Evaluated {test.get('created_at', 'at an unknown time')}. "
        f"Classes: {', '.join(test['labels'])}."
    )

    for key in R.KEYS:
        if key not in ev:
            entry = test.get("models", {}).get(key, {})
            reason = entry.get("reason") or "It was not evaluated in the last run."
            T.callout("pending", f"{R.DISPLAY[key]}: metrics unavailable", T.esc(reason))
    if table.empty:
        return

    best = R.best_model(table)
    top_acc = table.sort_values("accuracy", ascending=False).iloc[0]
    fastest = cost.dropna(subset=["inference_ms_per_review"]).sort_values("inference_ms_per_review").head(1)
    smallest = cost.dropna(subset=["model_size_mb"]).sort_values("model_size_mb").head(1)
    T.kpi_row(
        [
            ("Best macro F1", best["model"], f"macro F1 = {best['macro_f1']:.3f}", T.ACCENT),
            ("Highest accuracy", top_acc["model"], f"accuracy = {top_acc['accuracy']:.3f}", T.ACCENT),
            ("Fastest per review", fastest.iloc[0]["model"] if not fastest.empty else "n/a",
             f"{fastest.iloc[0]['inference_ms_per_review']:.3f} ms" if not fastest.empty else "", T.POSITIVE),
            ("Smallest on disk", smallest.iloc[0]["model"] if not smallest.empty else "n/a",
             f"{smallest.iloc[0]['model_size_mb']:.2f} MB" if not smallest.empty else "", T.POSITIVE),
        ]
    )

    tabs = st.tabs(["Scores", "Speed and size", "Per-class results", "Confusion matrices", "Classification reports", "Trade-offs"])

    # ---- scores
    with tabs[0]:
        view = table.rename(columns={"model": "Model", "accuracy": "Accuracy", "macro_precision": "Macro precision",
                                     "macro_recall": "Macro recall", "macro_f1": "Macro F1"})
        show = view[["Model", "Accuracy", "Macro precision", "Macro recall", "Macro F1"]].copy()
        base = test.get("baseline")
        if base:
            show.loc[len(show)] = ["Majority baseline (always 'positive')", base.get("accuracy"), None, None, base.get("macro_f1")]
        for col in show.columns[1:]:
            show[col] = show[col].map(lambda v: "n/a" if v is None or pd.isna(v) else f"{v:.3f}")
        show_df(show)
        a, b = st.columns(2)
        with a:
            _bar(table, "accuracy", "Accuracy", "Accuracy", key="cmp_acc")
        with b:
            _bar(table, "macro_f1", "Macro F1-score", "Macro F1", key="cmp_f1")
        st.caption("Macro F1 is the plain average of the F1-scores of the three classes, so the small neutral and negative classes count as much as the big positive class.")

    # ---- speed and size
    with tabs[1]:
        if cost.empty:
            T.callout("pending", "Cost measurements are not available", "Re-run the evaluation script.")
        else:
            a, b = st.columns(2)
            with a:
                _bar(cost, "train_seconds", "Training time", "Seconds", log=_use_log(cost["train_seconds"]),
                     fmt=R.human_time, key="cmp_train")
            with b:
                _bar(cost, "inference_ms_per_review", "Time to label one review", "Milliseconds",
                     log=_use_log(cost["inference_ms_per_review"]), fmt=lambda v: f"{v:.3g} ms", key="cmp_infer")
            c, d = st.columns(2)
            with c:
                _bar(cost, "model_size_mb", "Model size on disk", "Megabytes", log=_use_log(cost["model_size_mb"]),
                     fmt=lambda v: f"{v:.3g} MB", key="cmp_size")
            with d:
                _bar(cost, "n_parameters", "Number of parameters", "Parameters", log=_use_log(cost["n_parameters"]),
                     fmt=R.human_count, key="cmp_params")
            notes = cost[["model", "train_note", "inference_device"]].rename(
                columns={"model": "Model", "train_note": "How training time was obtained", "inference_device": "Inference ran on"})
            show_df(notes)
            st.caption("Bars use a log scale when values differ by more than 20 times, so small models stay visible.")

    # ---- per class
    with tabs[2]:
        pc = R.per_class_table(test)
        metric = st.selectbox("Metric", ["f1", "precision", "recall"], format_func=lambda m: {"f1": "F1-score", "precision": "Precision", "recall": "Recall"}[m], key="cmp_pc_metric")
        fig = go.Figure()
        for key in ev:
            sub = pc[pc["key"] == key]
            fig.add_trace(go.Bar(name=R.DISPLAY[key], x=[c.capitalize() for c in sub["class"]], y=sub[metric],
                                 marker_color=T.MODEL_COLORS[key], text=[f"{v:.2f}" for v in sub[metric]], textposition="outside",
                                 cliponaxis=False))
        fig.update_layout(**T.chart_layout(f"{ {'f1': 'F1-score', 'precision': 'Precision', 'recall': 'Recall'}[metric] } per class", height=380), barmode="group")
        fig.update_yaxes(range=[0, 1.08])
        T.style_axes(fig)
        show_plot(fig, key="cmp_pc")
        wide = pc.pivot(index="model", columns="class", values=metric).reindex([R.DISPLAY[k] for k in ev])[C.LABEL_NAMES]
        wide = wide.map(lambda v: f"{v:.3f}") if hasattr(wide, "map") else wide.applymap(lambda v: f"{v:.3f}")
        wide.columns = [c.capitalize() for c in wide.columns]
        show_df(wide.reset_index().rename(columns={"model": "Model"}))
        st.caption("Neutral is usually the hardest class: neutral reviews often contain both praise and complaints.")

    # ---- confusion matrices
    with tabs[3]:
        key = st.selectbox("Model", list(ev), format_func=lambda k: R.DISPLAY[k], key="cmp_cm_model")
        cm = np.array(ev[key]["confusion_matrix"], dtype=float)
        f1, f2 = _confusion_figs(cm, R.DISPLAY[key])
        a, b = st.columns(2)
        with a:
            show_plot(f1, key="cmp_cm_counts")
        with b:
            show_plot(f2, key="cmp_cm_pct")
        off = cm.copy()
        np.fill_diagonal(off, 0)
        if off.max() > 0:
            i, j = np.unravel_index(np.argmax(off), off.shape)
            st.caption(f"Reading it: rows are the true class, columns are what the model predicted. The most common mistake is a "
                       f"{C.LABEL_NAMES[i]} review predicted as {C.LABEL_NAMES[j]} ({int(off[i, j])} reviews).")

    # ---- reports
    with tabs[4]:
        key = st.selectbox("Model", list(ev), format_func=lambda k: R.DISPLAY[k], key="cmp_rep_model")
        show_df(_report_table(ev[key]))
        st.caption("Precision: when the model says this class, how often is it right? Recall: of the reviews that truly belong to this class, how many were found?")

    # ---- trade-offs
    with tabs[5]:
        summary = _tradeoff_summary(table, cost)
        if summary:
            T.callout("info", "What this project measured", summary)
        else:
            T.callout("pending", "Classical versus transformer", "A full comparison needs both families to be evaluated.")
        rows = pd.DataFrame(content.COMPARISON_ROWS, columns=["", "Classical ML (TF-IDF + linear model)", "Transformer (DistilBERT)"])
        show_df(rows)
