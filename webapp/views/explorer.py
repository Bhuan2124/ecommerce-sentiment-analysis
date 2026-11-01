"""Data Explorer: look at the dataset behind the models."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from webapp import artifacts as A
from webapp import data_access as D
from webapp import textstats
from webapp import theme as T
from webapp.compat import show_df, show_plot

PAGE_SIZES = [10, 25, 50, 100]


@st.cache_data(show_spinner=False)
def _word_stats(stamp: int):
    train = A.load_split("train")
    if train is None:
        return None
    return textstats.top_words(train[C.TEXT_COL], train[C.LABEL_COL], C.LABEL_NAMES)


def _all_splits():
    parts = []
    for name in ("train", "val", "test"):
        df = D.split(name)
        if df is None:
            return None
        parts.append(df.assign(split=name))
    full = pd.concat(parts, ignore_index=True)
    full["words"] = full[C.TEXT_COL].str.split().str.len()
    return full


def _bar_h(names, values, color, title, fmt):
    fig = go.Figure(go.Bar(x=values[::-1], y=names[::-1], orientation="h", marker_color=color,
                           text=[fmt(v) for v in values[::-1]], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{text}<extra></extra>"))
    fig.update_layout(**T.chart_layout(title, height=380, legend=False))
    T.style_axes(fig, grid_y=False, grid_x=True)
    return fig


def _summary_tab(full: pd.DataFrame) -> None:
    dsum = D.get("dataset_summary")
    raw = D.raw()
    sizes = {s: int((full["split"] == s).sum()) for s in ("train", "val", "test")}
    T.kpi_row(
        [
            ("Rows in the original file", f"{len(raw):,}" if raw is not None else "n/a",
             "Kaggle CSV, untouched" if raw is not None else "original CSV not found in data/raw/", T.MUTED),
            ("Reviews after cleaning", f"{len(full):,}", "no empty text, no duplicates", T.ACCENT),
            ("Training split", f"{sizes['train']:,}", "70%: models learn from these", T.ACCENT),
            ("Validation split", f"{sizes['val']:,}", "15%: used to choose settings", T.ACCENT),
            ("Test split", f"{sizes['test']:,}", "15%: used once, at the end", T.ACCENT),
        ]
    )
    if dsum:
        cl, ded = dsum.get("cleaning", {}), dsum.get("deduplication", {})
        T.section("What the cleaning step removed")
        show_df(pd.DataFrame(
            [
                ("Rows in the original file", cl.get("rows_in")),
                ("Dropped: no review text", cl.get("dropped_missing_text")),
                ("Dropped: invalid star rating", cl.get("dropped_invalid_rating")),
                ("Dropped: duplicate reviews", ded.get("dropped_duplicate_rows")),
                ("Dropped: same text with different labels", ded.get("dropped_conflicting_label_rows")),
                ("Reviews kept", ded.get("rows_out")),
            ],
            columns=["Step", "Reviews"],
        ))

    T.section("Columns in the original file", "Missing values are shown for the file exactly as downloaded.")
    if raw is None:
        T.callout("info", "The original CSV is not in data/raw/", "The processed splits are available, so the rest of this page still works.")
    else:
        overview = A.column_overview(raw)
        a, b = st.columns([1.4, 1])
        with a:
            show_df(overview)
        with b:
            miss = overview[overview["missing"] > 0].sort_values("missing %", ascending=False)
            if miss.empty:
                T.callout("ok", "No missing values")
            else:
                show_plot(_bar_h(list(miss["column"]), list(miss["missing %"]), T.ACCENT, "Missing values (% of rows)",
                                 lambda v: f"{v:.1f}%"), key="ex_missing")

    T.section("Columns used by the models (after cleaning)")
    show_df(A.column_overview(full[[C.TEXT_COL, C.RATING_COL, C.LABEL_COL, C.LABEL_ID_COL, "split"]]))


def _distribution_tab(full: pd.DataFrame) -> None:
    a, b = st.columns(2)
    with a:
        counts = full.groupby(["split", C.LABEL_COL]).size().unstack(fill_value=0).reindex(["train", "val", "test"])
        pct = counts.div(counts.sum(axis=1), axis=0) * 100
        fig = go.Figure()
        for name in C.LABEL_NAMES:
            fig.add_trace(go.Bar(name=name.capitalize(), x=["Train", "Validation", "Test"], y=pct[name],
                                 marker_color=T.SENTIMENT_COLORS[name], text=[f"{v:.1f}%" for v in pct[name]],
                                 textposition="inside", insidetextanchor="middle", textfont=dict(color="#FFFFFF")))
        fig.update_layout(**T.chart_layout("Sentiment mix in each split", height=340), barmode="stack")
        fig.update_yaxes(title_text="% of reviews", range=[0, 100])
        T.style_axes(fig)
        show_plot(fig, key="ex_mix")
    with b:
        stars = full.groupby(C.RATING_COL).size().reindex([1, 2, 3, 4, 5], fill_value=0)
        fig = go.Figure(go.Bar(x=[f"{s} star" + ("s" if s > 1 else "") for s in stars.index], y=stars.values,
                               marker_color=[T.SENTIMENT_COLORS[C.RATING_TO_LABEL[s]] for s in stars.index],
                               text=[f"{v:,}" for v in stars.values], textposition="outside", cliponaxis=False))
        fig.update_layout(**T.chart_layout("Star ratings (colour = label we derive)", height=340, legend=False))
        T.style_axes(fig)
        show_plot(fig, key="ex_stars")
    st.caption("The data is unbalanced: most reviews are positive. That is why we judge models by macro F1, not accuracy alone.")

    T.section("Review length", "Length is counted in words.")
    fig = go.Figure()
    for name in C.LABEL_NAMES:
        fig.add_trace(go.Histogram(x=full.loc[full[C.LABEL_COL] == name, "words"], name=name.capitalize(), opacity=0.6,
                                   marker_color=T.SENTIMENT_COLORS[name], xbins=dict(start=0, size=5), histnorm="percent"))
    fig.update_layout(**T.chart_layout("How long are reviews?", height=340), barmode="overlay")
    fig.update_xaxes(title_text="Words per review")
    fig.update_yaxes(title_text="% of reviews in the class")
    T.style_axes(fig)
    show_plot(fig, key="ex_len")
    stats = textstats.length_stats(full["words"], full[C.LABEL_COL], C.LABEL_NAMES)
    show_df(pd.DataFrame(
        [{"Sentiment": s["label"].capitalize(), "Reviews": f"{s['reviews']:,}", "Median words": f"{s['median']:.0f}",
          "Mean words": f"{s['mean']:.1f}", "90% are shorter than": f"{s['p90']:.0f} words"} for s in stats]
    ))


def _browse_tab(full: pd.DataFrame) -> None:
    c1, c2, c3 = st.columns([2, 1.3, 1])
    with c1:
        query = st.text_input("Search the review text", key="ex_query", placeholder="for example: runs small")
    with c2:
        chosen = st.multiselect("Sentiment", C.LABEL_NAMES, default=C.LABEL_NAMES, key="ex_labels", format_func=str.capitalize)
    with c3:
        which = st.selectbox("Split", ["all", "train", "val", "test"], key="ex_split")
    c4, c5, c6 = st.columns([1.3, 1, 1])
    with c4:
        lo, hi = st.slider("Star rating", 1, 5, (1, 5), key="ex_stars_range")
    with c5:
        size = st.selectbox("Rows per page", PAGE_SIZES, index=1, key="ex_page_size")

    view = full
    if which != "all":
        view = view[view["split"] == which]
    view = view[view[C.LABEL_COL].isin(chosen) & view[C.RATING_COL].between(lo, hi)]
    if query.strip():
        view = view[view[C.TEXT_COL].str.contains(query.strip(), case=False, regex=False)]

    pages = max(1, -(-len(view) // size))
    if st.session_state.get("ex_page", 1) > pages:  # filters changed and the old page no longer exists
        st.session_state["ex_page"] = 1
    with c6:
        page = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1, key="ex_page")
    start = (int(page) - 1) * size
    st.caption(f"{len(view):,} reviews match. Showing {min(start + 1, len(view)):,} to {min(start + size, len(view)):,}.")
    if view.empty:
        T.callout("info", "No reviews match these filters", "Try a shorter search word or select more sentiments.")
        return
    shown = view.iloc[start:start + size][["split", C.RATING_COL, C.LABEL_COL, "words", C.TEXT_COL]].rename(
        columns={"split": "Split", C.RATING_COL: "Stars", C.LABEL_COL: "Sentiment", "words": "Words", C.TEXT_COL: "Review"})
    show_df(shown, height=min(36 * len(shown) + 40, 620))
    st.download_button("Download these matches as CSV", view[["split", C.RATING_COL, C.LABEL_COL, C.TEXT_COL]].to_csv(index=False),
                       file_name="filtered_reviews.csv", mime="text/csv", key="ex_download")


def _examples_tab(full: pd.DataFrame) -> None:
    st.session_state.setdefault("ex_seed", 0)
    if st.button("Show different examples", key="ex_shuffle"):
        st.session_state["ex_seed"] += 1
    seed = st.session_state["ex_seed"]
    cols = st.columns(len(C.LABEL_NAMES))
    for col, name in zip(cols, C.LABEL_NAMES):
        pool = full[full[C.LABEL_COL] == name]
        sample = pool.sample(n=min(3, len(pool)), random_state=seed)
        with col:
            st.markdown(f'<div style="margin-bottom:8px">{T.badge(name.capitalize(), name)}</div>', unsafe_allow_html=True)
            for _, row in sample.iterrows():
                text = str(row[C.TEXT_COL])
                text = text if len(text) <= 420 else text[:420].rsplit(" ", 1)[0] + " ..."
                st.markdown(
                    f'<div class="esa-card" style="margin-bottom:10px;padding:14px 16px">'
                    f'<p style="font-size:13px;color:{T.MUTED};margin-bottom:4px">{int(row[C.RATING_COL])} star rating</p>'
                    f'<p style="font-size:13.5px">{T.esc(text)}</p></div>',
                    unsafe_allow_html=True,
                )


def _words_tab() -> None:
    stats = _word_stats(A.mtime_key("train"))
    if not stats:
        T.callout("info", "Word statistics need the training split")
        return
    st.caption("Counted in the training split. Each review counts once per word, and common English filler words are hidden here (the models still see them).")
    T.section("Most frequent words")
    cols = st.columns(3)
    for col, name in zip(cols, C.LABEL_NAMES):
        with col:
            items = stats[name]["frequent"]
            show_plot(_bar_h([w for w, _ in items], [c for _, c in items], T.SENTIMENT_COLORS[name],
                             f"{name.capitalize()} reviews", lambda v: f"{v:,}"), key=f"ex_freq_{name}")
    T.section("Words most typical of each sentiment", "How many times more often a word appears in this sentiment than in all reviews.")
    cols = st.columns(3)
    for col, name in zip(cols, C.LABEL_NAMES):
        with col:
            items = stats[name]["typical"]
            show_plot(_bar_h([w for w, _, _ in items], [l for _, l, _ in items], T.SENTIMENT_COLORS[name],
                             f"{name.capitalize()} reviews", lambda v: f"{v:.1f}x"), key=f"ex_typ_{name}")


def render() -> None:
    T.page_header("Data Explorer", "Look at the reviews behind the models: how many there are, how they are labelled, how long they are, and what words they use.")
    full = _all_splits()
    if full is None:
        T.callout(
            "warn", "The processed dataset was not found",
            "The explorer reads <code>data/processed/train.csv</code>, <code>val.csv</code> and <code>test.csv</code>. "
            "Create them by running <code>notebooks/01_data_exploration.ipynb</code>. "
            "The other pages that use saved models and results keep working without them.",
        )
        return
    tabs = st.tabs(["Summary", "Sentiment and length", "Browse reviews", "Examples", "Common words"])
    with tabs[0]:
        _summary_tab(full)
    with tabs[1]:
        _distribution_tab(full)
    with tabs[2]:
        _browse_tab(full)
    with tabs[3]:
        _examples_tab(full)
    with tabs[4]:
        _words_tab()
