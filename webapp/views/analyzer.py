"""Sentiment Analyzer: type a review, run a real saved model, see the result."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src import config as C
from webapp import history, inference
from webapp import theme as T
from webapp.compat import show_df, show_plot
from webapp.inference import (
    MAX_INPUT_CHARS, MODEL_INFOS, MODEL_ORDER, InvalidInputError, ModelLoadError, ModelUnavailableError,
)

EXAMPLES = {
    "Positive example": "Lovely dress, fits perfectly and the fabric is soft. I love it!",
    "Negative example": "The fabric feels cheap and it fell apart after one wash. I am returning it.",
    "Mixed example": "It is okay. Nice colour but it runs small and the material is a bit thin.",
}

SCORE_NOTES = {
    "probability": (
        "These percentages come from the model's probability output and add up to 100%."
    ),
    "decision_score": (
        "The Linear SVM does not produce probabilities. These are raw decision scores: the class with the highest score wins, "
        "and a bigger gap to the others means the model is more certain. They are not percentages."
    ),
    "softmax": (
        "DistilBERT's scores are a softmax of its output layer. They add up to 100% but are not guaranteed to be calibrated "
        "probabilities: the model can be wrong even when it shows a very high number."
    ),
}


@st.cache_resource(show_spinner=False)
def _load_cached(key: str):
    """Load a saved model once and keep it in memory. (Loading never trains anything.)"""
    return inference.load_model(key)


def _set_example(text: str) -> None:
    st.session_state["review_text"] = text


def _score_text(p) -> str:
    if p.confidence is not None:
        return f"{p.confidence * 100:.1f}%"
    top = max(p.scores.values())
    return f"score {top:+.2f}"


def _result_panel(p, name: str) -> None:
    color = T.SENTIMENT_COLORS[p.label]
    if p.score_kind == "decision_score":
        meta = f"Decision score {max(p.scores.values()):+.2f} (not a probability)"
    else:
        meta = f"Confidence {p.confidence * 100:.1f}%" + (" (softmax score)" if p.score_kind == "softmax" else "")
    meta += f" &nbsp;|&nbsp; {p.latency_ms:.1f} ms"
    st.markdown(
        f'<div class="esa-result" style="border-left:6px solid {color}">'
        f'<div class="esa-eyebrow" style="color:{T.MUTED}">Prediction by {T.esc(name)}</div>'
        f'<div class="esa-result-label" style="color:{color}">{T.esc(p.label.capitalize())}</div>'
        f'<div class="esa-result-meta">{meta}</div></div>',
        unsafe_allow_html=True,
    )


def _score_chart(p):
    labels = list(p.scores.keys())
    values = [p.scores[k] for k in labels]
    if p.confidence is not None:
        shown = [v * 100 for v in values]
        text = [f"{v:.1f}%" for v in shown]
        axis_title = "Share of the model's total score (%)"
        rng = [0, 105]
    else:
        shown = values
        text = [f"{v:+.2f}" for v in values]
        axis_title = "Raw decision score"
        pad = (max(values) - min(values)) * 0.25 + 0.1
        rng = [min(min(values), 0) - pad, max(max(values), 0) + pad]
    colors = [T.SENTIMENT_COLORS[k] if k == p.label else _fade(T.SENTIMENT_COLORS[k]) for k in labels]
    fig = go.Figure(
        go.Bar(x=shown, y=[k.capitalize() for k in labels], orientation="h", marker_color=colors, text=text,
               textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{text}<extra></extra>")
    )
    fig.update_layout(**T.chart_layout(None, height=210, legend=False))
    fig.update_xaxes(range=rng, title_text=axis_title)
    fig.update_yaxes(autorange="reversed")
    T.style_axes(fig, grid_y=False, grid_x=True)
    return fig


def _fade(hex_color: str) -> str:
    """A lighter version of a colour for the classes that were not predicted."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},0.35)"


def _run(keys, text):
    results, problems = [], []
    for key in keys:
        name = MODEL_INFOS[key].name
        try:
            model = _load_cached(key)
            results.append(inference.predict_text(model, text))
        except ModelUnavailableError as exc:
            problems.append((name, "not available", str(exc)))
        except ModelLoadError as exc:
            problems.append((name, "could not be loaded", str(exc)))
        except InvalidInputError:
            raise
        except Exception as exc:  # anything unexpected: show it, do not crash the page
            problems.append((name, "failed", f"{type(exc).__name__}: {exc}"))
    return results, problems


def render() -> None:
    T.page_header(
        "Sentiment Analyzer",
        "Paste a customer review, choose a model, and press Analyze. The prediction comes from a real saved model; "
        "nothing is hard-coded and nothing is trained when you click.",
    )
    availability = {k: inference.availability(k) for k in MODEL_ORDER}

    def label_of(key):
        return MODEL_INFOS[key].name + ("" if availability[key][0] else "  (not available yet)")

    left, right = st.columns([1.7, 1])
    with left:
        st.text_area(
            "Review text", key="review_text", height=170, max_chars=MAX_INPUT_CHARS,
            placeholder="Example: Lovely dress, fits perfectly and the fabric is soft. I love it!",
            help="Press Ctrl+Enter or click outside the box to confirm your text.",
        )
        words = len(str(st.session_state.get("review_text", "")).split())
        st.caption(f"{words:,} words in the box. The counter in the corner of the box shows characters (limit {MAX_INPUT_CHARS:,}).")

        ex_cols = st.columns(len(EXAMPLES))
        for col, (name, text) in zip(ex_cols, EXAMPLES.items()):
            with col:
                st.button(name, on_click=_set_example, args=(text,), key=f"ex_{name}")

        st.caption("The models were trained on women's clothing reviews, so they work best on reviews about clothes. "
                   "A review about another kind of product (or about delivery only) can get a wrong or unsure answer.")

        c1, c2 = st.columns([1.3, 1])
        with c1:
            default_index = next((i for i, k in enumerate(MODEL_ORDER) if availability[k][0]), 0)
            model_key = st.selectbox("Model", MODEL_ORDER, index=default_index, format_func=label_of, key="model_choice")
        with c2:
            st.write("")
            compare = st.checkbox("Compare all available models", key="compare_all")
        analyze = st.button("Analyze review", type="primary", key="analyze_btn")

    with right:
        rows = ""
        for key in MODEL_ORDER:
            ok = availability[key][0]
            dot = T.POSITIVE if ok else T.MUTED
            status = "ready" if ok else "not available"
            rows += (f'<div class="esa-side-row"><span><span class="esa-dot" style="background:{dot}"></span>'
                     f'{T.esc(MODEL_INFOS[key].name)}</span><span style="color:{T.MUTED}">{status}</span></div>')
        T.card("Model status", rows + f'<p style="margin-top:10px;color:{T.MUTED};font-size:12.5px">'
                                      "A model shows as ready when its saved files exist. DistilBERT runs on the CPU here, "
                                      "so the first prediction takes a few seconds while it loads.</p>")

    # ---- run
    if analyze:
        try:
            text = inference.validate_text(st.session_state.get("review_text"))
        except InvalidInputError as exc:
            st.session_state.pop("analysis", None)
            st.warning(str(exc))
        else:
            keys = [k for k in MODEL_ORDER if availability[k][0]] if compare else [model_key]
            if compare and model_key not in keys:
                keys = [model_key] + keys
            with st.spinner("Loading the model and analysing your review ..."):
                results, problems = _run(keys, text)
            st.session_state["analysis"] = {"text": text, "selected": model_key, "results": results, "problems": problems}
            primary = next((r for r in results if r.model_key == model_key), results[0] if results else None)
            if primary is not None:
                history.append(MODEL_INFOS[primary.model_key].name, primary.label, primary.confidence, text)

    analysis = st.session_state.get("analysis")
    if not analysis:
        return

    # ---- show results
    st.markdown("---")
    for name, kind, detail in analysis["problems"]:
        st.error(f"{name} {kind}. {detail}")
        if kind == "could not be loaded":
            st.caption("Tip: check that the file was saved with the same library versions as the ones installed here.")

    results = analysis["results"]
    if not results:
        return
    primary = next((r for r in results if r.model_key == analysis["selected"]), results[0])
    info = MODEL_INFOS[primary.model_key]

    r1, r2 = st.columns([1, 1.25])
    with r1:
        _result_panel(primary, info.name)
        st.markdown(f'<div style="margin-top:10px;font-size:13px;color:{T.MUTED}">Review analysed:</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="esa-callout">{T.esc(analysis["text"][:600])}{"..." if len(analysis["text"]) > 600 else ""}</div>',
                    unsafe_allow_html=True)
        if primary.truncated:
            st.warning(f"This review is longer than {C.MAX_LENGTH} tokens, so DistilBERT only read the first {C.MAX_LENGTH}.")
    with r2:
        show_plot(_score_chart(primary), key="an_chart")
        st.caption(SCORE_NOTES[primary.score_kind])

    if len(results) > 1:
        T.section("All models side by side")
        table = [
            {"Model": MODEL_INFOS[r.model_key].name, "Prediction": r.label.capitalize(), "Confidence / score": _score_text(r),
             "Type of score": {"probability": "probability", "decision_score": "decision score", "softmax": "softmax score"}[r.score_kind],
             "Time (ms)": round(r.latency_ms, 1)}
            for r in results
        ]
        show_df(table)
        labels = {r.label for r in results}
        if len(labels) == 1:
            T.callout("ok", f"All {len(results)} models agree: {next(iter(labels)).capitalize()}")
        else:
            T.callout("info", "The models disagree", "That is normal for mixed or short reviews. Neutral is the hardest class.")
