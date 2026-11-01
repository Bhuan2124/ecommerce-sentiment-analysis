"""Streamlit dashboard: Sentiment Analysis of E-Commerce Reviews, Classical ML vs. DistilBERT.

Start it from the project folder with:
    python -m streamlit run app.py

The app only LOADS saved models and result files. It never trains anything.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if not (ROOT / "src").is_dir():  # e.g. when run through a test harness
    ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402


def _assemble_distilbert() -> None:
    """GitHub limits files to 100 MB, so the 268 MB DistilBERT weights are stored as pieces in
    models/distilbert/parts/. Join them into model.safetensors once, the first time the app starts."""
    import hashlib
    ddir = ROOT / "models" / "distilbert"
    target, parts_dir = ddir / "model.safetensors", ddir / "parts"
    if target.is_file() or not parts_dir.is_dir():
        return
    parts = sorted(parts_dir.glob("model.safetensors.part*"))
    if not parts:
        return
    tmp = target.with_suffix(".tmp")
    digest = hashlib.sha256()
    with open(tmp, "wb") as out:
        for part in parts:
            data = part.read_bytes()
            digest.update(data)
            out.write(data)
    expected = (parts_dir / "SHA256").read_text().strip() if (parts_dir / "SHA256").is_file() else None
    if expected and digest.hexdigest() != expected:
        tmp.unlink(missing_ok=True)
        return
    tmp.replace(target)


try:
    _assemble_distilbert()
except OSError:
    pass  # DistilBERT will simply show as "pending"; the classical models still work

st.set_page_config(page_title="Sentiment Analysis of E-Commerce Reviews", layout="wide", initial_sidebar_state="expanded")

from webapp import artifacts, inference  # noqa: E402
from webapp import theme as T  # noqa: E402
from webapp.views import about, analyzer, comparison, explorer, insights, overview  # noqa: E402

PAGES = {
    "Overview Dashboard": overview.render,
    "Sentiment Analyzer": analyzer.render,
    "Model Comparison": comparison.render,
    "Data Explorer": explorer.render,
    "Model Insights": insights.render,
    "About Project": about.render,
}

T.inject_css()
if st.session_state.get("nav") not in PAGES:
    # a link such as http://localhost:8501/?page=Model%20Comparison opens that page directly
    _wanted = st.query_params.get("page")
    st.session_state["nav"] = _wanted if _wanted in PAGES else "Overview Dashboard"

with st.sidebar:
    st.markdown(
        '<div class="esa-brand"><div class="esa-logo">S</div><div><div class="esa-brand-name">Review Sentiment</div>'
        '<div class="esa-brand-sub">Classical ML vs DistilBERT</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="esa-side-label">Navigate</div>', unsafe_allow_html=True)
    st.radio("Navigate", list(PAGES), key="nav", label_visibility="collapsed")

    st.markdown('<div class="esa-side-label">Saved models</div>', unsafe_allow_html=True)
    rows = ""
    for key, ready in inference.available_models().items():
        dot = T.POSITIVE if ready else T.BORDER
        state = "ready" if ready else "pending"
        rows += (f'<div class="esa-side-row"><span><span class="esa-dot" style="background:{dot}"></span>'
                 f'{T.esc(inference.MODEL_INFOS[key].name)}</span><span style="color:{T.MUTED};font-size:12px">{state}</span></div>')
    st.markdown(rows, unsafe_allow_html=True)
    test_ready = artifacts.exists("test_results")
    st.markdown('<div class="esa-side-label">Test evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="esa-side-row"><span><span class="esa-dot" style="background:{T.POSITIVE if test_ready else T.NEUTRAL}"></span>'
        f'Held-out test set</span><span style="color:{T.MUTED};font-size:12px">{"done" if test_ready else "not run"}</span></div>',
        unsafe_allow_html=True,
    )

try:
    PAGES[st.session_state["nav"]]()
except Exception as exc:  # last safety net: show a friendly message instead of a crash
    st.error("Something went wrong while drawing this page. The rest of the app still works; try another page.")
    with st.expander("Technical details (for debugging)"):
        st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
