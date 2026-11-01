"""Colours, CSS and small HTML helpers so every page has the same look."""
from __future__ import annotations

import html
import re
from typing import Iterable, Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Palette (fixed by the project brief)
# ---------------------------------------------------------------------------
BG = "#F8FAFC"
TEXT = "#0F172A"
MUTED = "#64748B"
ACCENT = "#6366F1"
POSITIVE = "#16A34A"
NEGATIVE = "#DC2626"
NEUTRAL = "#D97706"
CARD = "#FFFFFF"
BORDER = "#E2E8F0"

SENTIMENT_COLORS = {"negative": NEGATIVE, "neutral": NEUTRAL, "positive": POSITIVE}
# Light backgrounds and darker text colours keep badges readable (contrast) on white
SENTIMENT_TINT = {"negative": "#FEE2E2", "neutral": "#FEF3C7", "positive": "#DCFCE7"}
SENTIMENT_TEXT = {"negative": "#991B1B", "neutral": "#92400E", "positive": "#166534"}

# One colour per model, chosen so they are not confused with the three sentiment colours
MODEL_COLORS = {
    "logreg": "#6366F1",
    "naive_bayes": "#0EA5E9",
    "linear_svm": "#0D9488",
    "distilbert": "#0F172A",
}

FONT_STACK = 'Inter, "Segoe UI", system-ui, -apple-system, Roboto, "Helvetica Neue", Arial, sans-serif'

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp {{ background: {BG}; color: {TEXT}; font-family: {FONT_STACK}; }}
.block-container {{ padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1240px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.stDeployButton, footer {{ visibility: hidden; }}
h1, h2, h3, h4 {{ color: {TEXT}; letter-spacing: -0.01em; font-family: {FONT_STACK}; }}
p, li, label, span {{ font-family: {FONT_STACK}; }}

/* Sidebar */
[data-testid="stSidebar"] {{ background: {CARD}; border-right: 1px solid {BORDER}; }}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 2px; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
    padding: 0.5rem 0.75rem; border-radius: 10px; width: 100%; cursor: pointer;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: {BG}; }}
.esa-brand {{ display: flex; align-items: center; gap: 10px; margin: 0 0 4px 0; }}
.esa-logo {{
    width: 34px; height: 34px; border-radius: 10px; background: {ACCENT}; color: #fff;
    display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 15px;
}}
.esa-brand-name {{ font-weight: 700; font-size: 15px; line-height: 1.2; color: {TEXT}; }}
.esa-brand-sub {{ font-size: 12px; color: {MUTED}; line-height: 1.2; }}
.esa-side-label {{
    font-size: 11px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
    color: {MUTED}; margin: 18px 0 6px 2px;
}}
.esa-side-row {{ display: flex; justify-content: space-between; align-items: center; font-size: 13px; padding: 3px 2px; color: {TEXT}; }}
.esa-dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 8px; }}

/* Page header */
.esa-eyebrow {{ font-size: 12px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: {ACCENT}; margin-bottom: 4px; }}
.esa-title {{ font-size: 30px; font-weight: 700; letter-spacing: -0.02em; color: {TEXT}; margin: 0; line-height: 1.2; }}
.esa-subtitle {{ font-size: 15px; color: {MUTED}; margin: 6px 0 22px 0; max-width: 820px; line-height: 1.55; }}
.esa-section {{ font-size: 17px; font-weight: 600; color: {TEXT}; margin: 28px 0 2px 0; }}
.esa-section-note {{ font-size: 13px; color: {MUTED}; margin: 0 0 10px 0; }}

/* Cards */
.esa-card {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px; padding: 20px 22px;
    box-shadow: 0 1px 2px rgba(15,23,42,0.04), 0 4px 14px rgba(15,23,42,0.04);
}}
.esa-card h4 {{ margin: 0 0 8px 0; font-size: 15px; }}
.esa-card p {{ margin: 0 0 8px 0; color: {TEXT}; font-size: 14px; line-height: 1.6; }}
.esa-card p:last-child {{ margin-bottom: 0; }}
.esa-hero {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 16px; padding: 26px 28px;
    box-shadow: 0 1px 2px rgba(15,23,42,0.04), 0 8px 24px rgba(99,102,241,0.08);
    border-left: 5px solid {ACCENT};
}}
.esa-hero .esa-title {{ font-size: 26px; }}

/* KPI tiles */
.esa-kpi {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px; padding: 16px 18px;
    box-shadow: 0 1px 2px rgba(15,23,42,0.04); position: relative; overflow: hidden; min-height: 104px;
}}
.esa-kpi::before {{ content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--tone, {ACCENT}); }}
.esa-kpi-label {{ font-size: 12px; font-weight: 600; color: {MUTED}; text-transform: uppercase; letter-spacing: 0.05em; }}
.esa-kpi-value {{ font-size: 26px; font-weight: 700; color: {TEXT}; margin-top: 6px; line-height: 1.15; letter-spacing: -0.01em; }}
.esa-kpi-sub {{ font-size: 12.5px; color: {MUTED}; margin-top: 4px; }}

/* Badges and callouts */
.esa-badge {{
    display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: 12.5px; font-weight: 600;
    border: 1px solid transparent;
}}
.esa-callout {{ border-radius: 12px; padding: 14px 16px; border: 1px solid {BORDER}; background: {CARD}; font-size: 14px; line-height: 1.55; }}
.esa-callout b {{ color: {TEXT}; }}
.esa-callout.info {{ background: #EEF2FF; border-color: #C7D2FE; }}
.esa-callout.pending {{ background: #FFFBEB; border-color: #FDE68A; }}
.esa-callout.warn {{ background: #FEF2F2; border-color: #FECACA; }}
.esa-callout.ok {{ background: #F0FDF4; border-color: #BBF7D0; }}
.esa-callout code, .esa-card code {{ background: rgba(15,23,42,0.06); padding: 1px 6px; border-radius: 6px; font-size: 12.5px; }}

/* Result panel in the analyzer */
.esa-result {{
    border-radius: 16px; padding: 22px 24px; border: 1px solid {BORDER}; background: {CARD};
    box-shadow: 0 1px 2px rgba(15,23,42,0.04), 0 8px 24px rgba(15,23,42,0.05);
}}
.esa-result-label {{ font-size: 34px; font-weight: 700; letter-spacing: -0.02em; line-height: 1.1; }}
.esa-result-meta {{ font-size: 13px; color: {MUTED}; margin-top: 6px; }}

/* Pipeline diagram */
.esa-flow {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: stretch; }}
.esa-step {{
    flex: 1 1 150px; background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px; padding: 12px 14px;
    font-size: 13px; line-height: 1.45; position: relative;
}}
.esa-step b {{ display: block; font-size: 13.5px; margin-bottom: 2px; color: {TEXT}; }}
.esa-step span {{ color: {MUTED}; }}

/* Buttons, inputs */
.stButton > button, .stDownloadButton > button {{ border-radius: 10px; font-weight: 600; padding: 0.5rem 1.1rem; border: 1px solid {BORDER}; }}
.stButton > button[kind="primary"] {{ background: {ACCENT}; border-color: {ACCENT}; color: #fff; }}
.stButton > button[kind="primary"]:hover {{ filter: brightness(0.94); }}
.stTextArea textarea, .stTextInput input {{ border-radius: 10px; }}
[data-testid="stExpander"] {{ border: 1px solid {BORDER}; border-radius: 12px; background: {CARD}; }}
[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 12px; overflow: hidden; }}
button[role="tab"] {{ font-weight: 600; }}
hr {{ border-color: {BORDER}; }}
"""


def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly helpers (plotly itself is imported only in the views)
# ---------------------------------------------------------------------------
def chart_layout(title: Optional[str] = None, height: int = 340, legend: bool = True, **extra) -> dict:
    """One consistent look for every chart. Use as: fig.update_layout(**chart_layout(...))."""
    layout = dict(
        height=height,
        margin=dict(l=8, r=8, t=52 if title else 20, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_STACK, size=13, color=TEXT),
        hoverlabel=dict(bgcolor="white", font_size=13, bordercolor=BORDER),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0, title_text=""),
    )
    if title:
        layout["title"] = dict(text=title, x=0, xanchor="left", font=dict(size=15, color=TEXT))
    layout.update(extra)
    return layout


def style_axes(fig, grid_y: bool = True, grid_x: bool = False) -> None:
    fig.update_xaxes(showgrid=grid_x, gridcolor=BORDER, linecolor=BORDER, zeroline=False, tickfont=dict(color=MUTED))
    fig.update_yaxes(showgrid=grid_y, gridcolor=BORDER, linecolor=BORDER, zeroline=False, tickfont=dict(color=MUTED))


# ---------------------------------------------------------------------------
# HTML helpers. Everything user-visible that could contain "<" is escaped.
# HTML is written on one line without indentation so Markdown does not treat it as code.
# ---------------------------------------------------------------------------
def esc(text) -> str:
    return html.escape(str(text), quote=True)


def inline_md(text: str) -> str:
    """Escape text and turn **bold** into <b>bold</b> (for text placed inside HTML cards)."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(str(text), quote=False))


def page_header(title: str, subtitle: str, eyebrow: str = "Sentiment analysis of e-commerce reviews") -> None:
    st.markdown(
        f'<div class="esa-eyebrow">{esc(eyebrow)}</div><div class="esa-title">{esc(title)}</div>'
        f'<div class="esa-subtitle">{esc(subtitle)}</div>',
        unsafe_allow_html=True,
    )


def section(title: str, note: Optional[str] = None) -> None:
    st.markdown(f'<div class="esa-section">{esc(title)}</div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<div class="esa-section-note">{esc(note)}</div>', unsafe_allow_html=True)


def kpi_html(label: str, value: str, sub: str = "", tone: str = ACCENT) -> str:
    sub_html = f'<div class="esa-kpi-sub">{esc(sub)}</div>' if sub else ""
    return (
        f'<div class="esa-kpi" style="--tone:{tone}"><div class="esa-kpi-label">{esc(label)}</div>'
        f'<div class="esa-kpi-value">{esc(value)}</div>{sub_html}</div>'
    )


def kpi_row(items: Iterable[tuple]) -> None:
    """items: (label, value, sub, tone) tuples shown side by side."""
    items = list(items)
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        label, value, *rest = item
        sub = rest[0] if len(rest) > 0 else ""
        tone = rest[1] if len(rest) > 1 else ACCENT
        with col:
            st.markdown(kpi_html(label, value, sub, tone), unsafe_allow_html=True)


def badge(label: str, kind: str) -> str:
    """Pill with the sentiment colours (kind = negative / neutral / positive)."""
    bg, fg, border = SENTIMENT_TINT.get(kind, "#EEF2FF"), SENTIMENT_TEXT.get(kind, ACCENT), SENTIMENT_COLORS.get(kind, ACCENT)
    return f'<span class="esa-badge" style="background:{bg};color:{fg};border-color:{border}">{esc(label)}</span>'


def callout(kind: str, title: str, body: str = "") -> None:
    """kind: info | pending | warn | ok. `body` may contain simple HTML such as <code>."""
    body_html = f"<div>{body}</div>" if body else ""
    st.markdown(f'<div class="esa-callout {kind}"><b>{esc(title)}</b>{body_html}</div>', unsafe_allow_html=True)


def card(title: str, body_html: str) -> None:
    st.markdown(f'<div class="esa-card"><h4>{esc(title)}</h4>{body_html}</div>', unsafe_allow_html=True)


def fmt_pct(x: Optional[float], digits: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:.{digits}f}%"


def fmt_num(x: Optional[float], digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"
