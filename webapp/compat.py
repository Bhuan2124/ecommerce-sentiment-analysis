"""Small wrappers so the pages work across Streamlit versions.

Newer Streamlit versions want `width="stretch"`; older ones want `use_container_width=True`.
These helpers try the new way first and fall back to the old way.
"""
from __future__ import annotations

import streamlit as st


def show_plot(fig, key=None) -> None:
    try:
        st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False})
    except Exception:  # older Streamlit: `width` only accepts integers
        st.plotly_chart(fig, use_container_width=True, key=key, config={"displaylogo": False})


def show_df(df, height=None, hide_index: bool = True) -> None:
    kwargs = {"hide_index": hide_index}
    if height is not None:
        kwargs["height"] = height
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except Exception:
        st.dataframe(df, use_container_width=True, **kwargs)
