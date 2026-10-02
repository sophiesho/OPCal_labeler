"""Plot builders for OPCAL‑Labeler.

This module centralizes Plotly figure creation so Streamlit pages stay focused on
UI control flow. Functions here are side-effect free: they build and return
figures without touching Streamlit state.
"""
from __future__ import annotations
from typing import Dict

import numpy as np
import plotly.graph_objects as go

from opcal_mlt.app.ui import apply_plotly_theme
from opcal_mlt.core.preprocess import pre_post_sd_rect_params


def make_workspace_figure(
    data: Dict,
    theme: Dict,
    *,
    dff_fixed: float = 0.2,
    height: int = 480,
) -> go.Figure:
    """Build the main workspace figure.

    Args:
        data: Dictionary produced by the cell processing function. Expected keys include
            "t", "x", "x_s", "base", "thr", "peaks" (NumPy arrays) and flags
            such as "smooth", "show_raw", "show_smoothed". When available, provide
            "stim_idx" and rectangle parameters ("rect_y0_pre", "rect_y1_pre",
            "rect_y0_post", "rect_y1_post").
        theme: Palette dictionary providing Plotly colors for traces and shaded regions.
        dff_fixed: Horizontal ΔF/F reference line value.
        height: Figure height in pixels.

    Returns:
        go.Figure: Configured Plotly figure for the labeling workspace.

    Notes:
        The builder is defensive: shapes and peaks are skipped gracefully when inputs
        are missing or inconsistent.
    """
    fig = go.Figure()

    # Raw / smoothed traces
    if data.get("show_raw", True):
        fig.add_trace(go.Scatter(x=data["t"], y=data["x"], name="raw", line=dict(width=1)))
    if data.get("smooth", True) and data.get("show_smoothed", True):
        fig.add_trace(go.Scatter(x=data["t"], y=data["x_s"], name="smoothed", line=dict(width=2)))

    # Baseline (dashed)
    fig.add_trace(go.Scatter(x=data["t"], y=data["base"], name="baseline", line=dict(width=1, dash="dash")))

    # Fixed ΔF/F reference line
    fig.add_trace(
        go.Scatter(
            x=data["t"],
            y=[float(dff_fixed)] * len(data["t"]),
            name="ΔF/F = 0.2",
            line=dict(width=1, dash="dot"),
        )
    )

    # Floating SD·k rectangles (pre/post)
    si = int(data.get("stim_idx", 0))
    t = data["t"]

    # Guards for short traces and index clamping
    if t is None or len(t) == 0:
        apply_plotly_theme(fig, theme)
        return fig
    si = max(0, min(int(si), len(t) - 1))

    if all(k in data for k in ("rect_y0_pre", "rect_y1_pre", "rect_y0_post", "rect_y1_post")):
        # Backward‑compatible path: use provided rectangle params as‑is
        y0_pre = float(data["rect_y0_pre"])
        y1_pre = float(data["rect_y1_pre"])
        y0_post = float(data["rect_y0_post"])
        y1_post = float(data["rect_y1_post"])
    else:
        # Fallback path: compute spans via the shared preprocessing helper so the
        # UI remains consistent even if upstream data lacks explicit rectangle params.
        x_for_rect = np.asarray(data.get("x_s", data.get("x", [])), dtype=float)
        fs_val = float(data.get("fs_hz", 1.0))
        stim_time_val = float(data.get("stim_time_s", 0.0))
        k_val = float(data.get("k", 3.0))

        y0_pre, y1_pre, y0_post, y1_post, _ = pre_post_sd_rect_params(
            x_for_rect, fs_val, stim_time_val, k=k_val, ref="median"
        )

    if y1_pre > y0_pre and len(t) >= 2 and si >= 0:
        fig.add_shape(
            type="rect", xref="x", yref="y",
            x0=float(t[0]), x1=float(t[si]), y0=y0_pre, y1=y1_pre,
            line=dict(width=0), fillcolor=theme.get("shade_pre", "rgba(99,102,241,0.10)"),
            opacity=1.0, layer="below",
        )
    if y1_post > y0_post and len(t) >= 2 and si < len(t):
        fig.add_shape(
            type="rect", xref="x", yref="y",
            x0=float(t[si]), x1=float(t[-1]), y0=y0_post, y1=y1_post,
            line=dict(width=0), fillcolor=theme.get("shade_post", "rgba(16,185,129,0.10)"),
            opacity=1.0, layer="below",
        )

    # Detected peak-episode regions (adaptive_zv method only): shade the whole
    # region so it reads as "this stretch was flagged", rather than a dense
    # scatter of one dot per frame.
    peak_episodes = data.get("peak_episodes") or []
    if peak_episodes:
        for ep_start, ep_end in peak_episodes:
            ep_start = max(0, min(int(ep_start), len(t) - 1))
            ep_end = max(0, min(int(ep_end), len(t) - 1))
            if ep_end < ep_start:
                continue
            fig.add_shape(
                type="rect", xref="x", yref="paper",
                x0=float(t[ep_start]), x1=float(t[ep_end]), y0=0, y1=1,
                line=dict(width=0),
                fillcolor=theme.get("shade_peak_episode", "rgba(147,51,234,0.12)"),
                opacity=1.0, layer="below",
            )

    # Peaks: one marker per detected episode, at its local max
    peaks = data.get("peaks")
    if peaks is not None:
        try:
            peaks = np.asarray(peaks, dtype=int)
            peaks = peaks[(peaks >= 0) & (peaks < len(t))]
        except Exception:
            peaks = np.array([], dtype=int)
    if peaks is not None and len(peaks) > 0:
        fig.add_trace(
            go.Scatter(
                x=t[peaks],
                y=data["x_s"][peaks],
                mode="markers",
                name="peaks",
                marker=dict(size=9, symbol="diamond", line=dict(width=1)),
            )
        )

    fig.update_layout(height=height, margin=dict(l=64, r=24, t=48, b=48))
    apply_plotly_theme(fig, theme)
    y_axis_kwargs = dict(
        tickformat=".3f",
        hoverformat=".4f",
        exponentformat="none",
    )
    if isinstance(data.get("y_range"), (list, tuple)) and len(data["y_range"]) == 2:
        try:
            y0, y1 = float(data["y_range"][0]), float(data["y_range"][1])
            if y0 < y1:
                y_axis_kwargs["range"] = [y0, y1]
        except Exception:
            pass
    fig.update_yaxes(**y_axis_kwargs)
    return fig


def make_status_figure(status: np.ndarray, theme: Dict, *, height: int = 90) -> go.Figure:
    """Build the mini status strip displayed beneath the cell selector.

    Args:
        status: Binary array indicating whether each cell is labeled.
        theme: Palette dictionary providing bar colors.
        height: Figure height in pixels.

    Returns:
        go.Figure: Plotly bar chart summarising label completion.
    """
    colors = [theme.get("status_unlabeled"), theme.get("status_labeled")]
    fig = go.Figure(
        go.Bar(
            x=list(range(len(status))),
            y=status,
            marker_color=[colors[status[i]] for i in range(len(status))],
        )
    )
    fig.update_yaxes(visible=False)
    fig.update_xaxes(title_text="Cells", tickmode="auto", nticks=10)
    fig.update_layout(height=height, margin=dict(l=4, r=4, t=4, b=4))
    apply_plotly_theme(fig, theme)
    return fig


def make_plateau_qc_figure(
    reference,
    score: float,
    theme: Dict,
    *,
    height: int = 220,
) -> go.Figure:
    """Small histogram of the HF / HO reference scores with the current cell marked.

    Args:
        reference: ``opcal_mlt.core.plateau_qc.Reference`` with ``hf``, ``ho`` and ``threshold``.
        score: Score of the current cell (ΔF/F·s per minute).
        theme: Palette dictionary.
        height: Figure height in pixels.
    """
    # log-spaced bins: the score spans ~2 orders of magnitude
    pos = np.concatenate([reference.hf, reference.ho])
    pos = pos[pos > 0]
    lo = np.log10(max(1e-2, float(np.min(pos)) if pos.size else 1e-2))
    hi = np.log10(max(float(np.max(pos)) if pos.size else 10.0, 10.0))
    edges = np.logspace(lo, hi, 30)
    centers = np.sqrt(edges[:-1] * edges[1:])
    fig = go.Figure()
    for vals, name, color in (
        (reference.hf, "HF (reference)", "rgba(255,140,0,0.55)"),
        (reference.ho, "HO (reference)", "rgba(214,39,40,0.55)"),
    ):
        counts, _ = np.histogram(np.clip(vals, edges[0], edges[-1]), bins=edges)
        fig.add_trace(go.Bar(x=centers, y=counts, name=name, marker_color=color,
                             width=np.diff(edges), hovertemplate="%{y} cells<extra>" + name + "</extra>"))
    fig.add_vline(x=float(reference.threshold), line_dash="dot", line_color=theme.get("muted", "#6b7280"))
    if np.isfinite(score):
        fig.add_vline(x=float(np.clip(score, edges[0], edges[-1])), line_width=3,
                      line_color=theme.get("accent", "#2563eb"))
    fig.update_layout(barmode="overlay", height=height, margin=dict(l=10, r=10, t=10, b=30),
                      legend=dict(orientation="h", y=1.15, x=0, font=dict(size=10)), bargap=0)
    fig.update_xaxes(type="log", title_text="area above plateau (ΔF/F·s/min)", title_font=dict(size=10))
    fig.update_yaxes(title_text="cells", title_font=dict(size=10))
    apply_plotly_theme(fig, theme)
    return fig
