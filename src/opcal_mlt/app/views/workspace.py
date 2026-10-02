"""
Step 3 — Labeling Workspace
===========================

This module implements the third step of the labeling workflow in the OPCAL MLT tool.
Users interact with the workspace to label traces, navigate cells, and adjust parameters.

Functions:
    render: Main entry point for the Streamlit page.
    _render_label_controls: UI for labeling controls and actions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from opcal_mlt.app.components import (
    render_navigation_and_progress,
    render_sidebar_params,
    render_session_diagnostics,
)
from opcal_mlt.app.state import StateAdapter
from opcal_mlt.app.state_store import mark_dirty
from opcal_mlt.app.theme import get_theme
from opcal_mlt.app.workspace_logic import ensure_workspace_state, process_trace_for_cell
from opcal_mlt.domain.enums import BaselineMethod, LabelClass
from opcal_mlt.domain.models import TraceSet
from opcal_mlt.services.labeling import LabelingService
from opcal_mlt.services.logging import SessionLogger
from opcal_mlt.version import get_app_version

# ==== Main Page Renderer ====
def render(*, state: StateAdapter, labeling_service: LabelingService) -> None:
    """Render the Stage 3 labeling workspace page.

    Args:
        state: State adapter bridging Streamlit session state to services.
        labeling_service: Service responsible for persisting labels and peaks.

    Returns:
        None: Streamlit elements are rendered inline; nothing is returned.
    """
    s = st.session_state
    if not ensure_workspace_state(s):
        return

    theme_name = state.get("theme", "Light")
    theme = get_theme(theme_name)

    render_sidebar_params(state)
    traces = getattr(s, "traces", None)
    if traces is None:
        st.warning("No traces loaded. Return to Upload step.")
        return
    _, N = traces.shape

    left, mid, right = st.columns([3, 8, 3], gap="small")

    render_navigation_and_progress(left, s, N, theme)

    data = process_trace_for_cell(s)
    if isinstance(data, dict):
        data.setdefault("k", float(s.get("k", 3.0)))
        data.setdefault("stim_time_s", float(s.get("stim_time_s", 0.0)))

    with mid:
        cell_label = s.cell_ids[s.current_cell] if s.get("cell_ids") else s.current_cell
        st.markdown(f"### Cell <code>{cell_label}</code>", unsafe_allow_html=True)
        from opcal_mlt.app.plots import make_workspace_figure
        fig = make_workspace_figure(data, theme, dff_fixed=0.2, height=480)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        _render_label_controls(state, labeling_service, data)
        _render_plateau_qc(data, theme)

    render_session_diagnostics(s)


def _render_plateau_qc(data: dict, theme: dict) -> None:
    """HF vs HO check: area above the plateau line, compared with reference cells.

    A labeling aid only — it never changes the label. Computed on the loaded
    trace (``x``) with the session's sampling rate and stimulus time.
    """
    from opcal_mlt.app.plots import make_plateau_qc_figure
    from opcal_mlt.core import plateau_qc

    with st.expander("HF vs HO check", expanded=False):
        try:
            ref = plateau_qc.load_reference()
            score = plateau_qc.area_above_plateau(
                np.asarray(data["x"], dtype=float),
                float(data["fs_hz"]),
                float(data.get("stim_time_s", 0.0)),
            )
            pos = ref.position(score)
        except Exception as exc:  # never block labeling because of the aid
            st.caption(f"Check unavailable: {exc}")
            return
        st.metric("Area above plateau", f"{score:.2f}", help="ΔF/F·s per minute, from the stimulus to the end")
        st.markdown(f"**{pos['verdict']}** · HF percentile {pos['percentile_in_HF']:.0f} · HO percentile {pos['percentile_in_HO']:.0f}")
        st.plotly_chart(make_plateau_qc_figure(ref, score, theme), use_container_width=True)
        st.caption(
            "Separates HO from HF in the reference set (ROC AUC ≈ 0.97). "
            "Meaningful only when choosing between HF and HO. Depends on the sampling rate and "
            "stimulus time set in the sidebar."
        )


def _render_label_controls(
    state: StateAdapter,
    labeling_service: LabelingService,
    data: dict,
) -> None:
    """Render label controls and keep widget state in sync.

    Args:
        state: State adapter used to update the label map.
        labeling_service: Service responsible for persisting label actions.
        data: Processed trace payload for the active cell.
    """
    s = st.session_state

    current_cell = int(s.current_cell)
    existing = s.label_map.get(current_cell)
    if s.get("prev_cell") != current_cell:
        st.session_state["workspace_label_value"] = existing["label"] if existing else LabelClass.default().value
        st.session_state["workspace_notes_value"] = existing["notes"] if existing else ""
        st.session_state["workspace_uncertain_value"] = bool(existing.get("uncertain", False)) if existing else False
        s.prev_cell = current_cell
        mark_dirty(st.session_state)

    st.subheader("Label")
    label_options = [cls.value for cls in LabelClass]
    label_choice = st.radio(
        "Class",
        label_options,
        key="workspace_label_value",
    )
    uncertain = st.checkbox(
        "Mark as uncertain",
        key="workspace_uncertain_value",
        help="Flag this label as uncertain",
    )
    notes = st.text_area(
        "Notes",
        key="workspace_notes_value",
        placeholder="Optional free text",
    )

    st.markdown('<div class="btn-action btn-lg">', unsafe_allow_html=True)
    if st.button("Save label (CSV)", key="workspace_btn_save_label"):
        _handle_save_label(state, labeling_service, data, label_choice, notes, uncertain)
    st.markdown('</div>', unsafe_allow_html=True)


def _handle_save_label(
    state: StateAdapter,
    labeling_service: LabelingService,
    data: dict,
    label_choice: str,
    notes: str,
    uncertain: bool,
) -> None:
    """Persist the current label and advance workspace state.

    Args:
        state: State adapter used to update session and label metadata.
        labeling_service: Service that writes CSV artifacts.
        data: Processed trace payload for the active cell.
        label_choice: Label class selected in the UI.
        notes: Free-text notes captured from the annotator.
        uncertain: Whether the annotator flagged the label as uncertain.
    """
    s = st.session_state
    session_dir = state.get_session_dir()
    if not session_dir:
        st.warning("Start a session (Annotator & Save dir) to save CSVs.")
        return

    try:
        trace_set = TraceSet(traces=s.traces, cell_ids=s.cell_ids, fs_hz=float(s.get("fs_hz", 1 / 1.08)))
    except Exception as exc:
        st.error(f"Invalid trace configuration: {exc}")
        return

    label_enum = LabelClass.from_str(label_choice)
    peaks = np.asarray(data.get("peaks", []), dtype=int)
    metadata = {
        "filter_type": "savgol" if data.get("smooth", True) else "none",
        "filter_window": int(s.get("window", 31)) if data.get("smooth", True) else 0,
        "filter_polyorder": int(s.get("poly", 3)) if data.get("smooth", True) else 0,
        "baseline_method": BaselineMethod.from_str(s.get("baseline_method", "rolling_median")),
        "baseline_window_s_or_q": float(s.get("window_s", 20)) if str(s.get("baseline_method", "rolling_median")).startswith("rolling") else 25.0,
        "sd_method": "MAD",
        "threshold_k": float(data.get("k", s.get("k", 3.0))),
        "version": get_app_version(),
    }

    logger = SessionLogger(Path(session_dir))
    previous_entry = s.label_map.get(int(s.current_cell)) if isinstance(getattr(s, "label_map", {}), dict) else None

    try:
        labeling_service.save_label(
            session_dir=Path(session_dir),
            trace_set=trace_set,
            cell_index=int(s.current_cell),
            smoothed_trace=np.asarray(data["x_s"], dtype=float),
            threshold=np.asarray(data["thr"], dtype=float),
            peaks=peaks,
            label=label_enum,
            notes=notes,
            uncertain=uncertain,
            recording_id=str(s.get("recording_id", "")),
            annotator_id=state.get_annotator() or "",
            metadata=metadata,
        )
    except Exception as exc:
        st.error(f"Failed to save label: {exc}")
        return

    state.update_label_map(int(s.current_cell), label_enum, notes, uncertain)
    s.history.append((int(s.current_cell), previous_entry))
    mark_dirty(st.session_state)
    logger(f"save cell_index={int(s.current_cell)} label={label_enum.value}")
    st.success(f"Saved → {Path(session_dir) / 'labels.csv'}")

    next_unlabeled = [
        idx
        for idx in range(trace_set.traces.shape[1])
        if idx not in state.get_label_map() and idx > int(s.current_cell)
    ]
    if next_unlabeled:
        s.current_cell = int(next_unlabeled[0])
        mark_dirty(st.session_state)
        st.rerun()
