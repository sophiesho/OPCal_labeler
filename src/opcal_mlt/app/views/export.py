"""Streamlit page: Step 4 — Finish & export."""
from __future__ import annotations

from pathlib import Path

import plotly.express as px
import streamlit as st

from opcal_mlt.app.state import StateAdapter
from opcal_mlt.app.state_store import load_snapshot
from opcal_mlt.core import features as ft
from opcal_mlt.services.export import ExportService
from opcal_mlt.services.sessions import SessionService


def render(*, state: StateAdapter, session_service: SessionService, export_service: ExportService) -> None:
    """Render the Stage 4 finish and export page.

    Args:
        state: Application state adapter.
        session_service: Service for hydrating saved session artifacts.
        export_service: Service that assembles ZIP exports.

    Returns:
        None: Streamlit renders UI elements directly.
    """
    st.markdown("<div class='step-header'>Step 4 — Finish & export</div>", unsafe_allow_html=True)
    session_dir_str = state.get_session_dir()
    if not session_dir_str:
        st.info("No session directory configured yet.")
        return

    session_dir = Path(session_dir_str)
    try:
        loaded = session_service.load_session(session_dir)
    except Exception as exc:
        st.error(f"Failed to load session data: {exc}")
        return

    if not state.get("_celebrated_finish", False):
        st.success("Great job! Labeling complete. You can now export this session as a ZIP archive.")
        try:
            st.balloons()
        except Exception:
            pass
        state.set("_celebrated_finish", True)

    st.markdown("---")
    st.subheader("Label statistics")

    display_map = {
        idx: {
            "label": record.label.value,
            "notes": record.notes,
            "uncertain": record.uncertain,
        }
        for idx, record in loaded.label_map.items()
    }
    if not display_map and state.get_label_map():
        display_map = state.get_label_map()

    total_cells = None
    traces = state.get("traces")
    if traces is not None and hasattr(traces, "shape"):
        total_cells = int(traces.shape[1])
    elif loaded.cell_ids:
        total_cells = len(loaded.cell_ids)

    labels_df, stats_df = ft.summarize_labels(display_map, loaded.cell_ids, total_cells=total_cells)

    if len(labels_df) == 0:
        st.info("No labels saved yet in this session.")
    else:
        st.caption(f"Found {len(labels_df)} labeled cells" + (f" / {total_cells} total" if total_cells else ""))
        try:
            fig = px.pie(stats_df, names="label", values="count", hole=0.45, title="Class distribution")
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

        df_display = labels_df.copy()
        if "uncertain" in df_display.columns:
            df_display["Uncertain?"] = df_display["uncertain"].map(lambda v: "✓" if bool(v) else "✗")
            cols = [c for c in ["cell_index", "cell_id", "label", "Uncertain?", "notes"] if c in df_display.columns]
            remaining = [c for c in df_display.columns if c not in cols and c != "uncertain"]
            df_display = df_display[cols + remaining]
        st.dataframe(df_display, use_container_width=True)

    st.markdown("---")
    st.subheader("Export")
    zip_col1, zip_col2 = st.columns([1, 2])
    with zip_col1:
        st.markdown('<div class="btn-action btn-lg">', unsafe_allow_html=True)
        trigger = st.button("Export session as ZIP", key="export_zip_btn")
        st.markdown('</div>', unsafe_allow_html=True)
    with zip_col2:
        st.caption("Creates a ZIP archive of the current session folder (labels.csv, peaks.csv, session.csv, cell_map.csv).")

    if trigger:
        try:
            archive_path = export_service.export_session(session_dir)
            state.set("export_done", True)
            st.success(f"Exported: {archive_path}")
            with open(archive_path, "rb") as f:
                st.download_button(
                    label="Download session ZIP",
                    data=f.read(),
                    file_name=Path(archive_path).name,
                    mime="application/zip",
                    key="download_zip_btn",
                )
        except Exception as exc:
            st.error(f"Export failed: {exc}")

    training_traces = _resolve_export_traces(state, session_dir)
    source_name = _resolve_source_name(state, loaded.metadata)

    csv_col1, csv_col2 = st.columns([1, 2])
    with csv_col1:
        st.markdown('<div class="btn-action btn-lg">', unsafe_allow_html=True)
        trigger_training = st.button(
            "Export training CSVs",
            key="export_training_csv_btn",
            disabled=training_traces is None,
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with csv_col2:
        st.caption(
            "Creates class-wise CSV files with timepoints as rows and ROIs as columns. "
            "Uncertain labels are exported separately."
        )
        if training_traces is None:
            st.warning("Training CSV export requires the loaded trace matrix. Re-open a session with saved traces or upload the source file again.")

    if trigger_training and training_traces is not None:
        try:
            result = export_service.export_training_csv_bundle(
                session_dir=session_dir,
                traces=training_traces,
                source_name=source_name,
            )
            counts = ", ".join(f"{name}: {count}" for name, count in result.counts_by_file.items())
            state.set("export_done", True)
            st.success(f"Exported training CSV bundle: {result.archive_path}")
            with open(result.archive_path, "rb") as f:
                st.download_button(
                    label="Download training CSV ZIP",
                    data=f.read(),
                    file_name=Path(result.archive_path).name,
                    mime="application/zip",
                    key="download_training_zip_btn",
                )
            st.caption(counts)
        except Exception as exc:
            st.error(f"Training CSV export failed: {exc}")


def _resolve_export_traces(state: StateAdapter, session_dir: Path):
    traces = state.get("traces")
    if traces is not None and hasattr(traces, "shape"):
        return traces

    snapshot = load_snapshot(session_dir)
    if snapshot is None or snapshot.traces is None:
        return None

    state.set("traces", snapshot.traces)
    cell_ids = snapshot.data.get("cell_ids")
    if cell_ids and not state.get_cell_ids():
        state.set_cell_ids(cell_ids)
    for key in ("source_filename", "source_sha256"):
        value = snapshot.data.get(key)
        if value and not state.get(key):
            state.set(key, value)
    return snapshot.traces


def _resolve_source_name(state: StateAdapter, metadata: dict) -> str:
    return str(
        state.get("source_filename")
        or metadata.get("source_path")
        or metadata.get("recording_id")
        or "recording"
    )
