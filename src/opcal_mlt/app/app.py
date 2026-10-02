"""
Application Entry Point
======================

Streamlit entry point orchestrating routing, layout, and services for OPCAL-Labeler.
Handles page configuration, service initialization, and main application flow.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict
from uuid import uuid4

import streamlit as st

from opcal_mlt.app.state_store import (
    HYDRATED_FLAG,
    clear_state_for_token,
    consume_dirty_flag,
    hydrate_state,
    mark_dirty,
    persist_state_from_mapping,
)
from opcal_mlt.app.views import export as page_export
from opcal_mlt.app.views import ingest as page_ingest
from opcal_mlt.app.views import start as page_start
from opcal_mlt.app.views import workspace as page_workspace
from opcal_mlt.app.routing import Router
from opcal_mlt.app.session_io import write_cell_map
from opcal_mlt.app.state import StateAdapter
from opcal_mlt.app.theme import get_theme
from opcal_mlt.app.ui import inject_theme_css, render_stepper_and_tips
from opcal_mlt.domain.enums import Stage
from opcal_mlt.domain.models import SessionConfig
from opcal_mlt.services.export import ExportService
from opcal_mlt.services.ingest import IngestService
from opcal_mlt.services.labeling import LabelingService
from opcal_mlt.services.logging import SessionLogger
from opcal_mlt.services.sessions import SessionService
from opcal_mlt.version import get_app_version

APP_NAME = "OPCAL-Labeler"
APP_VERSION = get_app_version()
STAGE_FLOW = [Stage.START, Stage.INGEST, Stage.WORKSPACE, Stage.EXPORT]
ASSETS_DIR = Path(__file__).parent / "assets"
DEFAULT_ICON = ASSETS_DIR / "logo.png"
SESSION_TOKEN_PARAM = "session"


def run() -> None:
    """
    Render the full Streamlit application.

    Sets up page configuration, initializes services, applies theme, and manages routing.
    """
    page_icon = str(DEFAULT_ICON) if DEFAULT_ICON.exists() else None
    st.set_page_config(
        page_title=APP_NAME,
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon=page_icon,
    )

    session_token = _ensure_session_token()
    hydrate_state(st.session_state, session_token)

    state = StateAdapter(st.session_state)
    services = {
        "sessions": SessionService(),
        "ingest": IngestService(),
        "labeling": LabelingService(),
        "export": ExportService(),
    }

    _initialize_state()
    _apply_stage_from_query(state)

    theme = get_theme(state.get("theme", "Light"))
    inject_theme_css(theme)

    st.markdown(
        f"""
        <div class="app-title">
          <div class="app-title-main">{APP_NAME}</div>
          <div class="app-title-sub">Manual labeling tool • v{APP_VERSION}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stage = state.get_stage()
    _ensure_query_stage(stage)
    if stage == Stage.WORKSPACE and _workspace_data_missing():
        _set_stage(state, Stage.INGEST)
        stage = Stage.INGEST

    render_stepper_and_tips(_stage_index(stage) + 1)

    _ensure_session_directory(state, services["sessions"])

    router = Router()
    router.register(Stage.START, lambda: page_start.render(state=state, session_service=services["sessions"]))
    router.register(Stage.INGEST, lambda: page_ingest.render(state=state, ingest_service=services["ingest"]))
    router.register(Stage.WORKSPACE, lambda: page_workspace.render(state=state, labeling_service=services["labeling"]))
    router.register(
        Stage.EXPORT,
        lambda: page_export.render(
            state=state,
            session_service=services["sessions"],
            export_service=services["export"],
        ),
    )

    router.dispatch(stage)

    _render_navigation(state, services["sessions"])
    _persist_if_dirty()


def _initialize_state() -> None:
    """Populate ``st.session_state`` with baseline defaults."""
    defaults = {
        "annotator": "",
        "save_dir": "",
        "stage": Stage.START,
        "params_confirmed": False,
        "export_done": False,
        "recording_id": "",
        "source_filename": "",
        "source_sha256": "",
        "label_map": {},
        "current_cell": 0,
        "traces": None,
        "cell_ids": None,
        "session_dir": "",
        "history": [],
        "fs_hz": 1 / 1.08,  # 1 frame every 1.08 s
        "smooth": True,
        "window": 31,
        "poly": 3,
        "show_raw": True,
        "show_smoothed": True,
        "stim_time_s": 50.0,
        "baseline_method": "rolling_median",
        "window_s": 20,
        "k": 3.0,
        "theme": "Light",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _stage_index(stage: Stage) -> int:
    """Return the index of the given stage within ``STAGE_FLOW``."""
    try:
        return STAGE_FLOW.index(stage)
    except ValueError:
        return 0


def _workspace_data_missing() -> bool:
    """Return True when workspace prerequisites are missing."""
    return (st.session_state.get("traces") is None) or (st.session_state.get("cell_ids") is None)


def _ensure_session_directory(state: StateAdapter, session_service: SessionService) -> None:
    """Create a session directory when prerequisites are satisfied."""
    if state.get_session_dir():
        return
    annotator = state.get_annotator().strip()
    save_dir = state.get_save_dir().strip()
    traces = st.session_state.get("traces")
    cell_ids = state.get_cell_ids() or []
    if not (annotator and save_dir and traces is not None and cell_ids):
        return

    recording_id = st.session_state.get("recording_id") or f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config = SessionConfig(
        annotator_id=annotator,
        save_root=Path(save_dir),
        created_at=datetime.now(timezone.utc),
    )
    metadata = {
        "fs_hz": float(st.session_state.get("fs_hz", 1 / 1.08)),
        "started_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "app_version": APP_VERSION,
        "source_path": str(st.session_state.get("source_filename", "")),
        "source_sha256": str(st.session_state.get("source_sha256", "")),
    }
    ctx = session_service.start(config, recording_id, metadata=metadata)
    state.set_session_dir(ctx.paths.session_dir)
    if st.session_state.get("recording_id") != recording_id:
        st.session_state["recording_id"] = recording_id
        mark_dirty(st.session_state)

    if cell_ids:
        write_cell_map(
            ctx.paths.session_dir,
            [{"cell_index": idx, "cell_id": cell_ids[idx]} for idx in range(len(cell_ids))],
        )

    SessionLogger(ctx.paths.session_dir)(f"session_start annotator={annotator} recording_id={recording_id}")


def _render_navigation(state: StateAdapter, session_service: SessionService) -> None:
    """Render footer navigation and manage stage transitions."""
    stage = state.get_stage()
    st.markdown("---")

    readiness: Dict[Stage, bool | None] = {
        Stage.START: bool(state.get_annotator() and state.get_save_dir()),
        Stage.INGEST: bool(st.session_state.get("traces") is not None and state.get_cell_ids()),
        Stage.WORKSPACE: None,
        Stage.EXPORT: False,
    }

    has_labels = bool(state.get_label_map())
    if not has_labels and state.get_session_dir():
        restored = session_service.hydrate_labels(Path(state.get_session_dir()))
        if restored:
            state.set_label_map_from_states(restored)
            has_labels = True
    readiness[Stage.WORKSPACE] = has_labels

    col_back, _, col_next = st.columns([1, 6, 1])
    idx = _stage_index(stage)

    with col_back:
        st.markdown('<div class="btn-nav btn-lg">', unsafe_allow_html=True)
        if st.button("Back", key="nav_back", use_container_width=True, disabled=(idx <= 0)):
            prev_stage = STAGE_FLOW[max(0, idx - 1)]
            _set_stage(state, prev_stage)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_next:
        st.markdown('<div class="btn-nav btn-lg">', unsafe_allow_html=True)
        if stage == Stage.EXPORT:
            if st.button("Start a new session", key="nav_restart", use_container_width=True):
                _reset_for_new_session(state)
                st.rerun()
        else:
            disabled = not bool(readiness.get(stage, False))
            if st.button("Next", key="nav_next", type="primary", use_container_width=True, disabled=disabled):
                next_stage = STAGE_FLOW[min(len(STAGE_FLOW) - 1, idx + 1)]
                _set_stage(state, next_stage)
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    if stage == Stage.WORKSPACE and not has_labels:
        st.caption("Save at least one label to proceed to Finish & export.")


def _reset_for_new_session(state: StateAdapter) -> None:
    """Reset session state while preserving user identity preferences."""
    token = st.session_state.get("session_token", "")
    prior_session_dir = st.session_state.get("session_dir")
    if token:
        clear_state_for_token(token, session_dir=prior_session_dir)
    st.session_state.pop(HYDRATED_FLAG, None)
    keep_keys = {"annotator", "save_dir", "theme"}
    for key in list(st.session_state.keys()):
        if key in keep_keys:
            continue
        if key == "stage":
            st.session_state[key] = Stage.START
            continue
        if key in {"label_map", "history"}:
            st.session_state[key] = {} if key == "label_map" else []
            continue
        st.session_state.pop(key, None)
    st.session_state.setdefault("traces", None)
    st.session_state.setdefault("cell_ids", None)
    _set_stage(state, Stage.START)
    mark_dirty(st.session_state)


def _apply_stage_from_query(state: StateAdapter) -> None:
    """Hydrate the active stage from URL query parameters."""
    query_stage = _read_stage_from_query()
    if query_stage is not None and state.get_stage() != query_stage:
        state.set_stage(query_stage)


def _read_stage_from_query() -> Stage | None:
    """Parse the ``?stage`` query parameter into a ``Stage`` enum."""
    value = _read_query_param("stage")
    if not value:
        return None
    value = value.lower()
    mapping = {
        "start": Stage.START,
        "ingest": Stage.INGEST,
        "workspace": Stage.WORKSPACE,
        "export": Stage.EXPORT,
    }
    return mapping.get(value)


def _ensure_query_stage(stage: Stage) -> None:
    """Synchronise the ``stage`` query parameter with the active stage."""
    desired = stage.name.lower()
    _ensure_query_param("stage", desired)


def _set_stage(state: StateAdapter, stage: Stage) -> None:
    """Set the current stage and propagate the change to the URL."""
    if state.get_stage() != stage:
        state.set_stage(stage)
    _ensure_query_stage(stage)


def _persist_if_dirty() -> None:
    """Persist session state when marked dirty."""

    if consume_dirty_flag(st.session_state):
        persist_state_from_mapping(st.session_state)


def _ensure_query_param(key: str, desired: str) -> None:
    """Ensure the query parameter ``key`` is set to ``desired``."""
    if not desired:
        return
    qp_proxy = st.query_params
    current = qp_proxy.get(key)
    if isinstance(current, list):
        current = current[0] if current else None
    if current == desired:
        return
    new_params = dict(qp_proxy)
    new_params[key] = desired
    st.query_params = new_params


def _read_query_param(key: str) -> str | None:
    """Return the first value for a query parameter if present."""
    value = st.query_params.get(key)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _ensure_session_token() -> str:
    """Ensure a stable session token exists and is reflected in the URL."""
    qp_token = _read_query_param(SESSION_TOKEN_PARAM)
    state_token = st.session_state.get("session_token")

    if qp_token and state_token and qp_token != state_token:
        state_token = qp_token
    if not state_token:
        state_token = qp_token or uuid4().hex

    st.session_state["session_token"] = state_token
    _ensure_query_param(SESSION_TOKEN_PARAM, state_token)
    return state_token


if __name__ == "__main__":
    run()
