"""Sidebar parameter controls for the labeling workspace."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import streamlit as st

from opcal_mlt.app.state import StateAdapter

DEFAULT_FRAME_INTERVAL_S = 1.08  # s per frame (OPC recordings)
DEFAULT_FS_HZ = 1.0 / DEFAULT_FRAME_INTERVAL_S

# Define UI option lists near the top so the choices remain centralized.
_BASELINE_OPTIONS: Sequence[Tuple[str, str]] = (
    ("rolling_median", "Rolling median (window)"),
    ("percentile (25)", "Percentile (25th)"),
)

_SCALE_OPTIONS: Sequence[Tuple[str, str]] = (
    ("auto", "Adaptive (per cell)"),
    ("dataset", "Fix to dataset extremes"),
    ("manual", "Manual range"),
)


def render_sidebar_params(state: StateAdapter) -> None:
    """Render sidebar parameters and persist values in session state.

    This implementation keeps all form controls in one place, consistently updates
    ``StateAdapter`` values, and caches dataset-wide statistics for downstream
    logic. The sidebar is organised into small sections so that Streamlit can
    render them responsively, even as the widget set grows.
    """

    dataset_range = _compute_dataset_range(state.get("traces"))
    state.set("_y_range_dataset", dataset_range)  # consumed by ``workspace_logic``

    with st.sidebar:
        st.markdown("### Labeling parameters")
        _render_sampling_section(state)
        _render_visibility_section(state)
        _render_smoothing_section(state)
        _render_baseline_section(state)
        _render_threshold_section(state)

        st.markdown("### ΔF/F scale")
        _render_scale_section(state, dataset_range)


def _render_sampling_section(state: StateAdapter) -> None:
    """Acquisition timing. The lab specifies time per frame (e.g. 1 frame every 1.08 s);
    internally the app keeps the sampling rate ``fs_hz = 1 / frame_interval_s``."""
    st.markdown("#### Acquisition")
    fs_current = float(state.get("fs_hz", DEFAULT_FS_HZ))
    frame_interval = st.number_input(
        "Frame interval (s per frame)",
        min_value=0.001,
        value=float(round(1.0 / fs_current, 4)) if fs_current > 0 else DEFAULT_FRAME_INTERVAL_S,
        step=0.01,
        format="%.3f",
        help=(
            "Time between two consecutive frames, in seconds. OPC recordings: 1 frame every "
            "1.08 s (= sampling rate 0.926 Hz). Not frames per second."
        ),
        key="sidebar_frame_interval",
    )
    state.set("fs_hz", 1.0 / float(frame_interval))
    st.caption(f"Sampling rate = {1.0 / float(frame_interval):.3f} Hz")


def _render_visibility_section(state: StateAdapter) -> None:
    st.markdown("#### Signal layers")
    col_raw, col_smooth = st.columns(2)
    with col_raw:
        show_raw = st.checkbox(
            "Raw signal",
            value=bool(state.get("show_raw", True)),
            help="Toggle the original unfiltered trace.",
            key="sidebar_show_raw",
        )
    with col_smooth:
        show_smoothed = st.checkbox(
            "Smoothed",
            value=bool(state.get("show_smoothed", True)),
            help="Toggle the Savitzky–Golay smoothed trace (when enabled).",
            key="sidebar_show_smoothed",
        )
    state.set("show_raw", bool(show_raw))
    state.set("show_smoothed", bool(show_smoothed))


def _render_smoothing_section(state: StateAdapter) -> None:
    st.markdown("#### Smoothing")
    smooth_enabled = st.checkbox(
        "Apply Savitzky–Golay smoothing",
        value=bool(state.get("smooth", True)),
        help=(
            "Phase-preserving smoothing that reduces noise without shifting peaks. "
            "Disable to view only the raw trace."
        ),
        key="sidebar_smoothing_toggle",
    )
    state.set("smooth", bool(smooth_enabled))

    if smooth_enabled:
        window_default = _to_odd_int(int(state.get("window", 31)), minimum=5, maximum=101)
        window = st.slider(
            "Smoothing window (samples)",
            min_value=5,
            max_value=101,
            value=window_default,
            step=2,
            help=(
                "Savitzky–Golay window length (odd samples). Larger windows smooth more strongly "
                "but may flatten short events. Typical: 21–61."
            ),
            key="sidebar_smoothing_window",
        )
        state.set("window", int(window))

        poly_max = max(1, min(5, int(window) - 1))
        poly_default = int(state.get("poly", 3))
        poly_default = max(1, min(poly_default, poly_max))
        poly = st.slider(
            "Polynomial order",
            min_value=1,
            max_value=poly_max,
            value=poly_default,
            help=(
                "Order of the fitted polynomial inside each window. Lower values = gentler smoothing; "
                "must remain below the window size. Typical: 2–3."
            ),
            key="sidebar_smoothing_poly",
        )
        state.set("poly", int(poly))


def _render_baseline_section(state: StateAdapter) -> None:
    st.markdown("#### Baseline")
    baseline_value = str(state.get("baseline_method", "rolling_median"))
    baseline_options = {value: label for value, label in _BASELINE_OPTIONS}
    if baseline_value not in baseline_options:
        baseline_value = "rolling_median"

    baseline_choice = st.radio(
        "Method",
        options=list(baseline_options.keys()),
        index=list(baseline_options.keys()).index(baseline_value),
        format_func=lambda value: baseline_options[value],
        key="sidebar_baseline_method",
    )
    state.set("baseline_method", baseline_choice)

    if baseline_choice.startswith("rolling"):
        window_default = int(state.get("window_s", 20))
        window_default = max(5, min(window_default, 60))
        window_s = st.slider(
            "Rolling median window (s)",
            min_value=5,
            max_value=60,
            value=window_default,
            help="Temporal window used for the rolling median baseline.",
            key="sidebar_baseline_window",
        )
        state.set("window_s", int(window_s))


def _render_threshold_section(state: StateAdapter) -> None:
    st.markdown("#### Thresholds")

    method_options = {
        "sd_k": "SD threshold (k × SD)",
        "adaptive_zv": "Adaptive (Zv + histogram valley)",
    }
    method_value = str(state.get("detection_method", "sd_k"))
    if method_value not in method_options:
        method_value = "sd_k"
    method_choice = st.radio(
        "Peak detection method",
        options=list(method_options.keys()),
        index=list(method_options.keys()).index(method_value),
        format_func=lambda value: method_options[value],
        help=(
            "SD threshold uses a fixed baseline + k×SD line. Adaptive (Zv) finds a "
            "per-cell threshold from the histogram valley between noise and activity, "
            "then filters out episodes shorter than the minimum peak duration."
        ),
        key="sidebar_detection_method",
    )
    state.set("detection_method", method_choice)

    if method_choice == "sd_k":
        k_value = st.slider(
            "SD threshold k",
            min_value=1.0,
            max_value=6.0,
            value=float(state.get("k", 3.0)),
            step=0.5,
            help="Standard deviation multiplier applied on top of the baseline to flag events.",
            key="sidebar_threshold_k",
        )
        state.set("k", float(k_value))
    else:
        min_duration_value = st.slider(
            "Minimum peak duration (s)",
            min_value=0.0,
            max_value=20.0,
            value=float(state.get("min_peak_duration_s", 5.0)),
            step=0.5,
            help=(
                "Episodes shorter than this are treated as noise, not peaks. Default (5s) "
                "was calibrated against the low_activity class: it removes ~59% of spurious "
                "short episodes there while keeping ~76% of real events in oscillatory classes."
            ),
            key="sidebar_min_peak_duration",
        )
        state.set("min_peak_duration_s", float(min_duration_value))

    # Stimulus: the lab often logs it as a frame number; the app stores seconds.
    fs_now = float(state.get("fs_hz", DEFAULT_FS_HZ))
    unit = st.radio(
        "Stimulus given as",
        ["frame number", "seconds"],
        index=0 if state.get("stim_unit", "frame number") == "frame number" else 1,
        horizontal=True,
        key="sidebar_stim_unit",
    )
    state.set("stim_unit", unit)
    stim_s_current = float(state.get("stim_time_s", 50.0))
    if unit == "frame number":
        stim_frame = st.number_input(
            "Stimulus frame",
            min_value=0,
            value=int(round(stim_s_current * fs_now)),
            step=1,
            help="Frame at which the stimulus was given (counted from the first frame of the loaded trace).",
            key="sidebar_stim_frame",
        )
        stim_time = float(stim_frame) / fs_now
        st.caption(f"= {stim_time:.1f} s")
    else:
        stim_time = st.number_input(
            "Stimulus time (s)",
            min_value=0.0,
            value=stim_s_current,
            step=1.0,
            help="Timestamp at which stimulation starts; used to split pre/post statistics.",
            key="sidebar_stim_time",
        )
        st.caption(f"= frame {round(float(stim_time) * fs_now)}")
    state.set("stim_time_s", float(stim_time))


def _render_scale_section(state: StateAdapter, dataset_range: Optional[Tuple[float, float]]) -> None:
    scale_value = str(state.get("y_scale_mode", "auto"))
    scale_options = {value: label for value, label in _SCALE_OPTIONS}
    if scale_value not in scale_options:
        scale_value = "auto"

    scale_choice = st.radio(
        "Mode",
        options=list(scale_options.keys()),
        index=list(scale_options.keys()).index(scale_value),
        format_func=lambda value: scale_options[value],
        help=(
            "Control how the y-axis range is chosen. Adaptive follows each cell, while fixed modes "
            "maintain consistent scaling across the dataset."
        ),
        key="sidebar_scale_mode",
    )
    state.set("y_scale_mode", scale_choice)

    if dataset_range and dataset_range[0] < dataset_range[1]:
        st.caption(f"Dataset extremes: {dataset_range[0]:.3f} … {dataset_range[1]:.3f}")
    elif scale_choice == "dataset":
        st.warning("Unable to compute dataset extremes for fixed scaling.")

    if scale_choice == "manual":
        default_min = float(state.get("y_manual_min", dataset_range[0] if dataset_range else -0.5))
        default_max = float(state.get("y_manual_max", dataset_range[1] if dataset_range else 0.5))

        manual_min = st.number_input(
            "Manual min",
            value=default_min,
            step=0.1,
            format="%.3f",
            key="sidebar_scale_manual_min",
        )
        manual_max = st.number_input(
            "Manual max",
            value=default_max if default_max > manual_min else manual_min + 1.0,
            step=0.1,
            format="%.3f",
            key="sidebar_scale_manual_max",
        )
        state.set("y_manual_min", float(manual_min))
        state.set("y_manual_max", float(manual_max))

        if float(manual_min) >= float(manual_max):
            st.error("Manual max must be greater than min.")


def _compute_dataset_range(traces) -> Optional[Tuple[float, float]]:
    """Return the global min/max for the loaded traces, if available."""
    if traces is None:
        return None
    try:
        arr = np.asarray(traces, dtype=float)
    except Exception:
        return None
    if arr.size == 0:
        return None

    try:
        y_min = float(np.nanmin(arr))
        y_max = float(np.nanmax(arr))
    except Exception:
        return None

    if not (np.isfinite(y_min) and np.isfinite(y_max) and y_min < y_max):
        return None
    return (y_min, y_max)


def _to_odd_int(value: int, *, minimum: int, maximum: int) -> int:
    """Clamp ``value`` to ``[minimum, maximum]`` and adjust to the nearest odd int."""
    value = int(value)
    value = max(minimum, min(value, maximum))
    if value % 2 == 0:
        value = value + 1 if value < maximum else value - 1
    if value < minimum:
        value = minimum if minimum % 2 == 1 else minimum + 1
    if value > maximum:
        value = maximum if maximum % 2 == 1 else maximum - 1
    return value


__all__ = ["render_sidebar_params"]
