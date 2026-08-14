"""Workspace data/processing helpers for the labeling screen.

This module keeps *logic* separate from Streamlit UI pages.
Functions here read/write Streamlit's `session_state` (`s`) only as needed and
return plain Python/numpy objects for plotting and saving.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from opcal_mlt.core import preprocess as pp
from opcal_mlt.core import peaks as pk
from opcal_mlt.core import peaks_zv as pkzv


__all__ = [
    "ensure_workspace_state",
    "process_trace_for_cell",
]


# Helper to compute stimulus index, clamped to array bounds
def _stim_index(n: int, fs_hz: float, stim_time_s: float) -> int:
    """Return the clamped stimulus index for an array of length ``n``.

    Args:
        n: Number of samples in the trace.
        fs_hz: Sampling frequency in Hertz.
        stim_time_s: Stimulus timestamp in seconds.

    Returns:
        int: Index (0-based) clamped to ``[0, n-1]``.
    """
    if n <= 0:
        return 0
    try:
        si = int(round(float(stim_time_s) * float(fs_hz)))
    except Exception:
        si = 0
    return max(0, min(int(si), n - 1))


def ensure_workspace_state(s) -> bool:
    """Ensure mandatory keys exist for the labeling workspace.

    Args:
        s: Streamlit session state object (``st.session_state``).

    Returns:
        bool: ``True`` when traces and cell IDs exist; otherwise ``False`` after
            displaying a warning.
    """
    if "label_map" not in s or not isinstance(s.label_map, dict):
        s.label_map = {}
    if "current_cell" not in s:
        s.current_cell = 0
    if "history" not in s or not isinstance(s.get("history"), list):
        s["history"] = []

    if s.get("traces") is None or s.get("cell_ids") is None:
        st.warning("No data loaded yet. Go back to Step 2 (Upload & indexing).")
        return False

    return True


def process_trace_for_cell(s):
    """Compute processed signals and thresholds for the current cell.

    Args:
        s: Streamlit session state (``st.session_state``) containing traces and
            workspace parameters.

    Returns:
        dict: Canonical data pack consumed by plotting and export layers. Keys
            include ``x``, ``x_s``, ``base``, ``thr``, ``peaks``, ``t``, ``stim_idx``,
            ``fs_hz``, ``k``, ``smooth``, ``show_raw``, ``show_smoothed``, ``sd_const``,
            ``rect_y0_pre``, ``rect_y1_pre``, ``rect_y0_post``, ``rect_y1_post``,
            ``y_scale_mode``, and ``y_range``.

    Notes:
        The returned structure intentionally mirrors ``make_workspace_figure`` to
        keep Streamlit pages declarative.
    """
    fs_hz = float(s.get("fs_hz", 1.08))
    smooth = bool(s.get("smooth", True))
    window = int(s.get("window", 31))
    poly = int(s.get("poly", 3))
    baseline_method = str(s.get("baseline_method", "rolling_median"))
    window_s = int(s.get("window_s", 20))
    k = float(s.get("k", 3.0))
    stim_time_s = float(s.get("stim_time_s", 5.0))

    x = s.traces[:, s.current_cell].astype(float)
    x_s = pp.smooth_signal(x, window=window, polyorder=poly) if smooth else x

    # Baseline for display (user choice)
    if baseline_method.startswith("rolling"):
        base_display = pp.baseline_rolling_median(x_s, fs_hz, window_s=window_s)
    else:
        base_display = pp.baseline_percentile(x_s, q=25.0)

    # Pre-stim index in samples
    stim_idx = _stim_index(int(x_s.size), fs_hz, stim_time_s)

    # Robust SD from pre-stim residuals relative to the display baseline
    if stim_idx > 0:
        sd_const = pp.robust_sd_from_mad(x_s[:stim_idx] - base_display[:stim_idx])
    else:
        sd_const = pp.robust_sd_from_mad(x_s - base_display)

    # Threshold vector used for display and (sd_k method) peak detection
    base = base_display
    thr = base_display + float(k) * float(sd_const)

    detection_method = str(s.get("detection_method", "sd_k"))
    zv_diagnostics = {}
    peak_episodes = []  # list of (start_idx, end_idx) in x_s index space, for shading
    if detection_method == "adaptive_zv":
        min_peak_duration_s = float(s.get("min_peak_duration_s", 5.0))
        zv_result = pkzv.normalize_to_zv(x_s, fs_hz)
        state_result = pkzv.detect_peaks_with_states(
            zv_result["zv"], fs_hz,
            method="histogram_valley",
            min_peak_duration_sec=min_peak_duration_s,
        )
        # One marker per episode, at its local max, instead of a dense dot
        # for every frame inside the episode — the region itself is shaded
        # separately (see peak_episodes) so the plot reads as "region + peak"
        # rather than a string of beads.
        episodes = state_result.get("episodes", [])
        peak_episodes = list(episodes)
        peak_list = []
        for ep_start, ep_end in episodes:
            local_max_offset = int(np.argmax(x_s[ep_start:ep_end + 1]))
            peak_list.append(ep_start + local_max_offset)
        peaks = np.array(peak_list, dtype=int)
        # Reflect the adaptive threshold back into original ΔF/F units so the
        # existing plot (which draws `thr` in raw units) stays meaningful.
        thr = zv_result["baseline"] + state_result["high_threshold"] * zv_result["nu"]
        zv_diagnostics = {
            "nu": zv_result["nu"],
            "n_peaks": state_result["n_peaks"],
            "raw_n_peaks": state_result["raw_n_peaks"],
            "min_peak_duration_s": min_peak_duration_s,
        }
    else:
        peaks = pk.detect_peaks(x_s, thr, fs_hz, min_distance_s=1.0)
    t = np.arange(x.size) / fs_hz

    # Parameters for floating SD·k rectangles (pre/post), independent of baseline UI
    y0_pre, y1_pre, y0_post, y1_post, _stim_idx_rect = pp.pre_post_sd_rect_params(
        x_s, fs_hz, stim_time_s, k=k, ref="mean"
    )

    scale_mode = str(s.get("y_scale_mode", "auto"))
    y_range = None
    if scale_mode == "dataset":
        dataset_range = s.get("_y_range_dataset")
        if dataset_range is None and isinstance(s.get("traces"), np.ndarray):
            try:
                arr = np.asarray(s.traces, dtype=float)
                if arr.size:
                    y_min = float(np.nanmin(arr))
                    y_max = float(np.nanmax(arr))
                    if np.isfinite(y_min) and np.isfinite(y_max) and y_min < y_max:
                        dataset_range = (y_min, y_max)
                        s["_y_range_dataset"] = dataset_range
            except Exception:
                dataset_range = None
        if dataset_range and dataset_range[0] < dataset_range[1]:
            y_range = dataset_range
    elif scale_mode == "manual":
        try:
            y_min = float(s.get("y_manual_min"))
            y_max = float(s.get("y_manual_max"))
            if np.isfinite(y_min) and np.isfinite(y_max) and y_min < y_max:
                y_range = (y_min, y_max)
        except Exception:
            y_range = None

    return {
        "x": x,
        "x_s": x_s,
        "base": base,
        "thr": thr,
        "peaks": peaks,
        "t": t,
        "stim_idx": int(stim_idx),
        "fs_hz": fs_hz,
        "k": k,
        "smooth": smooth,
        "show_raw": bool(s.get("show_raw", True)),
        "show_smoothed": bool(s.get("show_smoothed", True)),
        "sd_const": float(sd_const),
        "rect_y0_pre": float(y0_pre),
        "rect_y1_pre": float(y1_pre),
        "rect_y0_post": float(y0_post),
        "rect_y1_post": float(y1_post),
        "y_scale_mode": scale_mode,
        "y_range": y_range,
        "detection_method": detection_method,
        "zv_diagnostics": zv_diagnostics,
        "peak_episodes": peak_episodes,
    }
