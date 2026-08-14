"""
Preprocessing Utilities
======================

Signal preprocessing helpers for OPCAL‑Labeler.
Includes Savitzky–Golay smoothing, baseline estimation, robust SD calculation, and threshold construction.
All functions are designed for robustness and interpretability in research workflows.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
from scipy.signal import savgol_filter

# --- Constants ---------------------------------------------------------------
MAD_TO_SD: float = 1.4826   # Scale MAD → Gaussian-equivalent SD
EPS_SD: float = 1e-9        # Tiny positive floor for SD to avoid zero-width bands
DEFAULT_ROLLING_WINDOW_S: float = 20.0  # Typical rolling-median window (seconds)


def smooth_signal(x: np.ndarray, window: int = 31, polyorder: int = 3) -> np.ndarray:
    """
    Return a Savitzky–Golay smoothed copy of a 1D signal.

    Args:
        x (np.ndarray): 1D input signal of shape (T,).
        window (int, optional): Window length (samples). Will be coerced to an odd integer ≥ 5. Default is 31.
        polyorder (int, optional): Polynomial order for the filter (1–5 typical). Default is 3.

    Returns:
        np.ndarray: Smoothed signal of same shape as input.

    Notes:
        If the signal is shorter than the effective window, the original signal is returned unchanged to avoid edge artifacts.
    """

    # Ensure an odd window ≥ 5, clamp polyorder to a sensible range
    window = max(5, int(window) | 1)  # bitwise OR with 1 -> odd
    polyorder = max(1, min(int(polyorder), 5))

    if x.size < window:
        return x.copy()

    return savgol_filter(x, window_length=window, polyorder=polyorder, mode="interp")


def baseline_rolling_median(x: np.ndarray, fs_hz: float, window_s: float = DEFAULT_ROLLING_WINDOW_S) -> np.ndarray:
    """
    Estimate a slowly varying baseline using a rolling median window.

    Args:
        x (np.ndarray): 1D signal of shape (T,).
        fs_hz (float): Sampling rate in Hertz.
        window_s (float, optional): Median window size in seconds (typ. 10–30 s). Default is 20.0.

    Returns:
        np.ndarray: Baseline array of shape (T,) aligned to ``x``.
    """

    # Convert seconds to samples and make the window odd (≥ 1)
    w = int(max(1, round(window_s * fs_hz)))
    if w % 2 == 0:
        w += 1

    pad = w // 2
    # Edge‑padding to avoid shrinking at the boundaries
    xp = np.pad(x, (pad, pad), mode="edge")
    med = np.empty_like(x, dtype=float)

    # Simple rolling median (O(T·w)); for T up to order ~1e5 this is fine
    for i in range(x.size):
        med[i] = np.median(xp[i : i + w])

    return med


def baseline_percentile(x: np.ndarray, q: float = 25.0) -> np.ndarray:
    """
    Return a flat baseline at a global percentile of the signal.

    Args:
        x (np.ndarray): 1D signal of shape (T,).
        q (float, optional): Percentile in [0, 100]. Typically 20–30 captures the quiescent level. Default is 25.0.

    Returns:
        np.ndarray: Baseline array of shape (T,) with constant value at the given percentile.
    """

    b = np.percentile(x, q)
    return np.full_like(x, fill_value=float(b), dtype=float)


def robust_sd_from_mad(x: np.ndarray) -> float:
    """
    Estimate a robust SD as ``MAD_TO_SD × median(|x - median(x)|)``.

    Args:
        x (np.ndarray): 1D signal array.

    Returns:
        float: Robust SD estimate. Falls back to unbiased standard deviation when MAD is degenerate and clamps to a tiny positive floor (``EPS_SD``).
    """
    mad = np.median(np.abs(x - np.median(x)))
    sd = MAD_TO_SD * mad
    if not np.isfinite(sd) or sd == 0.0:
        sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    return float(max(sd, EPS_SD))


def pre_post_sd_rect_params(
    x: np.ndarray,
    fs_hz: float,
    stim_time_s: float,
    k: float = 3.0,
    ref: str = "mean",
) -> Tuple[float, float, float, float, int]:
    """
    Compute parameters for two *floating* SD·k rectangles (pre/post stim).

    Both rectangles are scaled by the **pre-stimulus (baseline) SD**, not each
    segment's own SD. Technical/instrument noise is present throughout the
    recording, including during the quiet pre-stimulus period, so the
    pre-stim SD gives a clean noise-only estimate. Scaling the post-stim
    rectangle by the post-stim segment's own SD would fold real calcium
    activity into the "noise" estimate, inflating the band and making it
    harder to tell a significant post-stim fluctuation from baseline noise.
    This mirrors the logic used for the actual peak-detection threshold
    (``thr = baseline + k * sd_pre`` in ``workspace_logic.py``), so the
    visualized bands now stay consistent with what actually drives peak
    detection.

    Args:
        x (np.ndarray): 1D signal of shape (T,).
        fs_hz (float): Sampling rate in Hertz.
        stim_time_s (float): Time (seconds) at which stimulation starts.
        k (float, optional): Multiplier applied to the **post** rectangle height via ``ref_post + k·SD_pre``. Default is 3.0.
        ref (str, optional): How to choose the constant reference level per segment. {"mean", "median", "zero"}. Default is "mean".

    Returns:
        Tuple[float, float, float, float, int]: Rectangle base/top values for the pre/post segments and the stimulus index used for splitting.
    """
    n = int(x.size)
    stim_idx = int(max(0, min(n - 1, round(float(stim_time_s) * float(fs_hz)))))

    # --- Pre segment: establishes the baseline noise SD used for BOTH rectangles ---
    pre = x[:stim_idx] if stim_idx > 0 else x
    ref_normalized = ref.lower().strip()
    if ref_normalized == "median":
        ref_normalized = "mean"

    if ref_normalized == "zero":
        ref_pre = 0.0
    else:
        ref_pre = float(np.mean(pre)) if pre.size else 0.0
    sd_pre = robust_sd_from_mad(pre - ref_pre)
    sd_pre = float(max(sd_pre, 1e-9))
    y0_pre = ref_pre
    y1_pre = ref_pre + sd_pre

    # --- Post segment: own reference level (where the signal actually sits),
    # but scaled by the SAME baseline sd_pre rather than the post segment's
    # own SD, so the band reflects "how many baseline-noise-SDs above normal"
    # the post-stim level is.
    post = x[stim_idx:] if stim_idx < n else x[-1:]
    if ref_normalized == "zero":
        ref_post = 0.0
    else:
        ref_post = float(np.mean(post)) if post.size else 0.0
    y0_post = ref_post
    y1_post = ref_post + float(k) * sd_pre

    return float(y0_pre), float(y1_pre), float(y0_post), float(y1_post), int(stim_idx)
