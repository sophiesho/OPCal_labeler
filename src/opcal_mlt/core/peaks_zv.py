"""
Adaptive (Zv-based) Peak Detection
===================================

Ported from Yoni's OPCal peak-detection algorithm. Unlike the fixed
``k × SD`` threshold in ``peaks.detect_peaks``, this method:

- Estimates per-ROI noise (ν) from frame-to-frame differences, which is
  robust to occasional large transients (unlike a plain SD).
- Normalizes the detrended signal to Zv(t) = (x - baseline) / ν.
- Finds the peak threshold as the histogram valley between the noise
  cluster and the activity cluster in Zv(t) — adaptive per ROI, not a
  fixed multiplier.
- Applies a minimum peak duration filter so that brief single-frame noise
  spikes crossing the threshold aren't counted as events.

Default ``min_peak_duration_sec=5.0`` was calibrated empirically: on the
OPCal low_activity class (no real transients expected), raw unfiltered
high-state episodes had a median duration of ~4.6s; a 5s cutoff removes
~59% of those spurious episodes while retaining ~76% of real events in
the oscillatory / high_oscillatory classes.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d


def compute_noise_level(trace: np.ndarray, fr: float, baseline_frames: int | None = None) -> float:
    """Robust per-ROI noise estimate: median(|frame-to-frame diff|) / sqrt(fr)."""
    baseline = trace[:baseline_frames] if baseline_frames is not None else trace
    diffs = np.abs(np.diff(baseline))
    median_diff = np.nanmedian(diffs)
    return float(median_diff / np.sqrt(fr))


def compute_local_baseline(trace: np.ndarray, fr: float, window_sec: float = 50.0) -> np.ndarray:
    """Rolling-median baseline B(t), window given in seconds."""
    window_frames = int(fr * window_sec)
    if window_frames % 2 == 0:
        window_frames += 1
    return median_filter(trace, size=window_frames, mode="nearest")


def normalize_to_zv(trace: np.ndarray, fr: float, window_sec: float = 50.0) -> dict:
    """Return Zv(t), ν, the local baseline, and the detrended trace."""
    nu = compute_noise_level(trace, fr)
    baseline = compute_local_baseline(trace, fr, window_sec)
    detrended = trace - baseline
    zv = detrended / nu if nu > 0 else detrended
    return {"zv": zv, "nu": nu, "baseline": baseline, "detrended": detrended}


def find_threshold_histogram_valley(zv_trace: np.ndarray) -> float:
    """Adaptive per-ROI threshold: the histogram valley in the positive Zv(t) region."""
    hist, bin_edges = np.histogram(zv_trace, bins=200)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    positive_mask = bin_centers > 0
    if not np.any(positive_mask):
        return float(np.median(zv_trace))

    positive_hist = hist[positive_mask]
    positive_bins = bin_centers[positive_mask]

    kernel_size = max(3, len(positive_hist) // 20)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smooth_hist = uniform_filter1d(positive_hist, size=kernel_size, mode="nearest")

    min_idx = np.argmin(smooth_hist)
    return float(positive_bins[min_idx])


def detect_peaks_with_states(
    zv_trace: np.ndarray,
    fr: float,
    method: str = "histogram_valley",
    min_peak_duration_sec: float = 5.0,
) -> dict:
    """
    Classify each frame as low/transition/high state and extract peak episodes
    that meet ``min_peak_duration_sec``.

    Returns a dict with: states, low_threshold, high_threshold, peak_indices,
    n_peaks, peak_durations, total_peak_time, raw_n_peaks, raw_durations,
    min_peak_duration_sec_used.
    """
    T = len(zv_trace)

    if method == "histogram_valley":
        high_threshold = find_threshold_histogram_valley(zv_trace)
        low_threshold = high_threshold * 0.7
    else:
        mean_val = float(np.mean(zv_trace))
        std_val = float(np.std(zv_trace))
        low_threshold = mean_val + 1.0 * std_val
        high_threshold = mean_val + 2.0 * std_val

    states = np.zeros(T, dtype=int)
    states[zv_trace > high_threshold] = 1
    states[zv_trace < low_threshold] = -1

    # Raw episodes (before duration filtering)
    raw_episodes = []
    in_peak = states[0] == 1
    episode_start = 0
    for i in range(1, T):
        currently_high = states[i] == 1
        if currently_high and not in_peak:
            episode_start = i
            in_peak = True
        elif not currently_high and in_peak:
            raw_episodes.append((episode_start, i - 1))
            in_peak = False
    if in_peak:
        raw_episodes.append((episode_start, T - 1))

    raw_durations = [(e - s + 1) / fr for s, e in raw_episodes]

    # Apply minimum-duration filter: demote short episodes back to "transition"
    min_frames = max(1, int(round(min_peak_duration_sec * fr)))
    kept_episodes = []
    for (s, e) in raw_episodes:
        if (e - s + 1) >= min_frames:
            kept_episodes.append((s, e))
        else:
            states[s:e + 1] = 0

    peak_indices = np.where(states == 1)[0]
    peak_durations = [(e - s + 1) / fr for s, e in kept_episodes]

    return {
        "states": states,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "peak_indices": peak_indices,
        "n_peaks": len(kept_episodes),
        "peak_durations": peak_durations,
        "total_peak_time": float(np.sum(states == 1) / fr),
        "raw_n_peaks": len(raw_episodes),
        "raw_durations": raw_durations,
        "min_peak_duration_sec_used": min_peak_duration_sec,
        "episodes": kept_episodes,
    }
