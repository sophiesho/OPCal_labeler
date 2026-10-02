"""
HF vs HO check — area above the plateau line
============================================

A QC / labeling aid, NOT a biological feature. The plateau line is a rolling
median (default 150 s). The score is the area between the trace and that line
(only where the trace is above it), from the stimulus to the end, per minute
(ΔF/F·s per min). On the OPCal labelled set (deduplicated, Oct 2026) it
separates High Oscillatory from High Flat with ROC AUC ≈ 0.97.

All acquisition parameters (sampling rate, stimulus time) come from the user.
Source of truth for this module: OPCal feature review, plateau_qc.py v1.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources

import numpy as np
from scipy.ndimage import median_filter

_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2 compatibility

REFERENCE_FILE = "plateau_qc_reference_v1.json"


def plateau_line(dff: np.ndarray, fs_hz: float, window_s: float = 150.0) -> np.ndarray:
    """Rolling-median plateau line; window given in seconds."""
    w = max(3, round(window_s * fs_hz) | 1)  # odd number of samples
    return median_filter(np.asarray(dff, float), size=w, mode="nearest")


def area_above_plateau(
    dff: np.ndarray,
    fs_hz: float,
    stim_time_s: float,
    window_s: float = 150.0,
    start_offset_s: float = 0.0,
) -> float:
    """Separation score for one ΔF/F trace (ΔF/F·s per minute)."""
    y = np.asarray(dff, float)
    if y.size == 0 or fs_hz <= 0:
        return float("nan")
    pl = plateau_line(y, fs_hz, window_s)
    st = round((stim_time_s + start_offset_s) * fs_hz)
    st = max(0, min(st, y.size - 1))
    seg = np.clip(y[st:] - pl[st:], 0, None)
    minutes = seg.size / fs_hz / 60.0
    return float(_trapz(seg, dx=1.0 / fs_hz) / minutes) if minutes > 0 else float("nan")


@dataclass
class Reference:
    """HF / HO reference distributions of the score."""
    hf: np.ndarray
    ho: np.ndarray
    threshold: float
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.hf = np.sort(np.asarray(self.hf, float))
        self.ho = np.sort(np.asarray(self.ho, float))

    def position(self, score: float) -> dict:
        """Percentile of ``score`` within each class and a simple verdict."""
        p_hf = float(np.searchsorted(self.hf, score) / len(self.hf) * 100)
        p_ho = float(np.searchsorted(self.ho, score) / len(self.ho) * 100)
        lo, hi = np.percentile(self.ho, 10), np.percentile(self.hf, 90)
        if not np.isfinite(score):
            verdict = "n/a"
        elif score >= max(lo, hi):
            verdict = "HO-like"
        elif score <= min(lo, hi):
            verdict = "HF-like"
        else:
            verdict = "ambiguous"
        return {
            "score": float(score),
            "percentile_in_HF": p_hf,
            "percentile_in_HO": p_ho,
            "verdict": verdict,
            "threshold": float(self.threshold),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Reference:
        return cls(d["hf"], d["ho"], d["threshold"], d.get("meta", {}))


def load_reference() -> Reference:
    """Load the packaged reference distributions."""
    text = resources.files("opcal_mlt.core").joinpath("data", REFERENCE_FILE).read_text()
    return Reference.from_dict(json.loads(text))
