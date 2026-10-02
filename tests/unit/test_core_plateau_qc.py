import numpy as np
import pytest

from opcal_mlt.core import plateau_qc as pq

FS = 1 / 1.08


def _trace(osc_amp=0.0, n=810, stim_s=110.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    y = np.where(t > stim_s, 0.6, 0.0) + rng.normal(0, 0.005, n)
    y += osc_amp * np.clip(np.sin(2 * np.pi * t / 60.0), 0, None) * (t > stim_s)
    return y


def test_flat_plateau_scores_low_and_oscillating_scores_high():
    flat = pq.area_above_plateau(_trace(0.0), FS, 110.0)
    osc = pq.area_above_plateau(_trace(0.3), FS, 110.0)
    assert flat < 0.5
    assert osc > 5 * flat


def test_score_is_nonnegative_and_finite():
    s = pq.area_above_plateau(_trace(0.1), FS, 110.0)
    assert np.isfinite(s) and s >= 0


def test_empty_trace_returns_nan():
    assert np.isnan(pq.area_above_plateau(np.array([]), FS, 0.0))


def test_packaged_reference_loads_and_classifies():
    ref = pq.load_reference()
    assert len(ref.hf) > 100 and len(ref.ho) > 100
    assert ref.position(0.1)["verdict"] == "HF-like"
    assert ref.position(20.0)["verdict"] == "HO-like"
    p = ref.position(ref.threshold)
    assert 0 <= p["percentile_in_HF"] <= 100 and 0 <= p["percentile_in_HO"] <= 100


def test_window_in_seconds_scales_with_sampling_rate():
    y = _trace(0.3)
    a = pq.plateau_line(y, FS, 150.0)
    b = pq.plateau_line(np.repeat(y, 2), 2 * FS, 150.0)[::2]
    assert np.corrcoef(a, b)[0, 1] > 0.99


@pytest.mark.parametrize("stim", [0.0, 110.0, 5000.0])
def test_stim_time_out_of_range_is_clamped(stim):
    assert np.isfinite(pq.area_above_plateau(_trace(0.2), FS, stim))
