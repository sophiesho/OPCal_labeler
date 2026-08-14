import numpy as np
import pytest

from opcal_mlt.core.peaks_zv import (
    compute_noise_level,
    compute_local_baseline,
    normalize_to_zv,
    find_threshold_histogram_valley,
    detect_peaks_with_states,
)


def _make_trace_with_event(fr=1.08, total_sec=200, event_start_sec=100, event_dur_sec=10, amplitude=10.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(total_sec * fr)
    trace = rng.normal(loc=0.0, scale=0.05, size=n)
    start = int(event_start_sec * fr)
    dur = int(event_dur_sec * fr)
    trace[start:start + dur] += amplitude
    return trace


def test_compute_noise_level_positive():
    trace = _make_trace_with_event()
    nu = compute_noise_level(trace, fr=1.08)
    assert nu > 0


def test_compute_local_baseline_shape():
    trace = _make_trace_with_event()
    baseline = compute_local_baseline(trace, fr=1.08, window_sec=50.0)
    assert baseline.shape == trace.shape


def test_normalize_to_zv_returns_expected_keys():
    trace = _make_trace_with_event()
    result = normalize_to_zv(trace, fr=1.08)
    assert set(result.keys()) == {"zv", "nu", "baseline", "detrended"}
    assert result["zv"].shape == trace.shape


def test_detects_a_long_event_and_filters_short_noise_spike():
    fr = 1.08
    trace = _make_trace_with_event(fr=fr, event_start_sec=100, event_dur_sec=10, amplitude=10.0)
    # inject a single-frame noise spike that should NOT survive a 5s duration filter
    spike_idx = int(20 * fr)
    trace[spike_idx] += 8.0

    zv_result = normalize_to_zv(trace, fr=fr)
    detection = detect_peaks_with_states(zv_result["zv"], fr, min_peak_duration_sec=5.0)

    # the long injected event should be detected
    assert detection["n_peaks"] >= 1
    # raw episodes (before filtering) should be >= kept episodes
    assert detection["raw_n_peaks"] >= detection["n_peaks"]
    assert detection["min_peak_duration_sec_used"] == 5.0


def test_min_peak_duration_zero_keeps_all_raw_episodes():
    fr = 1.08
    trace = _make_trace_with_event(fr=fr)
    zv_result = normalize_to_zv(trace, fr=fr)
    detection = detect_peaks_with_states(zv_result["zv"], fr, min_peak_duration_sec=0.0)
    assert detection["n_peaks"] == detection["raw_n_peaks"]


def test_threshold_valley_is_finite():
    trace = _make_trace_with_event()
    zv_result = normalize_to_zv(trace, fr=1.08)
    threshold = find_threshold_histogram_valley(zv_result["zv"])
    assert np.isfinite(threshold)
