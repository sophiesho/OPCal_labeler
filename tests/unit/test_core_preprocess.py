import numpy as np
import pytest
from scipy.signal import savgol_filter

from opcal_mlt.core.preprocess import (
    baseline_percentile,
    baseline_rolling_median,
    pre_post_sd_rect_params,
    robust_sd_from_mad,
    smooth_signal,
)


def test_smooth_signal_short_series_returns_copy():
    x = np.arange(4.0)
    smoothed = smooth_signal(x, window=11, polyorder=4)
    assert np.array_equal(smoothed, x)
    assert smoothed is not x


def test_smooth_signal_matches_savgol_with_normalized_params():
    x = np.linspace(0, 1, 100)
    result = smooth_signal(x, window=12, polyorder=6)
    expected = savgol_filter(x, window_length=13, polyorder=5, mode="interp")
    assert np.allclose(result, expected)


def test_baseline_rolling_median_matches_manual_window():
    x = np.array([1.0, 2.0, 100.0, 2.0, 1.0])
    baseline = baseline_rolling_median(x, fs_hz=1.0, window_s=3.0)
    assert np.array_equal(baseline, np.array([1.0, 2.0, 2.0, 2.0, 1.0]))


def test_baseline_percentile_returns_flat_array():
    x = np.array([0.0, 1.0, 10.0])
    baseline = baseline_percentile(x, q=50.0)
    assert np.array_equal(baseline, np.full_like(x, 1.0, dtype=float))


def test_robust_sd_from_mad_regular_and_degenerate_cases():
    x = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
    sd = robust_sd_from_mad(x)
    assert sd == pytest.approx(1.4826 * 3.0)

    flat = np.ones(5)
    sd_flat = robust_sd_from_mad(flat)
    assert sd_flat == pytest.approx(1e-9)


def test_pre_post_sd_rect_params_computes_expected_bounds():
    x = np.array([0.0, 0.0, 1.0, 2.0, 1.0, 0.0])
    y0_pre, y1_pre, y0_post, y1_post, stim_idx = pre_post_sd_rect_params(
        x, fs_hz=2.0, stim_time_s=1.0, k=3.0, ref="mean"
    )
    assert stim_idx == 2
    assert y0_pre == pytest.approx(0.0)
    assert y1_pre == pytest.approx(1e-9)
    assert y0_post == pytest.approx(1.0)
    # Post rectangle height is now scaled by the PRE-stim (baseline) SD, not
    # the post segment's own SD, so real post-stim activity doesn't inflate
    # the noise estimate used for the band.
    assert y1_post == pytest.approx(1.0 + 3.0 * robust_sd_from_mad(x[:2] - 0.0))


def test_pre_post_sd_rect_params_zero_reference():
    x = np.array([0.5, 0.3, 0.2, 0.1])
    y0_pre, y1_pre, y0_post, y1_post, stim_idx = pre_post_sd_rect_params(
        x, fs_hz=10.0, stim_time_s=0.1, k=2.0, ref="zero"
    )
    assert stim_idx == 1
    assert y0_pre == pytest.approx(0.0)
    assert y0_post == pytest.approx(0.0)
    assert y1_pre > 0.0
    assert y1_post > y1_pre
