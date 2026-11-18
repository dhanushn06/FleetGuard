import numpy as np
import pytest

from signal_processing.sampling import (
    estimate_sampling_rate,
    generate_uniform_timestamps,
    resample_nd_signal,
    resample_signal_packet,
)
from signal_processing.validation import REQUIRED_RGB_REGIONS


def make_valid_packet(n_samples: int = 60, fps: float = 30.0, jitter: float = 0.0):
    """Create a synthetic packet; optional timestamp jitter simulates webcam timing."""
    rng = np.random.default_rng(123)
    base_t = np.arange(n_samples, dtype=np.float64) / fps
    if jitter > 0:
        noise = rng.uniform(-jitter, jitter, size=n_samples)
        noise[0] = 0.0
        t = base_t + noise
        t = np.maximum.accumulate(t + 1e-6)  # enforce strictly increasing
    else:
        t = base_t

    rgb = {}
    for idx, region in enumerate(REQUIRED_RGB_REGIONS):
        phase = idx * 0.2
        # Simple deterministic signals per channel for interpolation checks
        r = 100.0 + 10.0 * np.sin(2 * np.pi * 1.0 * t + phase)
        g = 120.0 + 5.0 * np.cos(2 * np.pi * 1.2 * t + phase)
        b = 110.0 + 3.0 * np.sin(2 * np.pi * 0.8 * t + phase)
        rgb[region] = np.stack([r, g, b], axis=1)

    motion = 0.1 * np.sin(2 * np.pi * 0.5 * t)
    brightness = 150.0 + 2.0 * t
    track_ok = np.ones(n_samples, dtype=bool)
    track_ok[15:20] = False
    return {
        "t": t,
        "rgb": rgb,
        "motion": motion,
        "track_ok": track_ok,
        "brightness": brightness,
    }


def test_estimate_sampling_rate_uniform_median_and_mean():
    t = np.arange(0, 2.0, 1.0 / 25.0)
    fs_median = estimate_sampling_rate(t, method="median")
    fs_mean = estimate_sampling_rate(t, method="mean")
    assert fs_median == pytest.approx(25.0, rel=1e-12, abs=1e-12)
    assert fs_mean == pytest.approx(25.0, rel=1e-12, abs=1e-12)


def test_estimate_sampling_rate_jitter_median_is_robust():
    # Mostly 30 Hz with one large gap/outlier interval; median should stay near 30 Hz.
    t = [0.0]
    dt_nominal = 1.0 / 30.0
    for i in range(1, 30):
        if i == 10:
            t.append(t[-1] + 0.2)  # outlier gap
        else:
            t.append(t[-1] + dt_nominal)
    t = np.array(t, dtype=np.float64)

    fs_median = estimate_sampling_rate(t, method="median")
    fs_mean = estimate_sampling_rate(t, method="mean")
    assert fs_median == pytest.approx(30.0, rel=1e-6, abs=1e-6)
    assert fs_mean < fs_median  # mean pulled down by outlier


def test_estimate_sampling_rate_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        estimate_sampling_rate(np.array([0.0]))  # too few samples
    with pytest.raises(ValueError):
        estimate_sampling_rate(np.array([0.0, 0.1, 0.1]))  # not strictly increasing
    with pytest.raises(ValueError):
        estimate_sampling_rate(np.array([0.0, 0.1, 0.2]), method="unknown")


def test_generate_uniform_timestamps_covers_range():
    t = np.array([0.1, 0.14, 0.19, 0.23, 0.27], dtype=np.float64)
    fs = 20.0
    tu = generate_uniform_timestamps(t, fs)

    assert tu.ndim == 1
    assert tu.size >= 2
    assert tu[0] == pytest.approx(t[0])
    assert tu[-1] <= t[-1] + 1e-12
    dt = np.diff(tu)
    assert np.allclose(dt, 1.0 / fs, rtol=0, atol=1e-12)


def test_generate_uniform_timestamps_rejects_invalid_fs():
    t = np.array([0.0, 0.1, 0.2])
    with pytest.raises(ValueError):
        generate_uniform_timestamps(t, fs=0.0)
    with pytest.raises(ValueError):
        generate_uniform_timestamps(t, fs=-10.0)


def test_resample_nd_signal_linear_interpolation_1d_and_2d():
    t = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    t_uniform = np.array([0.0, 0.5, 1.5, 2.5, 3.0], dtype=np.float64)

    y1d = 2.0 * t + 1.0
    y1d_rs = resample_nd_signal(t, y1d, t_uniform, kind="linear")
    assert np.allclose(y1d_rs, 2.0 * t_uniform + 1.0)

    y2d = np.stack([2.0 * t + 1.0, -t + 4.0], axis=1)
    y2d_rs = resample_nd_signal(t, y2d, t_uniform, kind="linear")
    expected = np.stack([2.0 * t_uniform + 1.0, -t_uniform + 4.0], axis=1)
    assert y2d_rs.shape == (t_uniform.size, 2)
    assert np.allclose(y2d_rs, expected)


def test_resample_nd_signal_nearest_for_step_like_values():
    t = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    values = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
    t_uniform = np.array([0.2, 0.8, 1.2, 1.8, 2.6], dtype=np.float64)
    rs = resample_nd_signal(t, values, t_uniform, kind="nearest")
    # 2.6 is closer to 3.0 than 2.0, so nearest value comes from t=3.0 (0.0).
    assert np.array_equal(rs, np.array([0.0, 1.0, 1.0, 1.0, 0.0]))


def test_resample_signal_packet_produces_uniform_packet():
    packet = make_valid_packet(n_samples=50, fps=30.0, jitter=0.001)
    out = resample_signal_packet(packet, target_fs=30.0)

    assert set(out.keys()) == {"t", "rgb", "motion", "track_ok", "brightness"}
    t = out["t"]
    assert t.ndim == 1 and t.size >= 2
    assert np.allclose(np.diff(t), 1.0 / 30.0, rtol=0, atol=1e-12)

    n = t.size
    for region in REQUIRED_RGB_REGIONS:
        assert region in out["rgb"]
        assert out["rgb"][region].shape == (n, 3)
        assert np.all(np.isfinite(out["rgb"][region]))

    assert out["motion"].shape == (n,)
    assert out["brightness"].shape == (n,)
    assert out["track_ok"].shape == (n,)
    assert out["track_ok"].dtype == bool


def test_resample_signal_packet_estimates_fs_when_not_provided():
    packet = make_valid_packet(n_samples=45, fps=25.0, jitter=0.0)
    out = resample_signal_packet(packet, target_fs=None)
    dt = np.diff(out["t"])
    assert np.allclose(dt, dt[0])
    fs_est = 1.0 / dt[0]
    assert fs_est == pytest.approx(25.0, rel=1e-6, abs=1e-6)
