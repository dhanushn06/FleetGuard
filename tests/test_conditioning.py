import numpy as np
import pytest
from scipy import signal

from signal_processing.conditioning import (
    DEFAULT_HIGHHZ,
    DEFAULT_LOWHZ,
    MIN_CONDITION_DURATION_S,
    bandpass_filter,
    condition_signal,
    design_bandpass_filter,
    detrend_signal,
)


FS = 30.0
DURATION_S = 10.0
N_SAMPLES = int(FS * DURATION_S)


def make_time(fs: float = FS, n_samples: int = N_SAMPLES) -> np.ndarray:
    return np.arange(n_samples, dtype=np.float64) / fs


def dominant_frequency(x: np.ndarray, fs: float) -> float:
    """Estimate dominant frequency using FFT magnitude peak (positive freqs)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(x - np.mean(x)))
    # Ignore DC bin when searching for pulsatile component
    if mag.size > 1:
        idx = 1 + int(np.argmax(mag[1:]))
    else:
        idx = int(np.argmax(mag))
    return float(freqs[idx])


def test_detrend_signal_removes_linear_trend():
    rng = np.random.default_rng(0)
    t = make_time()
    linear_trend = 3.5 * t + 120.0
    sine = 2.0 * np.sin(2 * np.pi * 1.2 * t)
    noise = rng.normal(0.0, 0.05, size=t.size)
    x = linear_trend + sine + noise

    y = detrend_signal(x, method="linear")

    assert y.shape == x.shape
    assert np.all(np.isfinite(y))
    # Linear trend should be largely removed (small residual slope).
    slope = np.polyfit(t, y, 1)[0]
    assert abs(slope) < 0.05
    # DC offset from constant part of trend should be removed as well.
    assert abs(np.mean(y)) < 0.1


def test_detrend_signal_constant_removes_mean():
    rng = np.random.default_rng(1)
    t = make_time()
    x = 50.0 + 0.3 * np.sin(2 * np.pi * 1.0 * t) + rng.normal(0.0, 0.01, size=t.size)
    y = detrend_signal(x, method="constant")
    assert y.shape == x.shape
    assert abs(np.mean(y)) < 1e-10


def test_condition_signal_preserves_known_in_band_frequency():
    rng = np.random.default_rng(2)
    t = make_time()
    f0 = 1.2  # 72 BPM, inside [0.75, 3.0] Hz
    trend = 0.8 * t + 100.0
    in_band = 1.5 * np.sin(2 * np.pi * f0 * t)
    noise = rng.normal(0.0, 0.05, size=t.size)
    x = trend + in_band + noise

    y = condition_signal(x, fs=FS)

    assert y.shape == x.shape
    assert np.all(np.isfinite(y))

    f_est = dominant_frequency(y, FS)
    assert abs(f_est - f0) <= (FS / N_SAMPLES)  # within one FFT bin
    # Conditioned output should correlate strongly with a clean in-band reference.
    ref = signal.detrend(in_band, type="constant")
    corr = np.corrcoef(y, ref)[0, 1]
    assert corr > 0.98


def test_condition_signal_attenuates_out_of_band_frequency():
    rng = np.random.default_rng(3)
    t = make_time()
    low_out = 0.2   # below 0.75 Hz
    in_band = 1.3   # inside band
    high_out = 4.5  # above 3.0 Hz (and below Nyquist for 30 Hz)
    x = (
        0.8 * t
        + 2.0 * np.sin(2 * np.pi * low_out * t)
        + 1.0 * np.sin(2 * np.pi * in_band * t)
        + 1.5 * np.sin(2 * np.pi * high_out * t)
        + rng.normal(0.0, 0.02, size=t.size)
    )

    y = condition_signal(x, fs=FS)

    freqs = np.fft.rfftfreq(t.size, d=1.0 / FS)
    mag = np.abs(np.fft.rfft(y - np.mean(y)))

    def peak_near(freq_target: float, tol_bins: int = 2) -> float:
        idx = int(np.argmin(np.abs(freqs - freq_target)))
        lo = max(0, idx - tol_bins)
        hi = min(mag.size, idx + tol_bins + 1)
        return float(np.max(mag[lo:hi]))

    mag_in = peak_near(in_band)
    mag_low = peak_near(low_out)
    mag_high = peak_near(high_out)

    assert mag_in > 10.0 * mag_low
    assert mag_in > 10.0 * mag_high


def test_design_bandpass_filter_rejects_invalid_sampling_rates():
    with pytest.raises(ValueError):
        design_bandpass_filter(fs=0.0)
    with pytest.raises(ValueError):
        design_bandpass_filter(fs=-30.0)
    with pytest.raises(ValueError):
        design_bandpass_filter(fs=np.nan)


def test_design_bandpass_filter_rejects_invalid_cutoffs_and_nyquist():
    # low must be < high and > 0
    with pytest.raises(ValueError):
        design_bandpass_filter(fs=FS, low_hz=2.0, high_hz=1.0)
    with pytest.raises(ValueError):
        design_bandpass_filter(fs=FS, low_hz=0.0, high_hz=2.0)
    with pytest.raises(ValueError):
        design_bandpass_filter(fs=FS, low_hz=-0.5, high_hz=2.0)
    # high must be below Nyquist (fs/2 = 15 Hz here)
    with pytest.raises(ValueError):
        design_bandpass_filter(fs=FS, low_hz=0.75, high_hz=15.0)
    with pytest.raises(ValueError):
        design_bandpass_filter(fs=FS, low_hz=0.75, high_hz=20.0)


def test_condition_signal_rejects_nan_inf_inputs():
    t = make_time()
    x = np.sin(2 * np.pi * 1.2 * t)
    x_nan = x.copy()
    x_nan[10] = np.nan
    x_inf = x.copy()
    x_inf[20] = np.inf

    with pytest.raises(ValueError, match="NaN|infinite|non-finite"):
        condition_signal(x_nan, fs=FS)
    with pytest.raises(ValueError, match="NaN|infinite|non-finite"):
        condition_signal(x_inf, fs=FS)


def test_condition_signal_rejects_constant_signals():
    x_const = np.full(N_SAMPLES, 123.45, dtype=np.float64)
    with pytest.raises(ValueError, match="constant|degenerate"):
        condition_signal(x_const, fs=FS)

    # Near-constant / numerically degenerate
    x_near = np.full(N_SAMPLES, 10.0, dtype=np.float64)
    x_near[0] += 1e-15
    with pytest.raises(ValueError, match="constant|degenerate"):
        condition_signal(x_near, fs=FS)


def test_condition_signal_rejects_too_short_signals():
    t = make_time(fs=FS, n_samples=30)  # 1 second at 30 Hz, below 4 s minimum
    x = np.sin(2 * np.pi * 1.2 * t)
    with pytest.raises(ValueError, match="too short"):
        condition_signal(x, fs=FS)

    # Exactly below duration threshold
    too_few = int(np.ceil(FS * MIN_CONDITION_DURATION_S)) - 1
    t2 = make_time(fs=FS, n_samples=too_few)
    x2 = np.sin(2 * np.pi * 1.2 * t2)
    with pytest.raises(ValueError, match="too short"):
        condition_signal(x2, fs=FS)


def test_condition_signal_rejects_invalid_sampling_rate():
    t = make_time()
    x = np.sin(2 * np.pi * 1.2 * t) + 0.01 * t
    with pytest.raises(ValueError):
        condition_signal(x, fs=0.0)
    with pytest.raises(ValueError):
        condition_signal(x, fs=-10.0)


def test_condition_signal_rejects_invalid_cutoff_frequencies():
    t = make_time()
    x = np.sin(2 * np.pi * 1.2 * t) + 0.01 * t
    with pytest.raises(ValueError):
        condition_signal(x, fs=FS, low_hz=2.5, high_hz=2.0)
    with pytest.raises(ValueError):
        condition_signal(x, fs=FS, low_hz=DEFAULT_LOWHZ, high_hz=FS / 2.0)  # Nyquist boundary


def test_bandpass_filter_rejects_invalid_inputs_and_returns_shape():
    t = make_time()
    x = np.sin(2 * np.pi * 1.2 * t)
    sos = design_bandpass_filter(FS, DEFAULT_LOWHZ, DEFAULT_HIGHHZ, order=4)

    y = bandpass_filter(x, sos)
    assert y.shape == x.shape
    assert y.dtype == np.float64
    assert np.all(np.isfinite(y))

    with pytest.raises(ValueError):
        bandpass_filter(np.array([[1.0, 2.0], [3.0, 4.0]]), sos)  # 2D not supported
    with pytest.raises(ValueError):
        bandpass_filter(np.array([1.0, np.nan, 3.0, 4.0]), sos)
    with pytest.raises(ValueError):
        bandpass_filter(x, np.array([1.0, 2.0, 3.0]))  # invalid SOS shape


def test_detrend_signal_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        detrend_signal(np.array([[1.0, 2.0], [3.0, 4.0]]), method="linear")  # 2D
    with pytest.raises(ValueError):
        detrend_signal(np.array([1.0, np.nan, 2.0, 3.0]), method="linear")
    with pytest.raises(ValueError):
        detrend_signal(np.ones(100), method="unknown")
    with pytest.raises(ValueError):
        detrend_signal(np.full(100, 5.0), method="linear")  # constant/degenerate
