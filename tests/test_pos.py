import numpy as np
import pytest

from signal_processing.pos import estimate_pos_hr


FS = 30.0
DURATION_S = 12.0  # >= default min_duration_s (10 s)


def make_time(duration_s: float = DURATION_S, fs: float = FS) -> np.ndarray:
    n = int(np.round(duration_s * fs))
    return np.arange(n, dtype=np.float64) / fs


def synth_rgb(
    bpm: float,
    fs: float = FS,
    duration_s: float = DURATION_S,
    noise_std: float = 0.0,
    seed: int = 0,
    out_of_band: bool = False,
) -> np.ndarray:
    """Create a synthetic (N,3) RGB signal with a known pulsatile frequency.

    Simplified model for exercising POS:
      - temporal mean-normalizable channels
      - pulsatile modulation with different channel gains/phases
      - optional slow trends, out-of-band components, and Gaussian noise
    """
    rng = np.random.default_rng(seed)
    t = make_time(duration_s=duration_s, fs=fs)
    f0 = bpm / 60.0

    # Pulsatile components (green strongest, small phase differences).
    pr = np.sin(2 * np.pi * f0 * t + 0.20)
    pg = np.sin(2 * np.pi * f0 * t + 0.05)
    pb = np.sin(2 * np.pi * f0 * t + 0.35)

    r = 140.0 * (1.0 + 0.004 * pr) + 0.010 * t
    g = 120.0 * (1.0 + 0.012 * pg) + 0.015 * t
    b = 100.0 * (1.0 + 0.003 * pb) + 0.008 * t

    if out_of_band:
        # Low-frequency drift and high-frequency component (below Nyquist).
        drift = 2.5 * np.sin(2 * np.pi * 0.15 * t)
        hf = 0.9 * np.sin(2 * np.pi * 4.5 * t)
        r += 0.8 * drift + 0.4 * hf
        g += 1.2 * drift + 0.3 * hf
        b += 1.0 * drift + 0.5 * hf

    if noise_std > 0:
        r += rng.normal(0.0, noise_std, size=t.size)
        g += rng.normal(0.0, noise_std, size=t.size)
        b += rng.normal(0.0, noise_std, size=t.size)

    return np.stack([r, g, b], axis=1)


def assert_valid_pos_result(result: dict, expected_bpm: float, tol_bpm: float, n_samples: int):
    assert result["method"] == "pos"
    assert result["valid"] is True
    assert isinstance(result["bpm"], float)
    assert isinstance(result["frequency_hz"], float)
    assert abs(result["bpm"] - expected_bpm) <= tol_bpm
    assert abs(result["frequency_hz"] * 60.0 - result["bpm"]) < 1e-9
    assert 0.75 <= result["frequency_hz"] <= 3.0

    freqs = result["frequencies"]
    psd = result["psd"]
    cond = result["conditioned_signal"]
    assert isinstance(freqs, np.ndarray) and freqs.ndim == 1
    assert isinstance(psd, np.ndarray) and psd.ndim == 1
    assert freqs.shape == psd.shape
    assert isinstance(cond, np.ndarray) and cond.shape == (n_samples,)
    assert np.all(np.isfinite(cond))
    assert np.all(np.isfinite(freqs))
    assert np.all(np.isfinite(psd))
    assert "alpha" in result
    assert np.isfinite(float(result["alpha"]))


@pytest.mark.parametrize("expected_bpm", [60.0, 72.0, 90.0])
def test_estimate_pos_hr_known_heart_rates(expected_bpm):
    rgb = synth_rgb(bpm=expected_bpm, seed=21, noise_std=0.05)
    result = estimate_pos_hr(rgb, fs=FS)
    assert_valid_pos_result(result, expected_bpm=expected_bpm, tol_bpm=6.0, n_samples=rgb.shape[0])


def test_estimate_pos_hr_noisy_signal():
    expected_bpm = 72.0
    rgb = synth_rgb(bpm=expected_bpm, seed=22, noise_std=0.35)
    result = estimate_pos_hr(rgb, fs=FS)
    assert_valid_pos_result(result, expected_bpm=expected_bpm, tol_bpm=7.0, n_samples=rgb.shape[0])


def test_estimate_pos_hr_out_of_band_components():
    expected_bpm = 78.0
    rgb = synth_rgb(bpm=expected_bpm, seed=23, noise_std=0.05, out_of_band=True)
    result = estimate_pos_hr(rgb, fs=FS)
    assert_valid_pos_result(result, expected_bpm=expected_bpm, tol_bpm=7.0, n_samples=rgb.shape[0])
    assert 0.75 <= result["frequency_hz"] <= 3.0


def test_estimate_pos_hr_rejects_invalid_input_shape():
    rgb = synth_rgb(bpm=72.0, seed=1)
    result = estimate_pos_hr(rgb[:, :2], fs=FS)
    assert result["method"] == "pos"
    assert result["valid"] is False
    assert result["bpm"] is None
    assert "shape" in result["reason"].lower()


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_estimate_pos_hr_rejects_nan_and_infinity(bad_value):
    rgb = synth_rgb(bpm=72.0, seed=2)
    rgb[30, 1] = bad_value
    result = estimate_pos_hr(rgb, fs=FS)
    assert result["valid"] is False
    assert result["bpm"] is None
    assert "nan" in result["reason"].lower() or "infinite" in result["reason"].lower() or "non-finite" in result["reason"].lower()


@pytest.mark.parametrize("bad_fs", [0.0, -30.0, np.nan, np.inf])
def test_estimate_pos_hr_rejects_invalid_sampling_frequency(bad_fs):
    rgb = synth_rgb(bpm=72.0, seed=3)
    result = estimate_pos_hr(rgb, fs=bad_fs)
    assert result["valid"] is False
    assert result["bpm"] is None
    assert "fs" in result["reason"].lower() or "sampling" in result["reason"].lower() or "finite" in result["reason"].lower()


def test_estimate_pos_hr_rejects_invalid_frequency_band_and_nyquist():
    rgb = synth_rgb(bpm=72.0, seed=4)

    # invalid order
    result_order = estimate_pos_hr(rgb, fs=FS, low_hz=2.0, high_hz=1.0)
    assert result_order["valid"] is False and result_order["bpm"] is None

    # Nyquist violation / boundary
    result_nyq = estimate_pos_hr(rgb, fs=FS, low_hz=0.75, high_hz=15.0)
    assert result_nyq["valid"] is False and result_nyq["bpm"] is None
    assert "nyquist" in result_nyq["reason"].lower() or "high_hz" in result_nyq["reason"].lower()

    result_nyq_boundary = estimate_pos_hr(rgb, fs=FS, low_hz=0.75, high_hz=FS / 2.0)
    assert result_nyq_boundary["valid"] is False and result_nyq_boundary["bpm"] is None


def test_estimate_pos_hr_rejects_insufficient_signal_duration():
    rgb = synth_rgb(bpm=72.0, duration_s=5.0, seed=5)  # < default 10 s
    result = estimate_pos_hr(rgb, fs=FS)
    assert result["valid"] is False
    assert result["bpm"] is None
    assert "too short" in result["reason"].lower()


def test_estimate_pos_hr_rejects_degenerate_constant_signals():
    n = int(FS * DURATION_S)
    rgb_const = np.full((n, 3), 120.0, dtype=np.float64)
    result = estimate_pos_hr(rgb_const, fs=FS)
    assert result["valid"] is False
    assert result["bpm"] is None
    assert "degenerate" in result["reason"].lower() or "constant" in result["reason"].lower()


def test_estimate_pos_hr_rejects_zero_or_near_zero_channel_means():
    n = int(FS * DURATION_S)
    t = make_time()
    pulse = np.sin(2 * np.pi * 1.2 * t)
    rgb = np.stack([
        100.0 + 0.5 * pulse,
        120.0 + 0.6 * pulse,
        0.0 * t,  # zero mean channel
    ], axis=1)

    result = estimate_pos_hr(rgb, fs=FS)
    assert result["valid"] is False
    assert result["bpm"] is None
    assert "mean" in result["reason"].lower() or "normalization" in result["reason"].lower() or "degenerate" in result["reason"].lower()


def test_estimate_pos_hr_rejects_zero_or_near_zero_projection_std():
    t = make_time()
    f0 = 1.2
    pulse = np.sin(2 * np.pi * f0 * t)

    # Construct channels so normalized G and B are identical -> S1 = gn - bn == 0.
    r = 140.0 * (1.0 + 0.006 * pulse)
    g = 120.0 * (1.0 + 0.010 * pulse)
    b = 100.0 * (1.0 + 0.010 * pulse)  # same relative modulation as G
    rgb = np.stack([r, g, b], axis=1)

    result = estimate_pos_hr(rgb, fs=FS)
    assert result["valid"] is False
    assert result["bpm"] is None
    assert "projection" in result["reason"].lower() or "degenerate" in result["reason"].lower() or "std" in result["reason"].lower()


def test_estimate_pos_hr_output_structure_valid_case():
    rgb = synth_rgb(bpm=66.0, seed=6, noise_std=0.1)
    result = estimate_pos_hr(rgb, fs=FS)

    assert result["valid"] is True
    required = {"method", "valid", "bpm", "frequency_hz", "frequencies", "psd", "conditioned_signal", "alpha"}
    assert required.issubset(result.keys())
    assert result["method"] == "pos"
    assert isinstance(result["bpm"], float)
    assert result["frequency_hz"] == pytest.approx(result["bpm"] / 60.0)
    assert result["frequencies"].shape == result["psd"].shape
    assert result["conditioned_signal"].shape == (rgb.shape[0],)
    assert np.isfinite(float(result["alpha"]))


def test_estimate_pos_hr_peak_within_configured_band():
    rgb = synth_rgb(bpm=66.0, seed=7, noise_std=0.05)
    low_hz, high_hz = 0.9, 1.4
    result = estimate_pos_hr(rgb, fs=FS, low_hz=low_hz, high_hz=high_hz)

    assert result["valid"] is True
    assert low_hz <= result["frequency_hz"] <= high_hz
    assert low_hz * 60.0 <= result["bpm"] <= high_hz * 60.0
