import numpy as np
import pytest

from signal_processing.chrom import estimate_chrom_hr


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

    Simplified model:
        channel = DC * (1 + pulse_gain * sin(2*pi*f*t)) + trend + noise + artifacts
    Green carries a stronger pulse component than red/blue (typical for rPPG),
    with small phase/amplitude differences to exercise CHROM projections.
    """
    rng = np.random.default_rng(seed)
    t = make_time(duration_s=duration_s, fs=fs)
    f0 = bpm / 60.0
    pulse = np.sin(2 * np.pi * f0 * t + 0.2)

    # Baseline intensities and relative pulse strengths (G strongest).
    r = 140.0 * (1.0 + 0.004 * pulse) + 0.01 * t
    g = 120.0 * (1.0 + 0.010 * np.sin(2 * np.pi * f0 * t + 0.1)) + 0.015 * t
    b = 100.0 * (1.0 + 0.003 * np.sin(2 * np.pi * f0 * t + 0.25)) + 0.008 * t

    if out_of_band:
        # Illumination/motion-like low-frequency drift and high-frequency component.
        drift = 2.0 * np.sin(2 * np.pi * 0.15 * t)
        hf = 0.8 * np.sin(2 * np.pi * 4.5 * t)
        r += drift + 0.5 * hf
        g += 1.2 * drift + 0.3 * hf
        b += 0.9 * drift + 0.4 * hf

    if noise_std > 0:
        r += rng.normal(0.0, noise_std, size=t.size)
        g += rng.normal(0.0, noise_std, size=t.size)
        b += rng.normal(0.0, noise_std, size=t.size)

    return np.stack([r, g, b], axis=1)


def assert_valid_chrom_result(result: dict, expected_bpm: float, tol_bpm: float, n_samples: int):
    assert result["method"] == "chrom"
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
def test_estimate_chrom_known_heart_rates(expected_bpm):
    rgb = synth_rgb(bpm=expected_bpm, seed=10, noise_std=0.05)
    result = estimate_chrom_hr(rgb, fs=FS)

    # Tolerance reflects finite-window Welch bin resolution.
    assert_valid_chrom_result(result, expected_bpm=expected_bpm, tol_bpm=6.0, n_samples=rgb.shape[0])


def test_estimate_chrom_with_noise():
    expected_bpm = 72.0
    rgb = synth_rgb(bpm=expected_bpm, seed=11, noise_std=0.3)
    result = estimate_chrom_hr(rgb, fs=FS)
    assert_valid_chrom_result(result, expected_bpm=expected_bpm, tol_bpm=7.0, n_samples=rgb.shape[0])


def test_estimate_chrom_with_strong_out_of_band_components():
    expected_bpm = 78.0
    rgb = synth_rgb(bpm=expected_bpm, seed=12, noise_std=0.05, out_of_band=True)
    result = estimate_chrom_hr(rgb, fs=FS)
    assert_valid_chrom_result(result, expected_bpm=expected_bpm, tol_bpm=7.0, n_samples=rgb.shape[0])


def test_estimate_chrom_rejects_invalid_shape_and_nonfinite():
    rgb = synth_rgb(bpm=72.0, seed=1)

    # wrong shape
    res_shape = estimate_chrom_hr(rgb[:, :2], fs=FS)
    assert res_shape["valid"] is False and res_shape["bpm"] is None
    assert "shape" in res_shape["reason"].lower()

    # NaN/Inf
    rgb_nan = rgb.copy()
    rgb_nan[10, 1] = np.nan
    res_nan = estimate_chrom_hr(rgb_nan, fs=FS)
    assert res_nan["valid"] is False and res_nan["bpm"] is None
    assert "nan" in res_nan["reason"].lower() or "non-finite" in res_nan["reason"].lower() or "infinite" in res_nan["reason"].lower()

    rgb_inf = rgb.copy()
    rgb_inf[20, 0] = np.inf
    res_inf = estimate_chrom_hr(rgb_inf, fs=FS)
    assert res_inf["valid"] is False and res_inf["bpm"] is None


def test_estimate_chrom_rejects_invalid_sampling_rate_and_band():
    rgb = synth_rgb(bpm=72.0, seed=2)

    res_fs = estimate_chrom_hr(rgb, fs=0.0)
    assert res_fs["valid"] is False and res_fs["bpm"] is None

    res_band = estimate_chrom_hr(rgb, fs=FS, low_hz=2.0, high_hz=1.0)
    assert res_band["valid"] is False and res_band["bpm"] is None

    res_nyq = estimate_chrom_hr(rgb, fs=FS, low_hz=0.75, high_hz=15.0)
    assert res_nyq["valid"] is False and res_nyq["bpm"] is None
    assert "nyquist" in res_nyq["reason"].lower() or "high_hz" in res_nyq["reason"].lower()


def test_estimate_chrom_rejects_insufficient_duration():
    rgb = synth_rgb(bpm=72.0, duration_s=5.0, seed=3)  # < 10 s default
    res = estimate_chrom_hr(rgb, fs=FS)
    assert res["valid"] is False
    assert res["bpm"] is None
    assert res["frequency_hz"] is None
    assert "too short" in res["reason"].lower()


def test_estimate_chrom_rejects_degenerate_or_invalid_mean_channels():
    n = int(FS * DURATION_S)

    # constant channel (degenerate)
    rgb_const = synth_rgb(bpm=72.0, seed=4)
    rgb_const[:, 1] = 120.0
    res_const = estimate_chrom_hr(rgb_const, fs=FS)
    assert res_const["valid"] is False and res_const["bpm"] is None
    assert "degenerate" in res_const["reason"].lower() or "constant" in res_const["reason"].lower()

    # near-zero mean channel (invalid for normalization)
    rgb_zero_mean = np.ones((n, 3), dtype=np.float64) * 100.0
    rgb_zero_mean[:, 2] = 0.0
    # add tiny variance? still mean zero
    res_mean = estimate_chrom_hr(rgb_zero_mean, fs=FS)
    assert res_mean["valid"] is False and res_mean["bpm"] is None
    assert "mean" in res_mean["reason"].lower() or "normalization" in res_mean["reason"].lower() or "degenerate" in res_mean["reason"].lower()


def test_estimate_chrom_output_structure_valid_case():
    rgb = synth_rgb(bpm=66.0, seed=5, noise_std=0.1)
    res = estimate_chrom_hr(rgb, fs=FS)

    assert res["valid"] is True
    required = {"method", "bpm", "valid", "frequency_hz", "frequencies", "psd", "conditioned_signal", "alpha"}
    assert required.issubset(res.keys())
    assert res["method"] == "chrom"
    assert res["conditioned_signal"].shape == (rgb.shape[0],)
    assert res["frequencies"].shape == res["psd"].shape
    assert res["frequency_hz"] == pytest.approx(res["bpm"] / 60.0)


def test_estimate_chrom_dominant_frequency_within_configured_band():
    rgb = synth_rgb(bpm=66.0, seed=6, noise_std=0.05)
    low_hz, high_hz = 0.9, 1.4
    res = estimate_chrom_hr(rgb, fs=FS, low_hz=low_hz, high_hz=high_hz)

    assert res["valid"] is True
    assert low_hz <= res["frequency_hz"] <= high_hz
    assert low_hz * 60.0 <= res["bpm"] <= high_hz * 60.0
