import numpy as np
import pytest

from signal_processing.green import estimate_green_hr


FS = 30.0
DURATION_S = 12.0  # >= default min_duration_s (10 s)


def make_time(duration_s: float = DURATION_S, fs: float = FS) -> np.ndarray:
    n = int(np.round(duration_s * fs))
    return np.arange(n, dtype=np.float64) / fs


def synth_green(
    bpm: float,
    fs: float = FS,
    duration_s: float = DURATION_S,
    noise_std: float = 0.0,
    trend: float = 0.02,
    seed: int = 0,
    out_of_band: bool = False,
) -> np.ndarray:
    """Create a synthetic 1D green-channel signal with known pulse rate.

    Model (simplified): DC + slow linear trend + pulsatile sinusoid (+ optional
    out-of-band components and Gaussian noise). This is sufficient to validate
    conditioning + Welch peak-picking behavior on deterministic inputs.
    """
    rng = np.random.default_rng(seed)
    t = make_time(duration_s=duration_s, fs=fs)
    f0 = bpm / 60.0
    dc = 120.0
    pulse = 2.0 * np.sin(2 * np.pi * f0 * t + 0.3)
    x = dc + trend * t + pulse

    if out_of_band:
        # Strong low-frequency drift-like component and high-frequency component
        # (still below Nyquist for typical fs) to stress band-limited selection.
        x += 3.0 * np.sin(2 * np.pi * 0.15 * t)
        x += 1.5 * np.sin(2 * np.pi * 4.5 * t)

    if noise_std > 0:
        x += rng.normal(0.0, noise_std, size=t.size)

    return x


def assert_valid_green_result(result: dict, expected_bpm: float, tol_bpm: float, n_samples: int):
    assert result["method"] == "green"
    assert result["valid"] is True
    assert isinstance(result["bpm"], float)
    assert isinstance(result["frequency_hz"], float)
    assert abs(result["bpm"] - expected_bpm) <= tol_bpm
    assert abs(result["frequency_hz"] * 60.0 - result["bpm"]) < 1e-9

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


@pytest.mark.parametrize("expected_bpm", [60.0, 72.0, 90.0])
def test_estimate_green_hr_known_heart_rates(expected_bpm):
    x = synth_green(bpm=expected_bpm, seed=42, noise_std=0.02, out_of_band=False)
    result = estimate_green_hr(x, fs=FS)

    # Tolerance accounts for Welch bin resolution on finite windows (not overly strict).
    assert_valid_green_result(result, expected_bpm=expected_bpm, tol_bpm=6.0, n_samples=x.size)
    assert 0.75 <= result["frequency_hz"] <= 3.0


def test_estimate_green_hr_with_noise():
    expected_bpm = 72.0
    x = synth_green(bpm=expected_bpm, seed=7, noise_std=0.4, out_of_band=False)
    result = estimate_green_hr(x, fs=FS)

    assert_valid_green_result(result, expected_bpm=expected_bpm, tol_bpm=7.0, n_samples=x.size)
    assert result["valid"] is True


def test_estimate_green_hr_with_strong_out_of_band_components():
    expected_bpm = 78.0
    x = synth_green(bpm=expected_bpm, seed=11, noise_std=0.05, out_of_band=True)
    result = estimate_green_hr(x, fs=FS)

    assert_valid_green_result(result, expected_bpm=expected_bpm, tol_bpm=7.0, n_samples=x.size)
    # Peak must be selected within configured HR band despite strong out-of-band energy.
    assert 0.75 <= result["frequency_hz"] <= 3.0


def test_estimate_green_hr_rejects_insufficient_duration():
    # 5 seconds < default min_duration_s (10 s)
    x = synth_green(bpm=72.0, duration_s=5.0, seed=1)
    result = estimate_green_hr(x, fs=FS)

    assert result["method"] == "green"
    assert result["valid"] is False
    assert result["bpm"] is None
    assert result["frequency_hz"] is None
    assert isinstance(result["reason"], str) and "too short" in result["reason"].lower()


def test_estimate_green_hr_rejects_invalid_sampling_rate():
    x = synth_green(bpm=72.0, seed=2)
    result_zero = estimate_green_hr(x, fs=0.0)
    result_neg = estimate_green_hr(x, fs=-30.0)

    for result in (result_zero, result_neg):
        assert result["valid"] is False
        assert result["bpm"] is None
        assert result["frequency_hz"] is None
        assert "fs" in result["reason"].lower() or "sampling" in result["reason"].lower()


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_estimate_green_hr_rejects_nan_inf_input(bad_value):
    x = synth_green(bpm=72.0, seed=3)
    x[25] = bad_value
    result = estimate_green_hr(x, fs=FS)

    assert result["valid"] is False
    assert result["bpm"] is None
    assert result["frequency_hz"] is None
    assert "nan" in result["reason"].lower() or "infinite" in result["reason"].lower() or "non-finite" in result["reason"].lower()


def test_estimate_green_hr_rejects_constant_or_degenerate_signal():
    x_const = np.full(int(FS * DURATION_S), 120.0, dtype=np.float64)
    result = estimate_green_hr(x_const, fs=FS)

    assert result["valid"] is False
    assert result["bpm"] is None
    assert result["frequency_hz"] is None
    assert "constant" in result["reason"].lower() or "degenerate" in result["reason"].lower()


def test_estimate_green_hr_rejects_invalid_frequency_band():
    x = synth_green(bpm=72.0, seed=4)

    # low >= high
    result_bad_order = estimate_green_hr(x, fs=FS, low_hz=2.0, high_hz=1.0)
    assert result_bad_order["valid"] is False
    assert result_bad_order["bpm"] is None
    assert result_bad_order["frequency_hz"] is None

    # high above Nyquist (fs/2 = 15 Hz here)
    result_bad_nyquist = estimate_green_hr(x, fs=FS, low_hz=0.75, high_hz=15.0)
    assert result_bad_nyquist["valid"] is False
    assert result_bad_nyquist["bpm"] is None
    assert result_bad_nyquist["frequency_hz"] is None
    assert "nyquist" in result_bad_nyquist["reason"].lower() or "high_hz" in result_bad_nyquist["reason"].lower()


def test_estimate_green_hr_output_structure_valid_case():
    x = synth_green(bpm=66.0, seed=5, noise_std=0.1)
    result = estimate_green_hr(x, fs=FS)

    assert result["valid"] is True
    required_keys = {"method", "bpm", "valid", "frequency_hz", "frequencies", "psd", "conditioned_signal"}
    assert required_keys.issubset(result.keys())
    assert result["method"] == "green"
    assert isinstance(result["bpm"], float)
    assert result["frequency_hz"] == pytest.approx(result["bpm"] / 60.0)
    assert result["frequencies"].shape == result["psd"].shape
    assert result["conditioned_signal"].shape == x.shape


def test_estimate_green_hr_dominant_frequency_within_configured_band():
    # True pulse at 66 BPM (1.1 Hz), narrow band around it; result must stay in-band.
    x = synth_green(bpm=66.0, seed=6, noise_std=0.05)
    low_hz, high_hz = 0.9, 1.4
    result = estimate_green_hr(x, fs=FS, low_hz=low_hz, high_hz=high_hz)

    assert result["valid"] is True
    assert low_hz <= result["frequency_hz"] <= high_hz
    assert low_hz * 60.0 <= result["bpm"] <= high_hz * 60.0

    # Also verify the PSD peak index corresponds to an in-band frequency.
    freqs = result["frequencies"]
    psd = result["psd"]
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    peak_idx = int(np.argmax(psd[mask]))
    peak_freq = freqs[np.where(mask)[0][peak_idx]]
    assert peak_freq == pytest.approx(result["frequency_hz"])
