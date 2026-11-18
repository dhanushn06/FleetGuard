"""Tests for signal_processing.quality_metrics module."""

import numpy as np
import pytest

from signal_processing.quality_metrics import (
    calculate_snr,
    calculate_spectral_peak_prominence,
    calculate_spectral_concentration,
    calculate_method_agreement,
    calculate_roi_agreement,
    calculate_motion_summary,
    calculate_tracking_summary,
    calculate_brightness_summary,
)


# =============================================================================
# Fixtures and synthetic data generators
# =============================================================================

@pytest.fixture
def synthetic_psd():
    """Generate a synthetic PSD with a clear peak at 1.2 Hz (72 BPM)."""
    fs = 30.0
    n = 300  # 10 seconds
    t = np.arange(n) / fs
    # Signal: 1.2 Hz sine + harmonics + noise
    signal = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 2.4 * t)
    noise = 0.1 * np.random.randn(n)
    x = signal + noise

    # Welch PSD
    from scipy.signal import welch
    freqs, psd = welch(x, fs=fs, window="hann", nperseg=128, noverlap=64)
    return freqs, psd, 1.2  # freqs, psd, true HR frequency


@pytest.fixture
def flat_psd():
    """Generate a flat PSD (white noise)."""
    fs = 30.0
    n = 300
    x = np.random.randn(n)
    from scipy.signal import welch
    freqs, psd = welch(x, fs=fs, window="hann", nperseg=128, noverlap=64)
    return freqs, psd


@pytest.fixture
def zero_psd():
    """Generate a zero PSD (constant signal)."""
    fs = 30.0
    n = 300
    x = np.ones(n)
    from scipy.signal import welch
    freqs, psd = welch(x, fs=fs, window="hann", nperseg=128, noverlap=64)
    return freqs, psd


# =============================================================================
# Tests for calculate_snr
# =============================================================================

class TestCalculateSNR:
    def test_snr_valid_peak(self, synthetic_psd):
        freqs, psd, hr_freq = synthetic_psd
        # Use wider signal window to ensure peak is captured
        result = calculate_snr(freqs, psd, hr_freq, signal_window_hz=0.3)
        assert result["status"] == "ok"
        assert result["snr_db"] is not None
        assert np.isfinite(result["snr_db"])
        assert result["signal_power"] > 0
        assert result["noise_power"] > 0

    def test_snr_hr_outside_band(self, synthetic_psd):
        freqs, psd, _ = synthetic_psd
        # HR frequency outside default band (0.75, 3.0)
        result = calculate_snr(freqs, psd, 0.5)
        assert result["status"] == "invalid_input"
        assert result["snr_db"] is None

    def test_snr_zero_noise_power(self):
        # Create PSD with signal only in peak window, zero elsewhere
        freqs = np.linspace(0, 15, 300)
        psd = np.zeros_like(freqs)
        # Put power only at 1.2 Hz
        peak_idx = np.argmin(np.abs(freqs - 1.2))
        psd[peak_idx] = 1.0
        result = calculate_snr(freqs, psd, 1.2, signal_window_hz=0.05)
        assert result["status"] == "zero_noise_power"
        assert result["snr_db"] is None
        assert result["noise_power"] == 0.0

    def test_snr_zero_total_power(self, zero_psd):
        freqs, psd = zero_psd
        result = calculate_snr(freqs, psd, 1.2)
        assert result["status"] == "zero_total_power"
        assert result["snr_db"] is None

    def test_snr_invalid_inputs(self):
        # Mismatched shapes
        result = calculate_snr([1, 2, 3], [1, 2], 1.2)
        assert result["status"] == "invalid_input"

        # Non-finite HR frequency
        result = calculate_snr([1, 2, 3], [1, 2, 3], np.nan)
        assert result["status"] == "invalid_input"

        # Non-finite frequencies
        result = calculate_snr([1, np.nan, 3], [1, 2, 3], 1.2)
        assert result["status"] == "invalid_input"

    def test_snr_custom_band_and_window(self, synthetic_psd):
        freqs, psd, hr_freq = synthetic_psd
        result = calculate_snr(freqs, psd, hr_freq, signal_window_hz=0.2, hr_band=(0.5, 4.0))
        assert result["status"] == "ok"


# =============================================================================
# Tests for calculate_spectral_peak_prominence
# =============================================================================

class TestCalculateSpectralPeakProminence:
    def test_prominence_valid_peak(self, synthetic_psd):
        freqs, psd, hr_freq = synthetic_psd
        result = calculate_spectral_peak_prominence(freqs, psd, hr_freq)
        assert result["status"] == "ok"
        assert result["prominence"] is not None
        assert np.isfinite(result["prominence"])
        assert result["peak_psd"] > 0
        assert result["baseline_psd"] >= 0

    def test_prominence_flat_spectrum(self, flat_psd):
        freqs, psd = flat_psd
        # Pick a frequency in the middle
        hr_freq = freqs[len(freqs) // 2]
        result = calculate_spectral_peak_prominence(freqs, psd, hr_freq)
        # Flat spectrum -> prominence near zero (but noise causes small variations)
        assert result["status"] == "ok"
        assert result["prominence"] is not None
        # White noise PSD has random variations, so prominence is small but not zero
        assert abs(result["prominence"]) < 1.0  # reasonable bound for noise

    def test_prominence_insufficient_data(self):
        # Very narrow frequency array
        freqs = np.array([1.0, 1.1])
        psd = np.array([1.0, 0.5])
        result = calculate_spectral_peak_prominence(freqs, psd, 1.0, prominence_window_hz=0.01)
        assert result["status"] == "insufficient_data"

    def test_prominence_invalid_inputs(self):
        result = calculate_spectral_peak_prominence([1, 2], [1], 1.0)
        assert result["status"] == "invalid_input"

        result = calculate_spectral_peak_prominence([1, 2, 3], [1, 2, 3], np.nan)
        assert result["status"] == "invalid_input"


# =============================================================================
# Tests for calculate_spectral_concentration
# =============================================================================

class TestCalculateSpectralConcentration:
    def test_concentration_valid_peak(self, synthetic_psd):
        freqs, psd, hr_freq = synthetic_psd
        # Use wider concentration window to ensure peak is captured
        result = calculate_spectral_concentration(freqs, psd, hr_freq, concentration_window_hz=0.3)
        assert result["status"] == "ok"
        assert result["concentration"] is not None
        assert 0.0 <= result["concentration"] <= 1.0
        assert result["peak_power"] > 0
        assert result["total_band_power"] > 0

    def test_concentration_bounds(self, synthetic_psd):
        freqs, psd, hr_freq = synthetic_psd
        result = calculate_spectral_concentration(freqs, psd, hr_freq)
        assert 0.0 <= result["concentration"] <= 1.0

    def test_concentration_zero_total_power(self, zero_psd):
        freqs, psd = zero_psd
        result = calculate_spectral_concentration(freqs, psd, 1.2)
        assert result["status"] == "zero_total_power"
        assert result["concentration"] is None

    def test_concentration_hr_outside_band(self, synthetic_psd):
        freqs, psd, _ = synthetic_psd
        result = calculate_spectral_concentration(freqs, psd, 0.5)
        assert result["status"] == "invalid_input"

    def test_concentration_invalid_inputs(self):
        result = calculate_spectral_concentration([1, 2], [1], 1.0)
        assert result["status"] == "invalid_input"

        result = calculate_spectral_concentration([1, 2, 3], [1, 2, 3], np.nan)
        assert result["status"] == "invalid_input"


# =============================================================================
# Tests for calculate_method_agreement
# =============================================================================

class TestCalculateMethodAgreement:
    def test_agreement_three_valid(self):
        results = {
            "green": {"valid": True, "bpm": 72.0},
            "chrom": {"valid": True, "bpm": 73.5},
            "pos": {"valid": True, "bpm": 71.0},
        }
        result = calculate_method_agreement(results)
        assert result["status"] == "ok"
        assert result["num_valid"] == 3
        assert abs(result["mean_bpm"] - 72.166) < 0.1
        assert result["std_bpm"] is not None
        assert result["std_bpm"] > 0
        assert result["range_bpm"] == 2.5
        assert set(result["valid_methods"]) == {"green", "chrom", "pos"}

    def test_agreement_two_valid(self):
        results = {
            "green": {"valid": True, "bpm": 72.0},
            "chrom": {"valid": True, "bpm": 73.0},
            "pos": {"valid": False, "bpm": None},
        }
        result = calculate_method_agreement(results)
        assert result["status"] == "ok"
        assert result["num_valid"] == 2
        assert result["mean_bpm"] == 72.5
        assert result["std_bpm"] is not None

    def test_agreement_one_valid(self):
        results = {
            "green": {"valid": True, "bpm": 72.0},
            "chrom": {"valid": False, "bpm": None},
            "pos": {"valid": False, "bpm": None},
        }
        result = calculate_method_agreement(results)
        assert result["status"] == "insufficient_valid_methods"
        assert result["num_valid"] == 1
        assert result["mean_bpm"] == 72.0
        assert result["std_bpm"] is None
        assert result["range_bpm"] is None

    def test_agreement_zero_valid(self):
        results = {
            "green": {"valid": False, "bpm": None},
            "chrom": {"valid": False, "bpm": None},
            "pos": {"valid": False, "bpm": None},
        }
        result = calculate_method_agreement(results)
        assert result["status"] == "insufficient_valid_methods"
        assert result["num_valid"] == 0
        assert result["mean_bpm"] is None

    def test_agreement_invalid_bpm_values(self):
        results = {
            "green": {"valid": True, "bpm": np.nan},
            "chrom": {"valid": True, "bpm": np.inf},
            "pos": {"valid": True, "bpm": 72.0},
        }
        result = calculate_method_agreement(results)
        assert result["num_valid"] == 1  # only pos is valid
        assert result["status"] == "insufficient_valid_methods"

    def test_agreement_invalid_input(self):
        result = calculate_method_agreement("not a dict")
        assert result["status"] == "invalid_input"

        result = calculate_method_agreement({"green": "not a dict"})
        assert result["status"] == "invalid_input"


# =============================================================================
# Tests for calculate_roi_agreement
# =============================================================================

class TestCalculateROIAgreement:
    def test_roi_agreement_three_regions(self):
        results = {
            "forehead": {
                "green": {"valid": True, "bpm": 72.0},
                "chrom": {"valid": True, "bpm": 73.0},
                "pos": {"valid": True, "bpm": 71.0},
            },
            "left_cheek": {
                "green": {"valid": True, "bpm": 72.5},
                "chrom": {"valid": True, "bpm": 73.5},
                "pos": {"valid": True, "bpm": 71.5},
            },
            "right_cheek": {
                "green": {"valid": True, "bpm": 71.5},
                "chrom": {"valid": True, "bpm": 72.5},
                "pos": {"valid": True, "bpm": 70.5},
            },
        }
        result = calculate_roi_agreement(results)
        assert result["status"] == "ok"
        assert result["num_valid_regions"] == 3
        assert result["mean_bpm"] is not None
        assert result["std_bpm"] is not None
        assert result["range_bpm"] is not None
        assert len(result["region_means"]) == 3

    def test_roi_agreement_two_regions(self):
        results = {
            "forehead": {
                "green": {"valid": True, "bpm": 72.0},
                "chrom": {"valid": True, "bpm": 73.0},
                "pos": {"valid": True, "bpm": 71.0},
            },
            "left_cheek": {
                "green": {"valid": True, "bpm": 72.5},
                "chrom": {"valid": True, "bpm": 73.5},
                "pos": {"valid": True, "bpm": 71.5},
            },
            "right_cheek": {
                "green": {"valid": False, "bpm": None},
                "chrom": {"valid": False, "bpm": None},
                "pos": {"valid": False, "bpm": None},
            },
        }
        result = calculate_roi_agreement(results)
        assert result["status"] == "ok"
        assert result["num_valid_regions"] == 2

    def test_roi_agreement_one_region(self):
        results = {
            "forehead": {
                "green": {"valid": True, "bpm": 72.0},
                "chrom": {"valid": True, "bpm": 73.0},
                "pos": {"valid": True, "bpm": 71.0},
            },
            "left_cheek": {
                "green": {"valid": False, "bpm": None},
                "chrom": {"valid": False, "bpm": None},
                "pos": {"valid": False, "bpm": None},
            },
        }
        result = calculate_roi_agreement(results)
        assert result["status"] == "insufficient_valid_regions"
        assert result["num_valid_regions"] == 1
        assert result["mean_bpm"] is not None
        assert result["std_bpm"] is None

    def test_roi_agreement_zero_regions(self):
        results = {
            "forehead": {"green": {"valid": False}, "chrom": {"valid": False}, "pos": {"valid": False}},
        }
        result = calculate_roi_agreement(results)
        assert result["status"] == "insufficient_valid_regions"
        assert result["num_valid_regions"] == 0

    def test_roi_agreement_invalid_input(self):
        result = calculate_roi_agreement("not a dict")
        assert result["status"] == "invalid_input"

        result = calculate_roi_agreement({"forehead": "not a dict"})
        assert result["status"] == "invalid_input"


# =============================================================================
# Tests for calculate_motion_summary
# =============================================================================

class TestCalculateMotionSummary:
    def test_motion_valid_data(self):
        motion = np.array([0.1, 0.2, 0.15, 0.3, 0.25])
        result = calculate_motion_summary(motion)
        assert result["status"] == "ok"
        assert result["num_valid"] == 5
        assert result["num_total"] == 5
        assert result["mean"] == pytest.approx(0.2)
        assert result["median"] == pytest.approx(0.2)
        assert result["max"] == pytest.approx(0.3)

    def test_motion_with_nan_inf(self):
        motion = np.array([0.1, np.nan, 0.2, np.inf, 0.3, -np.inf])
        result = calculate_motion_summary(motion)
        assert result["status"] == "ok"
        assert result["num_valid"] == 3
        assert result["num_total"] == 6
        assert result["mean"] == pytest.approx(0.2)

    def test_motion_all_invalid(self):
        motion = np.array([np.nan, np.inf, -np.inf])
        result = calculate_motion_summary(motion)
        assert result["status"] == "no_valid_data"
        assert result["num_valid"] == 0
        assert result["mean"] is None

    def test_motion_none_input(self):
        result = calculate_motion_summary(None)
        assert result["status"] == "no_valid_data"
        assert result["num_total"] == 0

    def test_motion_invalid_shape(self):
        motion = np.array([[0.1, 0.2], [0.3, 0.4]])  # 2D
        result = calculate_motion_summary(motion)
        assert result["status"] == "invalid_input"

    def test_motion_empty_array(self):
        motion = np.array([])
        result = calculate_motion_summary(motion)
        assert result["status"] == "no_valid_data"
        assert result["num_total"] == 0


# =============================================================================
# Tests for calculate_tracking_summary
# =============================================================================

class TestCalculateTrackingSummary:
    def test_tracking_all_true(self):
        track_ok = np.array([True, True, True, True, True])
        result = calculate_tracking_summary(track_ok)
        assert result["status"] == "ok"
        assert result["tracking_fraction"] == 1.0
        assert result["total_samples"] == 5
        assert result["tracked_samples"] == 5
        assert result["lost_samples"] == 0

    def test_tracking_mixed(self):
        track_ok = np.array([True, False, True, True, False])
        result = calculate_tracking_summary(track_ok)
        assert result["status"] == "ok"
        assert result["tracking_fraction"] == 0.6
        assert result["total_samples"] == 5
        assert result["tracked_samples"] == 3
        assert result["lost_samples"] == 2

    def test_tracking_integer_input(self):
        track_ok = np.array([1, 0, 1, 1, 0])
        result = calculate_tracking_summary(track_ok)
        assert result["status"] == "ok"
        assert result["tracking_fraction"] == 0.6

    def test_tracking_none_input(self):
        result = calculate_tracking_summary(None)
        assert result["status"] == "invalid_input"
        assert result["total_samples"] == 0

    def test_tracking_invalid_shape(self):
        track_ok = np.array([[True, False], [True, True]])
        result = calculate_tracking_summary(track_ok)
        assert result["status"] == "invalid_input"

    def test_tracking_empty_array(self):
        track_ok = np.array([], dtype=bool)
        result = calculate_tracking_summary(track_ok)
        assert result["status"] == "ok"
        assert result["total_samples"] == 0
        assert result["tracking_fraction"] == 0.0


# =============================================================================
# Tests for calculate_brightness_summary
# =============================================================================

class TestCalculateBrightnessSummary:
    def test_brightness_valid_data(self):
        brightness = np.array([100.0, 120.0, 110.0, 130.0, 115.0])
        result = calculate_brightness_summary(brightness)
        assert result["status"] == "ok"
        assert result["num_valid"] == 5
        assert result["num_total"] == 5
        assert result["mean"] == pytest.approx(115.0)
        assert result["median"] == pytest.approx(115.0)
        assert result["min"] == pytest.approx(100.0)
        assert result["max"] == pytest.approx(130.0)

    def test_brightness_with_nan_inf(self):
        brightness = np.array([100.0, np.nan, 120.0, np.inf, 110.0, -np.inf])
        result = calculate_brightness_summary(brightness)
        assert result["status"] == "ok"
        assert result["num_valid"] == 3
        assert result["num_total"] == 6
        assert result["mean"] == pytest.approx(110.0)
        assert result["min"] == pytest.approx(100.0)
        assert result["max"] == pytest.approx(120.0)

    def test_brightness_all_invalid(self):
        brightness = np.array([np.nan, np.inf, -np.inf])
        result = calculate_brightness_summary(brightness)
        assert result["status"] == "no_valid_data"
        assert result["num_valid"] == 0
        assert result["mean"] is None

    def test_brightness_none_input(self):
        result = calculate_brightness_summary(None)
        assert result["status"] == "no_valid_data"
        assert result["num_total"] == 0

    def test_brightness_invalid_shape(self):
        brightness = np.array([[100, 120], [110, 130]])
        result = calculate_brightness_summary(brightness)
        assert result["status"] == "invalid_input"

    def test_brightness_empty_array(self):
        brightness = np.array([])
        result = calculate_brightness_summary(brightness)
        assert result["status"] == "no_valid_data"
        assert result["num_total"] == 0


# =============================================================================
# Integration-style tests
# =============================================================================

class TestQualityMetricsIntegration:
    def test_all_metrics_on_synthetic_signal(self, synthetic_psd):
        """Run all spectral metrics on a synthetic signal with known HR."""
        freqs, psd, hr_freq = synthetic_psd

        # Use wider windows to ensure peak is captured
        snr_result = calculate_snr(freqs, psd, hr_freq, signal_window_hz=0.3)
        prom_result = calculate_spectral_peak_prominence(freqs, psd, hr_freq)
        conc_result = calculate_spectral_concentration(freqs, psd, hr_freq, concentration_window_hz=0.3)

        assert snr_result["status"] == "ok"
        assert prom_result["status"] == "ok"
        assert conc_result["status"] == "ok"

        # All should produce finite, reasonable values
        assert np.isfinite(snr_result["snr_db"])
        assert np.isfinite(prom_result["prominence"])
        assert 0.0 <= conc_result["concentration"] <= 1.0

    def test_method_and_roi_agreement_consistency(self):
        """Test that method and ROI agreement work together."""
        # Simulate multiregion output structure
        region_results = {
            "forehead": {
                "green": {"valid": True, "bpm": 72.0},
                "chrom": {"valid": True, "bpm": 73.0},
                "pos": {"valid": True, "bpm": 71.0},
            },
            "left_cheek": {
                "green": {"valid": True, "bpm": 72.5},
                "chrom": {"valid": True, "bpm": 73.5},
                "pos": {"valid": True, "bpm": 71.5},
            },
        }

        # Method agreement per region
        for region, methods in region_results.items():
            method_result = calculate_method_agreement(methods)
            assert method_result["status"] == "ok"
            assert method_result["num_valid"] == 3

        # ROI agreement across regions
        roi_result = calculate_roi_agreement(region_results)
        assert roi_result["status"] == "ok"
        assert roi_result["num_valid_regions"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])