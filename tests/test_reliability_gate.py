"""Tests for signal_processing.reliability_gate module."""

import numpy as np
import pytest

from signal_processing.reliability_gate import (
    ReliabilityThresholds,
    evaluate_reliability,
    _extract_valid_method_bpms,
    _extract_valid_roi_bpms,
    _check_method_agreement,
    _check_roi_agreement,
    _check_snr,
    _check_spectral_concentration,
    _check_tracking,
    _check_motion,
    _check_brightness,
    _check_duration,
    _determine_confidence,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def default_thresholds():
    return ReliabilityThresholds()


@pytest.fixture
def valid_method_results():
    """Three valid methods with good agreement."""
    return {
        "green": {"valid": True, "bpm": 72.0},
        "chrom": {"valid": True, "bpm": 73.0},
        "pos": {"valid": True, "bpm": 71.5},
    }


@pytest.fixture
def valid_region_results():
    """Three valid regions with good agreement."""
    return {
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


@pytest.fixture
def good_quality_metrics():
    return {
        "snr": {"snr_db": 5.0, "status": "ok"},
        "spectral_concentration": {"concentration": 0.5, "status": "ok"},
    }


@pytest.fixture
def good_tracking_summary():
    return {"tracking_fraction": 0.95, "total_samples": 300, "tracked_samples": 285, "lost_samples": 15}


@pytest.fixture
def good_motion_summary():
    return {"mean": 0.05, "median": 0.04, "max": 0.12, "num_valid": 300, "num_total": 300}


@pytest.fixture
def good_brightness_summary():
    return {"mean": 120.0, "median": 118.0, "min": 100.0, "max": 140.0, "num_valid": 300, "num_total": 300}


@pytest.fixture
def full_valid_input(
    valid_method_results,
    valid_region_results,
    good_quality_metrics,
    good_tracking_summary,
    good_motion_summary,
    good_brightness_summary,
):
    return {
        "method_results": valid_method_results,
        "region_results": valid_region_results,
        "quality_metrics": good_quality_metrics,
        "tracking_summary": good_tracking_summary,
        "motion_summary": good_motion_summary,
        "brightness_summary": good_brightness_summary,
        "duration_s": 15.0,
    }


# =============================================================================
# Helper function tests
# =============================================================================

class TestHelperFunctions:
    def test_extract_valid_method_bpms(self, valid_method_results):
        bpms = _extract_valid_method_bpms(valid_method_results)
        assert len(bpms) == 3
        assert all(np.isfinite(b) for b in bpms)

    def test_extract_valid_method_bpms_with_invalid(self):
        results = {
            "green": {"valid": True, "bpm": 72.0},
            "chrom": {"valid": False, "bpm": None},
            "pos": {"valid": True, "bpm": 71.0},
        }
        bpms = _extract_valid_method_bpms(results)
        assert len(bpms) == 2
        assert bpms == [72.0, 71.0]

    def test_extract_valid_method_bpms_with_nan(self):
        results = {
            "green": {"valid": True, "bpm": np.nan},
            "chrom": {"valid": True, "bpm": 73.0},
            "pos": {"valid": True, "bpm": 71.0},
        }
        bpms = _extract_valid_method_bpms(results)
        assert len(bpms) == 2

    def test_extract_valid_roi_bpms(self, valid_region_results):
        roi_bpms = _extract_valid_roi_bpms(valid_region_results)
        assert len(roi_bpms) == 3
        assert all(np.isfinite(v) for v in roi_bpms.values())

    def test_extract_valid_roi_bpms_partial(self):
        results = {
            "forehead": {"green": {"valid": True, "bpm": 72.0}, "chrom": {"valid": False}, "pos": {"valid": False}},
            "left_cheek": {"green": {"valid": False}, "chrom": {"valid": False}, "pos": {"valid": False}},
        }
        roi_bpms = _extract_valid_roi_bpms(results)
        assert len(roi_bpms) == 1
        assert "forehead" in roi_bpms


# =============================================================================
# Individual check tests
# =============================================================================

class TestMethodAgreement:
    def test_pass_three_valid(self, valid_method_results, default_thresholds):
        result = _check_method_agreement(valid_method_results, default_thresholds)
        assert result.passed is True
        assert result.value <= default_thresholds.max_method_range_bpm

    def test_fail_range_exceeded(self, default_thresholds):
        results = {
            "green": {"valid": True, "bpm": 72.0},
            "chrom": {"valid": True, "bpm": 85.0},  # 13 BPM difference
            "pos": {"valid": True, "bpm": 71.0},
        }
        result = _check_method_agreement(results, default_thresholds)
        assert result.passed is False
        assert result.value > default_thresholds.max_method_range_bpm

    def test_fail_insufficient_methods(self, default_thresholds):
        results = {
            "green": {"valid": True, "bpm": 72.0},
            "chrom": {"valid": False, "bpm": None},
            "pos": {"valid": False, "bpm": None},
        }
        result = _check_method_agreement(results, default_thresholds)
        assert result.passed is False
        assert result.value == 1.0

    def test_disabled_threshold(self, valid_method_results):
        thresholds = ReliabilityThresholds(max_method_range_bpm=None)
        result = _check_method_agreement(valid_method_results, thresholds)
        assert result.passed is None
        assert result.threshold is None


class TestROIAgreement:
    def test_pass_three_regions(self, valid_region_results, default_thresholds):
        result = _check_roi_agreement(valid_region_results, default_thresholds)
        assert result.passed is True
        assert result.value <= default_thresholds.max_roi_range_bpm

    def test_fail_range_exceeded(self, default_thresholds):
        results = {
            "forehead": {"green": {"valid": True, "bpm": 72.0}, "chrom": {"valid": True, "bpm": 73.0}, "pos": {"valid": True, "bpm": 71.0}},
            "left_cheek": {"green": {"valid": True, "bpm": 85.0}, "chrom": {"valid": True, "bpm": 86.0}, "pos": {"valid": True, "bpm": 84.0}},
        }
        result = _check_roi_agreement(results, default_thresholds)
        assert result.passed is False
        assert result.value > default_thresholds.max_roi_range_bpm

    def test_fail_insufficient_regions(self, default_thresholds):
        results = {
            "forehead": {"green": {"valid": True, "bpm": 72.0}, "chrom": {"valid": True, "bpm": 73.0}, "pos": {"valid": True, "bpm": 71.0}},
        }
        result = _check_roi_agreement(results, default_thresholds)
        assert result.passed is False
        assert result.value == 1.0

    def test_no_roi_data(self, default_thresholds):
        result = _check_roi_agreement({}, default_thresholds)
        assert result.passed is None
        assert result.value is None

    def test_disabled_threshold(self, valid_region_results):
        thresholds = ReliabilityThresholds(max_roi_range_bpm=None)
        result = _check_roi_agreement(valid_region_results, thresholds)
        assert result.passed is None


class TestSNR:
    def test_pass(self, good_quality_metrics, default_thresholds):
        result = _check_snr(good_quality_metrics, default_thresholds)
        assert result.passed is True
        assert result.value == 5.0

    def test_fail_below_threshold(self, default_thresholds):
        metrics = {"snr": {"snr_db": -2.0, "status": "ok"}}
        result = _check_snr(metrics, default_thresholds)
        assert result.passed is False

    def test_unavailable_snr(self, default_thresholds):
        metrics = {"snr": {"snr_db": None, "status": "zero_noise_power"}}
        result = _check_snr(metrics, default_thresholds)
        assert result.passed is None

    def test_missing_snr_data(self, default_thresholds):
        result = _check_snr({}, default_thresholds)
        assert result.passed is None

    def test_disabled_threshold(self, good_quality_metrics):
        thresholds = ReliabilityThresholds(min_snr_db=None)
        result = _check_snr(good_quality_metrics, thresholds)
        assert result.passed is None


class TestSpectralConcentration:
    def test_pass(self, good_quality_metrics, default_thresholds):
        result = _check_spectral_concentration(good_quality_metrics, default_thresholds)
        assert result.passed is True
        assert result.value == 0.5

    def test_fail_below_threshold(self, default_thresholds):
        metrics = {"spectral_concentration": {"concentration": 0.1, "status": "ok"}}
        result = _check_spectral_concentration(metrics, default_thresholds)
        assert result.passed is False

    def test_unavailable(self, default_thresholds):
        metrics = {"spectral_concentration": {"concentration": None, "status": "zero_total_power"}}
        result = _check_spectral_concentration(metrics, default_thresholds)
        assert result.passed is None

    def test_disabled_threshold(self, good_quality_metrics):
        thresholds = ReliabilityThresholds(min_spectral_concentration=None)
        result = _check_spectral_concentration(good_quality_metrics, thresholds)
        assert result.passed is None


class TestTracking:
    def test_pass(self, good_tracking_summary, default_thresholds):
        result = _check_tracking(good_tracking_summary, default_thresholds)
        assert result.passed is True
        assert result.value == 0.95

    def test_fail_below_threshold(self, default_thresholds):
        summary = {"tracking_fraction": 0.5}
        result = _check_tracking(summary, default_thresholds)
        assert result.passed is False

    def test_disabled_threshold(self, good_tracking_summary):
        thresholds = ReliabilityThresholds(min_tracking_fraction=None)
        result = _check_tracking(good_tracking_summary, thresholds)
        assert result.passed is None


class TestMotion:
    def test_pass(self, good_motion_summary, default_thresholds):
        thresholds = ReliabilityThresholds(max_motion=0.1)
        result = _check_motion(good_motion_summary, thresholds)
        assert result.passed is True

    def test_fail_exceeds_threshold(self, default_thresholds):
        thresholds = ReliabilityThresholds(max_motion=0.01)
        summary = {"mean": 0.05}
        result = _check_motion(summary, thresholds)
        assert result.passed is False

    def test_informational_no_threshold(self, good_motion_summary, default_thresholds):
        # Default threshold is None
        result = _check_motion(good_motion_summary, default_thresholds)
        assert result.passed is None
        assert result.threshold is None


class TestBrightness:
    def test_pass_within_range(self, good_brightness_summary):
        thresholds = ReliabilityThresholds(min_brightness=80.0, max_brightness=200.0)
        result = _check_brightness(good_brightness_summary, thresholds)
        assert result.passed is True

    def test_fail_below_min(self, good_brightness_summary):
        thresholds = ReliabilityThresholds(min_brightness=150.0)
        result = _check_brightness(good_brightness_summary, thresholds)
        assert result.passed is False

    def test_fail_above_max(self, good_brightness_summary):
        thresholds = ReliabilityThresholds(max_brightness=100.0)
        result = _check_brightness(good_brightness_summary, thresholds)
        assert result.passed is False

    def test_informational_no_thresholds(self, good_brightness_summary, default_thresholds):
        result = _check_brightness(good_brightness_summary, default_thresholds)
        assert result.passed is None
        assert result.threshold is None


class TestDuration:
    def test_pass(self, default_thresholds):
        result = _check_duration(15.0, default_thresholds)
        assert result.passed is True
        assert result.value == 15.0

    def test_fail_below_minimum(self, default_thresholds):
        result = _check_duration(5.0, default_thresholds)
        assert result.passed is False

    def test_fail_none(self, default_thresholds):
        result = _check_duration(None, default_thresholds)
        assert result.passed is False
        assert result.value is None


class TestDetermineConfidence:
    def test_low_when_rejected(self):
        checks = [
            type('Check', (), {'passed': False, 'name': 'test'})(),
        ]
        assert _determine_confidence(checks, False) == "LOW"

    def test_high_when_all_pass(self):
        checks = [
            type('Check', (), {'passed': True, 'name': 'test1'})(),
            type('Check', (), {'passed': True, 'name': 'test2'})(),
            type('Check', (), {'passed': True, 'name': 'test3'})(),
        ]
        assert _determine_confidence(checks, True) == "HIGH"

    def test_moderate_when_some_unavailable(self):
        checks = [
            type('Check', (), {'passed': True, 'name': 'test1'})(),
            type('Check', (), {'passed': None, 'name': 'test2'})(),
            type('Check', (), {'passed': True, 'name': 'test3'})(),
        ]
        assert _determine_confidence(checks, True) == "MODERATE"


# =============================================================================
# Integration tests - evaluate_reliability
# =============================================================================

class TestEvaluateReliability:
    def test_all_checks_pass(self, full_valid_input, default_thresholds):
        result = evaluate_reliability(full_valid_input, default_thresholds)
        assert result["status"] == "ACCEPTED"
        assert result["heart_rate_bpm"] is not None
        assert result["confidence"] in ("HIGH", "MODERATE")
        assert len(result["reasons"]) == 0

    def test_accepted_hr_is_median(self, full_valid_input, default_thresholds):
        # Methods: 72.0, 73.0, 71.5 -> median = 72.0
        result = evaluate_reliability(full_valid_input, default_thresholds)
        assert result["heart_rate_bpm"] == 72.0

    def test_method_disagreement_rejects(self, full_valid_input, default_thresholds):
        full_valid_input["method_results"]["chrom"]["bpm"] = 90.0  # 18 BPM diff
        result = evaluate_reliability(full_valid_input, default_thresholds)
        assert result["status"] == "REJECTED"
        assert result["heart_rate_bpm"] is None
        assert any("method_agreement" in r for r in result["reasons"])

    def test_roi_disagreement_rejects(self, full_valid_input, default_thresholds):
        full_valid_input["region_results"]["left_cheek"]["green"]["bpm"] = 90.0
        full_valid_input["region_results"]["left_cheek"]["chrom"]["bpm"] = 91.0
        full_valid_input["region_results"]["left_cheek"]["pos"]["bpm"] = 89.0
        result = evaluate_reliability(full_valid_input, default_thresholds)
        assert result["status"] == "REJECTED"
        assert result["heart_rate_bpm"] is None
        assert any("roi_agreement" in r for r in result["reasons"])

    def test_low_snr_rejects(self, full_valid_input, default_thresholds):
        full_valid_input["quality_metrics"]["snr"]["snr_db"] = -5.0
        result = evaluate_reliability(full_valid_input, default_thresholds)
        assert result["status"] == "REJECTED"
        assert result["heart_rate_bpm"] is None
        assert any("snr" in r for r in result["reasons"])

    def test_low_spectral_concentration_rejects(self, full_valid_input, default_thresholds):
        full_valid_input["quality_metrics"]["spectral_concentration"]["concentration"] = 0.1
        result = evaluate_reliability(full_valid_input, default_thresholds)
        assert result["status"] == "REJECTED"
        assert result["heart_rate_bpm"] is None
        assert any("spectral_concentration" in r for r in result["reasons"])

    def test_poor_tracking_rejects(self, full_valid_input, default_thresholds):
        full_valid_input["tracking_summary"]["tracking_fraction"] = 0.5
        result = evaluate_reliability(full_valid_input, default_thresholds)
        assert result["status"] == "REJECTED"
        assert result["heart_rate_bpm"] is None
        assert any("tracking" in r for r in result["reasons"])

    def test_insufficient_duration_rejects(self, full_valid_input, default_thresholds):
        full_valid_input["duration_s"] = 5.0
        result = evaluate_reliability(full_valid_input, default_thresholds)
        assert result["status"] == "REJECTED"
        assert result["heart_rate_bpm"] is None
        assert any("duration" in r for r in result["reasons"])

    def test_motion_rejects_when_threshold_configured(self, full_valid_input):
        thresholds = ReliabilityThresholds(max_motion=0.01)
        full_valid_input["motion_summary"]["mean"] = 0.05
        result = evaluate_reliability(full_valid_input, thresholds)
        assert result["status"] == "REJECTED"
        assert result["heart_rate_bpm"] is None
        assert any("motion" in r for r in result["reasons"])

    def test_motion_informational_when_no_threshold(self, full_valid_input, default_thresholds):
        # Default threshold is None
        full_valid_input["motion_summary"]["mean"] = 100.0  # huge motion
        result = evaluate_reliability(full_valid_input, default_thresholds)
        # Should not reject based on motion
        assert "motion" not in " ".join(result["reasons"])
        assert any("motion" in w for w in result["warnings"])

    def test_brightness_rejects_when_thresholds_configured(self, full_valid_input):
        thresholds = ReliabilityThresholds(min_brightness=200.0)
        result = evaluate_reliability(full_valid_input, thresholds)
        assert result["status"] == "REJECTED"
        assert result["heart_rate_bpm"] is None
        assert any("brightness" in r for r in result["reasons"])

    def test_brightness_informational_when_no_thresholds(self, full_valid_input, default_thresholds):
        result = evaluate_reliability(full_valid_input, default_thresholds)
        assert "brightness" not in " ".join(result["reasons"])
        assert any("brightness" in w for w in result["warnings"])

    def test_missing_optional_metrics(self, valid_method_results, default_thresholds):
        input_data = {
            "method_results": valid_method_results,
            "duration_s": 15.0,
        }
        result = evaluate_reliability(input_data, default_thresholds)
        assert result["status"] == "ACCEPTED"
        assert len(result["warnings"]) > 0

    def test_insufficient_valid_methods_rejects(self, default_thresholds):
        input_data = {
            "method_results": {
                "green": {"valid": True, "bpm": 72.0},
                "chrom": {"valid": False, "bpm": None},
                "pos": {"valid": False, "bpm": None},
            },
            "duration_s": 15.0,
        }
        result = evaluate_reliability(input_data, default_thresholds)
        assert result["status"] == "REJECTED"
        assert result["heart_rate_bpm"] is None

    def test_rejected_result_has_none_bpm(self, full_valid_input, default_thresholds):
        full_valid_input["method_results"]["chrom"]["bpm"] = 90.0
        result = evaluate_reliability(full_valid_input, default_thresholds)
        assert result["status"] == "REJECTED"
        assert result["heart_rate_bpm"] is None

    def test_threshold_customization(self, full_valid_input):
        # Custom thresholds that should pass
        thresholds = ReliabilityThresholds(
            max_method_range_bpm=20.0,
            max_roi_range_bpm=20.0,
            min_snr_db=-10.0,
            min_spectral_concentration=0.1,
            min_tracking_fraction=0.5,
            min_duration_s=5.0,
        )
        # Make everything bad but within custom thresholds
        full_valid_input["method_results"]["chrom"]["bpm"] = 85.0
        full_valid_input["quality_metrics"]["snr"]["snr_db"] = -5.0
        full_valid_input["quality_metrics"]["spectral_concentration"]["concentration"] = 0.15
        full_valid_input["tracking_summary"]["tracking_fraction"] = 0.6
        full_valid_input["duration_s"] = 8.0
        result = evaluate_reliability(full_valid_input, thresholds)
        assert result["status"] == "ACCEPTED"

    def test_invalid_input_handling(self, default_thresholds):
        result = evaluate_reliability({}, default_thresholds)
        assert result["status"] == "REJECTED"
        assert result["heart_rate_bpm"] is None
        assert len(result["reasons"]) > 0

    def test_checks_structure(self, full_valid_input, default_thresholds):
        result = evaluate_reliability(full_valid_input, default_thresholds)
        checks = result["checks"]
        expected_checks = {
            "method_agreement", "roi_agreement", "snr",
            "spectral_concentration", "tracking", "motion",
            "brightness", "duration"
        }
        assert set(checks.keys()) == expected_checks
        for check in checks.values():
            assert "name" in check
            assert "passed" in check
            assert "value" in check
            assert "threshold" in check
            assert "explanation" in check


if __name__ == "__main__":
    pytest.main([__file__, "-v"])