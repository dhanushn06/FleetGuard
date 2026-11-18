"""Reliability Gate for rPPG heart-rate estimation.

This module evaluates whether an estimated heart-rate measurement is sufficiently
reliable to be accepted or rejected based on signal-quality metrics.

It uses the outputs of the existing signal-processing and quality-metrics modules.
No machine learning, no medical diagnosis, no claims about driver fitness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np


@dataclass
class ReliabilityThresholds:
    """Configurable thresholds for the reliability gate.

    All thresholds are optional. If a threshold is None, the corresponding
    check is treated as informational (not a rejection criterion).
    """
    # Method agreement: maximum BPM range across valid methods
    max_method_range_bpm: Optional[float] = 8.0

    # ROI agreement: maximum BPM range across valid regions
    max_roi_range_bpm: Optional[float] = 10.0

    # SNR: minimum acceptable SNR in dB
    min_snr_db: Optional[float] = 0.0

    # Spectral concentration: minimum acceptable concentration (0-1)
    min_spectral_concentration: Optional[float] = 0.30

    # Tracking: minimum tracking fraction (0-1)
    min_tracking_fraction: Optional[float] = 0.80

    # Motion: maximum acceptable motion (mean displacement)
    # If None, motion is informational only
    max_motion: Optional[float] = None

    # Brightness: acceptable range [min, max]
    # If None, brightness is informational only
    min_brightness: Optional[float] = None
    max_brightness: Optional[float] = None

    # Duration: minimum continuous valid segment duration in seconds
    min_duration_s: float = 10.0

    # Minimum number of valid HR methods required
    min_valid_methods: int = 2

    # Minimum number of valid ROIs required (when ROI data available)
    min_valid_rois: int = 2


@dataclass
class CheckResult:
    """Result of a single reliability check."""
    name: str
    passed: Optional[bool]  # True=pass, False=fail, None=not_available
    value: Optional[float]
    threshold: Optional[float]
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "value": self.value,
            "threshold": self.threshold,
            "explanation": self.explanation,
        }


def _extract_valid_method_bpms(
    method_results: Mapping[str, Mapping[str, Any]]
) -> List[float]:
    """Extract valid BPM values from method results (Green, CHROM, POS)."""
    bpms = []
    for method in ("green", "chrom", "pos"):
        result = method_results.get(method)
        if isinstance(result, Mapping) and result.get("valid") is True:
            bpm = result.get("bpm")
            if isinstance(bpm, (int, float, np.floating)) and np.isfinite(bpm):
                bpms.append(float(bpm))
    return bpms


def _extract_valid_roi_bpms(
    region_results: Mapping[str, Mapping[str, Any]]
) -> Dict[str, float]:
    """Extract mean BPM per region from region results."""
    roi_bpms = {}
    for region, result in region_results.items():
        if not isinstance(result, Mapping):
            continue
        region_method_bpms = []
        for method in ("green", "chrom", "pos"):
            method_result = result.get(method)
            if isinstance(method_result, Mapping) and method_result.get("valid") is True:
                bpm = method_result.get("bpm")
                if isinstance(bpm, (int, float, np.floating)) and np.isfinite(bpm):
                    region_method_bpms.append(float(bpm))
        if region_method_bpms:
            roi_bpms[str(region)] = float(np.mean(region_method_bpms))
    return roi_bpms


def _check_method_agreement(
    method_results: Mapping[str, Mapping[str, Any]],
    thresholds: ReliabilityThresholds,
) -> CheckResult:
    """Check agreement across Green, CHROM, POS methods."""
    bpms = _extract_valid_method_bpms(method_results)
    num_valid = len(bpms)

    if num_valid < thresholds.min_valid_methods:
        return CheckResult(
            name="method_agreement",
            passed=False,
            value=float(num_valid),
            threshold=float(thresholds.min_valid_methods),
            explanation=f"Only {num_valid} valid method(s), need at least {thresholds.min_valid_methods}",
        )

    if thresholds.max_method_range_bpm is None:
        return CheckResult(
            name="method_agreement",
            passed=None,
            value=float(np.max(bpms) - np.min(bpms)) if num_valid > 1 else 0.0,
            threshold=None,
            explanation=f"Method agreement check disabled (threshold not configured). Range: {np.max(bpms) - np.min(bpms):.1f} BPM",
        )

    bpm_range = float(np.max(bpms) - np.min(bpms))
    passed = bpm_range <= thresholds.max_method_range_bpm

    return CheckResult(
        name="method_agreement",
        passed=passed,
        value=bpm_range,
        threshold=thresholds.max_method_range_bpm,
        explanation=(
            f"Method BPM range: {bpm_range:.1f} BPM "
            f"{'(PASS)' if passed else '(FAIL - exceeds threshold)'}"
        ),
    )


def _check_roi_agreement(
    region_results: Mapping[str, Mapping[str, Any]],
    thresholds: ReliabilityThresholds,
) -> CheckResult:
    """Check agreement across facial regions."""
    roi_bpms = _extract_valid_roi_bpms(region_results)
    num_valid_rois = len(roi_bpms)

    if num_valid_rois == 0:
        return CheckResult(
            name="roi_agreement",
            passed=None,
            value=None,
            threshold=None,
            explanation="No valid ROI data available",
        )

    if num_valid_rois < thresholds.min_valid_rois:
        return CheckResult(
            name="roi_agreement",
            passed=False,
            value=float(num_valid_rois),
            threshold=float(thresholds.min_valid_rois),
            explanation=f"Only {num_valid_rois} valid ROI(s), need at least {thresholds.min_valid_rois}",
        )

    if thresholds.max_roi_range_bpm is None:
        bpm_range = float(np.max(list(roi_bpms.values())) - np.min(list(roi_bpms.values()))) if num_valid_rois > 1 else 0.0
        return CheckResult(
            name="roi_agreement",
            passed=None,
            value=bpm_range,
            threshold=None,
            explanation=f"ROI agreement check disabled (threshold not configured). Range: {bpm_range:.1f} BPM",
        )

    bpm_range = float(np.max(list(roi_bpms.values())) - np.min(list(roi_bpms.values())))
    passed = bpm_range <= thresholds.max_roi_range_bpm

    return CheckResult(
        name="roi_agreement",
        passed=passed,
        value=bpm_range,
        threshold=thresholds.max_roi_range_bpm,
        explanation=(
            f"ROI BPM range: {bpm_range:.1f} BPM "
            f"{'(PASS)' if passed else '(FAIL - exceeds threshold)'}"
        ),
    )


def _check_snr(
    quality_metrics: Mapping[str, Any],
    thresholds: ReliabilityThresholds,
) -> CheckResult:
    """Check SNR from quality metrics."""
    snr_data = quality_metrics.get("snr")
    if not isinstance(snr_data, Mapping):
        return CheckResult(
            name="snr",
            passed=None,
            value=None,
            threshold=thresholds.min_snr_db,
            explanation="SNR data not available",
        )

    snr_db = snr_data.get("snr_db")
    if snr_db is None or not np.isfinite(snr_db):
        return CheckResult(
            name="snr",
            passed=None,
            value=None,
            threshold=thresholds.min_snr_db,
            explanation=f"SNR unavailable (status: {snr_data.get('status', 'unknown')})",
        )

    if thresholds.min_snr_db is None:
        return CheckResult(
            name="snr",
            passed=None,
            value=float(snr_db),
            threshold=None,
            explanation=f"SNR check disabled (threshold not configured). SNR: {snr_db:.1f} dB",
        )

    passed = float(snr_db) >= thresholds.min_snr_db
    return CheckResult(
        name="snr",
        passed=passed,
        value=float(snr_db),
        threshold=thresholds.min_snr_db,
        explanation=f"SNR: {snr_db:.1f} dB {'(PASS)' if passed else '(FAIL - below threshold)'}",
    )


def _check_spectral_concentration(
    quality_metrics: Mapping[str, Any],
    thresholds: ReliabilityThresholds,
) -> CheckResult:
    """Check spectral concentration from quality metrics."""
    conc_data = quality_metrics.get("spectral_concentration")
    if not isinstance(conc_data, Mapping):
        return CheckResult(
            name="spectral_concentration",
            passed=None,
            value=None,
            threshold=thresholds.min_spectral_concentration,
            explanation="Spectral concentration data not available",
        )

    concentration = conc_data.get("concentration")
    if concentration is None or not np.isfinite(concentration):
        return CheckResult(
            name="spectral_concentration",
            passed=None,
            value=None,
            threshold=thresholds.min_spectral_concentration,
            explanation=f"Spectral concentration unavailable (status: {conc_data.get('status', 'unknown')})",
        )

    if thresholds.min_spectral_concentration is None:
        return CheckResult(
            name="spectral_concentration",
            passed=None,
            value=float(concentration),
            threshold=None,
            explanation=f"Spectral concentration check disabled (threshold not configured). Value: {concentration:.3f}",
        )

    passed = float(concentration) >= thresholds.min_spectral_concentration
    return CheckResult(
        name="spectral_concentration",
        passed=passed,
        value=float(concentration),
        threshold=thresholds.min_spectral_concentration,
        explanation=f"Spectral concentration: {concentration:.3f} {'(PASS)' if passed else '(FAIL - below threshold)'}",
    )


def _check_tracking(
    tracking_summary: Mapping[str, Any],
    thresholds: ReliabilityThresholds,
) -> CheckResult:
    """Check tracking fraction."""
    if not isinstance(tracking_summary, Mapping):
        return CheckResult(
            name="tracking",
            passed=None,
            value=None,
            threshold=thresholds.min_tracking_fraction,
            explanation="Tracking summary not available",
        )

    tracking_fraction = tracking_summary.get("tracking_fraction")
    if tracking_fraction is None or not np.isfinite(tracking_fraction):
        return CheckResult(
            name="tracking",
            passed=None,
            value=None,
            threshold=thresholds.min_tracking_fraction,
            explanation="Tracking fraction unavailable",
        )

    if thresholds.min_tracking_fraction is None:
        return CheckResult(
            name="tracking",
            passed=None,
            value=float(tracking_fraction),
            threshold=None,
            explanation=f"Tracking check disabled (threshold not configured). Fraction: {tracking_fraction:.2f}",
        )

    passed = float(tracking_fraction) >= thresholds.min_tracking_fraction
    return CheckResult(
        name="tracking",
        passed=passed,
        value=float(tracking_fraction),
        threshold=thresholds.min_tracking_fraction,
        explanation=f"Tracking fraction: {tracking_fraction:.2f} {'(PASS)' if passed else '(FAIL - below threshold)'}",
    )


def _check_motion(
    motion_summary: Mapping[str, Any],
    thresholds: ReliabilityThresholds,
) -> CheckResult:
    """Check motion level."""
    if not isinstance(motion_summary, Mapping):
        return CheckResult(
            name="motion",
            passed=None,
            value=None,
            threshold=thresholds.max_motion,
            explanation="Motion summary not available",
        )

    motion_mean = motion_summary.get("mean")
    if motion_mean is None or not np.isfinite(motion_mean):
        return CheckResult(
            name="motion",
            passed=None,
            value=None,
            threshold=thresholds.max_motion,
            explanation="Motion mean unavailable",
        )

    if thresholds.max_motion is None:
        return CheckResult(
            name="motion",
            passed=None,
            value=float(motion_mean),
            threshold=None,
            explanation=f"Motion check disabled (threshold not configured). Mean motion: {motion_mean:.3f}",
        )

    passed = float(motion_mean) <= thresholds.max_motion
    return CheckResult(
        name="motion",
        passed=passed,
        value=float(motion_mean),
        threshold=thresholds.max_motion,
        explanation=f"Mean motion: {motion_mean:.3f} {'(PASS)' if passed else '(FAIL - exceeds threshold)'}",
    )


def _check_brightness(
    brightness_summary: Mapping[str, Any],
    thresholds: ReliabilityThresholds,
) -> CheckResult:
    """Check brightness range."""
    if not isinstance(brightness_summary, Mapping):
        return CheckResult(
            name="brightness",
            passed=None,
            value=None,
            threshold=None,
            explanation="Brightness summary not available",
        )

    brightness_mean = brightness_summary.get("mean")
    if brightness_mean is None or not np.isfinite(brightness_mean):
        return CheckResult(
            name="brightness",
            passed=None,
            value=None,
            threshold=None,
            explanation="Brightness mean unavailable",
        )

    # Check if any threshold is configured
    has_min = thresholds.min_brightness is not None
    has_max = thresholds.max_brightness is not None

    if not has_min and not has_max:
        return CheckResult(
            name="brightness",
            passed=None,
            value=float(brightness_mean),
            threshold=None,
            explanation=f"Brightness check disabled (thresholds not configured). Mean brightness: {brightness_mean:.1f}",
        )

    passed = True
    reasons = []
    if has_min and float(brightness_mean) < thresholds.min_brightness:
        passed = False
        reasons.append(f"below min ({thresholds.min_brightness})")
    if has_max and float(brightness_mean) > thresholds.max_brightness:
        passed = False
        reasons.append(f"above max ({thresholds.max_brightness})")

    threshold_str = ""
    if has_min and has_max:
        threshold_str = f"[{thresholds.min_brightness}, {thresholds.max_brightness}]"
    elif has_min:
        threshold_str = f">= {thresholds.min_brightness}"
    elif has_max:
        threshold_str = f"<= {thresholds.max_brightness}"

    explanation = f"Mean brightness: {brightness_mean:.1f} "
    if passed:
        explanation += f"(PASS, range {threshold_str})"
    else:
        explanation += f"(FAIL - {', '.join(reasons)})"

    return CheckResult(
        name="brightness",
        passed=passed,
        value=float(brightness_mean),
        threshold=threshold_str if threshold_str else None,
        explanation=explanation,
    )


def _check_duration(
    duration_s: Optional[float],
    thresholds: ReliabilityThresholds,
) -> CheckResult:
    """Check signal duration."""
    if duration_s is None or not np.isfinite(duration_s):
        return CheckResult(
            name="duration",
            passed=False,
            value=None,
            threshold=thresholds.min_duration_s,
            explanation="Duration unavailable",
        )

    passed = float(duration_s) >= thresholds.min_duration_s
    return CheckResult(
        name="duration",
        passed=passed,
        value=float(duration_s),
        threshold=thresholds.min_duration_s,
        explanation=f"Duration: {duration_s:.1f} s {'(PASS)' if passed else '(FAIL - below minimum)'}",
    )


def _determine_confidence(
    checks: List[CheckResult],
    overall_passed: bool,
) -> str:
    """Determine confidence level based on check results."""
    if not overall_passed:
        return "LOW"

    # Count available checks that passed
    available_checks = [c for c in checks if c.passed is not None]
    passed_checks = [c for c in available_checks if c.passed is True]

    # If all available checks pass, HIGH confidence
    if len(passed_checks) == len(available_checks) and len(available_checks) >= 3:
        return "HIGH"

    # Otherwise MODERATE
    return "MODERATE"


def evaluate_reliability(
    input_data: Mapping[str, Any],
    thresholds: Optional[ReliabilityThresholds] = None,
) -> Dict[str, Any]:
    """Evaluate reliability of an rPPG heart-rate estimate.

    Args:
        input_data: Dictionary containing:
            - "method_results": dict with "green", "chrom", "pos" results
            - "region_results": dict with per-region results (optional)
            - "quality_metrics": dict with SNR, spectral_concentration, etc. (optional)
            - "tracking_summary": dict from calculate_tracking_summary (optional)
            - "motion_summary": dict from calculate_motion_summary (optional)
            - "brightness_summary": dict from calculate_brightness_summary (optional)
            - "duration_s": float, continuous valid segment duration in seconds (optional)
        thresholds: Optional ReliabilityThresholds for customization

    Returns:
        Dictionary with:
            - "status": "ACCEPTED" or "REJECTED"
            - "heart_rate_bpm": float or None
            - "confidence": "HIGH", "MODERATE", or "LOW"
            - "checks": dict of check results
            - "reasons": list of rejection reasons (empty if accepted)
            - "warnings": list of warnings
    """
    if thresholds is None:
        thresholds = ReliabilityThresholds()

    # Extract inputs
    method_results = input_data.get("method_results", {})
    region_results = input_data.get("region_results", {})
    quality_metrics = input_data.get("quality_metrics", {})
    tracking_summary = input_data.get("tracking_summary", {})
    motion_summary = input_data.get("motion_summary", {})
    brightness_summary = input_data.get("brightness_summary", {})
    duration_s = input_data.get("duration_s")

    # Run all checks
    checks = []

    # 1. Method agreement (mandatory)
    method_check = _check_method_agreement(method_results, thresholds)
    checks.append(method_check)

    # 2. ROI agreement (if region data available)
    roi_check = _check_roi_agreement(region_results, thresholds)
    checks.append(roi_check)

    # 3. SNR
    snr_check = _check_snr(quality_metrics, thresholds)
    checks.append(snr_check)

    # 4. Spectral concentration
    conc_check = _check_spectral_concentration(quality_metrics, thresholds)
    checks.append(conc_check)

    # 5. Tracking
    tracking_check = _check_tracking(tracking_summary, thresholds)
    checks.append(tracking_check)

    # 6. Motion
    motion_check = _check_motion(motion_summary, thresholds)
    checks.append(motion_check)

    # 7. Brightness
    brightness_check = _check_brightness(brightness_summary, thresholds)
    checks.append(brightness_check)

    # 8. Duration
    duration_check = _check_duration(duration_s, thresholds)
    checks.append(duration_check)

    # Determine overall pass/fail
    # A check fails only if passed is False (not None)
    mandatory_checks = [c for c in checks if c.name in ("method_agreement", "duration")]
    optional_checks = [c for c in checks if c.name not in ("method_agreement", "duration")]

    # Method agreement and duration are mandatory
    all_mandatory_pass = all(c.passed is True for c in mandatory_checks)
    # Optional checks: fail only if explicitly failed (not None)
    all_optional_pass = all(c.passed is not False for c in optional_checks)

    overall_passed = all_mandatory_pass and all_optional_pass

    # Compute accepted HR (median of valid methods)
    valid_bpms = _extract_valid_method_bpms(method_results)
    heart_rate_bpm = float(np.median(valid_bpms)) if valid_bpms else None

    # If rejected, HR is None
    if not overall_passed:
        heart_rate_bpm = None

    # Collect reasons and warnings
    reasons = []
    warnings = []

    for check in checks:
        if check.passed is False:
            reasons.append(f"{check.name}: {check.explanation}")
        elif check.passed is None and check.threshold is not None:
            # Check was available but threshold not configured - warning
            warnings.append(f"{check.name}: threshold not configured, check skipped")
        elif check.passed is None and check.threshold is None and check.value is not None:
            # Check is informational (threshold disabled) but data was available
            warnings.append(f"{check.name}: check disabled (no threshold configured), value={check.value}")

    # Add specific warnings for missing optional data
    if not quality_metrics:
        warnings.append("quality_metrics not provided, SNR and spectral concentration checks skipped")
    if not tracking_summary:
        warnings.append("tracking_summary not provided, tracking check skipped")
    if not motion_summary:
        warnings.append("motion_summary not provided, motion check skipped")
    if not brightness_summary:
        warnings.append("brightness_summary not provided, brightness check skipped")
    if duration_s is None:
        warnings.append("duration_s not provided, duration check skipped")

    confidence = _determine_confidence(checks, overall_passed)

    return {
        "status": "ACCEPTED" if overall_passed else "REJECTED",
        "heart_rate_bpm": heart_rate_bpm,
        "confidence": confidence,
        "checks": {c.name: c.to_dict() for c in checks},
        "reasons": reasons,
        "warnings": warnings,
    }