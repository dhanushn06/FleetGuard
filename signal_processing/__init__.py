"""Signal processing utilities for FleetGuard rPPG prototype.

This package provides signal conditioning, sampling, rPPG algorithms (Green, CHROM, POS),
multi-region aggregation, signal quality metrics, and the reliability quality gate.
"""

from .quality_intelligence import (
    QualityGate,
    QualityThresholds,
    evaluate_quality,
)
from .reliability_gate import (
    ReliabilityThresholds,
    evaluate_reliability,
)
from .quality_metrics import (
    calculate_snr,
    calculate_spectral_concentration,
    calculate_spectral_peak_prominence,
    calculate_method_agreement,
    calculate_roi_agreement,
    calculate_motion_summary,
    calculate_tracking_summary,
    calculate_brightness_summary,
)
from .multiregion import (
    ALL_REGIONS,
    BASELINE_3_REGIONS,
    estimate_multiregion_hr,
    estimate_multiregion_hr_advanced,
)
from .green import estimate_green_hr
from .chrom import estimate_chrom_hr
from .pos import estimate_pos_hr

__all__ = [
    "QualityGate",
    "QualityThresholds",
    "evaluate_quality",
    "ReliabilityThresholds",
    "evaluate_reliability",
    "calculate_snr",
    "calculate_spectral_concentration",
    "calculate_spectral_peak_prominence",
    "calculate_method_agreement",
    "calculate_roi_agreement",
    "calculate_motion_summary",
    "calculate_tracking_summary",
    "calculate_brightness_summary",
    "ALL_REGIONS",
    "BASELINE_3_REGIONS",
    "estimate_multiregion_hr",
    "estimate_multiregion_hr_advanced",
    "estimate_green_hr",
    "estimate_chrom_hr",
    "estimate_pos_hr",
]
