"""Signal Quality Intelligence & Adaptive Region Selection for FleetGuard rPPG.

Member 3 Responsibility:
This module evaluates camera-derived rPPG signals and multi-algorithm heart-rate
estimates to produce an explainable, deterministic acceptance or rejection decision.

Key capabilities:
1. Per-region signal quality (SNR, spectral concentration, peak prominence).
2. Motion contamination and lighting suitability assessment.
3. Signal dropout and tracking reliability assessment.
4. Multi-algorithm agreement comparison (Green, CHROM, POS).
5. Explainable Adaptive Region Selection across all eight facial ROIs.
6. Transparent rejection reasons and composite quality scoring.

ETHICAL & SCIENTIFIC SAFEGUARDS:
- Inter-algorithm agreement is treated as a consistency heuristic, NOT proof of ground-truth accuracy.
- Signal quality is explicitly distinguished from clinical physiological accuracy.
- Visual alertness indicators (EAR, MAR, head pose) are NOT evaluated here and must NEVER
  be used to declare driver fitness to drive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .quality_metrics import (
    calculate_brightness_summary,
    calculate_method_agreement,
    calculate_motion_summary,
    calculate_roi_agreement,
    calculate_snr,
    calculate_spectral_concentration,
    calculate_spectral_peak_prominence,
    calculate_tracking_summary,
)
from .multiregion import (
    ALL_REGIONS,
    _extract_region_data_from_timeseries,
    _split_into_continuous_segments,
    _resample_segment,
)
from .green import estimate_green_hr
from .chrom import estimate_chrom_hr
from .pos import estimate_pos_hr


@dataclass
class QualityThresholds:
    """Documented thresholds for signal quality intelligence and gating."""

    # 1. Temporal & Tracking Reliability
    min_duration_s: float = 10.0
    """Minimum continuous tracked segment duration in seconds.
    Justification: 10 s ensures Welch PSD frequency resolution of ~0.1 Hz,
    resolving individual cardiac peaks in the 0.75 - 3.0 Hz band."""

    min_tracking_fraction: float = 0.80
    """Minimum fraction of frames with positive face tracking (80%).
    Justification: Excessive dropped frames require interpolation that distorts rPPG periodicity."""

    # 2. Illumination & Motion Contamination
    min_brightness: float = 30.0
    """Minimum 8-bit regional pixel intensity.
    Justification: Low light (<30) elevates camera sensor photon shot noise and quantization errors."""

    max_brightness: float = 230.0
    """Maximum 8-bit regional pixel intensity.
    Justification: High brightness (>230) causes sensor clipping, eliminating pulsatile amplitude modulation."""

    max_mean_motion: float = 5.0
    """Maximum allowable mean frame displacement in pixels.
    Justification: Facial displacement causes non-pulsatile boundary shifts and Doppler optical artifacts."""

    max_peak_motion: float = 15.0
    """Maximum allowable peak motion spike in pixels."""

    # 3. Spectral Signal Quality (per region)
    min_region_snr_db: float = 0.0
    """Minimum acceptable SNR in dB within the 0.75 - 3.0 Hz band.
    Justification: SNR < 0 dB means noise power exceeds pulsatile cardiac power."""

    min_region_concentration: float = 0.25
    """Minimum spectral concentration within +/- 0.2 Hz of peak.
    Justification: Pure pulse signals concentrate >=25% energy in the primary harmonic."""

    min_valid_regions: int = 2
    """Minimum number of accepted facial regions required for a valid check-in.
    Justification: At least 2 regions are required to cross-verify physiological synchrony."""

    # 4. Multi-Algorithm & Regional Agreement (Consistency Checks)
    max_method_range_bpm: float = 8.0
    """Maximum allowable range across valid Green, CHROM, POS estimates.
    Justification: Divergence > 8 BPM indicates algorithm failure due to motion/chromatic distortion."""

    max_roi_range_bpm: float = 10.0
    """Maximum allowable range across valid facial region estimates.
    Justification: Human facial perfusion is synchronous; regional divergence >10 BPM indicates localized artifact."""


def _robust_extract_region_from_packets(
    packets: Sequence[Mapping[str, Any]], region_name: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Robustly extract per-region time series supporting both Member 1's contract and legacy fields.

    Returns:
        t: (N,) timestamps
        rgb: (N, 3) float64 array with NaN for missing/untracked
        motion: (N,) float64
        brightness: (N,) float64
        track_ok: (N,) bool
        total_count: int
        valid_count: int
    """
    t_list = []
    rgb_list = []
    motion_list = []
    bright_list = []
    track_list = []
    valid_count = 0

    for pkt in packets:
        if not isinstance(pkt, Mapping):
            continue

        ts = pkt.get("timestamp")
        if ts is None or not np.isfinite(ts):
            continue

        tracked = bool(pkt.get("track_ok", pkt.get("tracking_status", False)))

        # Motion
        m = pkt.get("motion", pkt.get("motion_displacement", 0.0))
        m_val = float(m) if m is not None and np.isfinite(m) else 0.0

        # Brightness (support scalar, dict by region, or overall_brightness)
        b_val = 0.0
        if "brightness" in pkt:
            b_field = pkt["brightness"]
            if isinstance(b_field, Mapping) and region_name in b_field:
                b_val = float(b_field[region_name]) if b_field[region_name] is not None else 0.0
            elif isinstance(b_field, (int, float, np.floating)):
                b_val = float(b_field)
        elif "overall_brightness" in pkt and pkt["overall_brightness"] is not None:
            b_val = float(pkt["overall_brightness"])

        # RGB
        reg_rgb = None
        if "rgb" in pkt and isinstance(pkt["rgb"], Mapping) and region_name in pkt["rgb"]:
            reg_rgb = pkt["rgb"][region_name]
        elif "regions" in pkt and isinstance(pkt["regions"], Mapping) and region_name in pkt["regions"]:
            reg_info = pkt["regions"][region_name]
            if isinstance(reg_info, Mapping):
                reg_rgb = reg_info.get("rgb")
                if b_val == 0.0 and "brightness" in reg_info and reg_info["brightness"] is not None:
                    b_val = float(reg_info["brightness"])

        # Parse RGB into [R, G, B]
        rgb_parsed = [np.nan, np.nan, np.nan]
        is_valid_sample = False

        if tracked and reg_rgb is not None:
            if isinstance(reg_rgb, Mapping):
                if all(k in reg_rgb for k in ("r", "g", "b")):
                    r, g, b = reg_rgb["r"], reg_rgb["g"], reg_rgb["b"]
                    if r is not None and g is not None and b is not None and np.all(np.isfinite([r, g, b])):
                        rgb_parsed = [float(r), float(g), float(b)]
                        is_valid_sample = True
            elif isinstance(reg_rgb, (list, tuple, np.ndarray)) and len(reg_rgb) >= 3:
                r, g, b = reg_rgb[0], reg_rgb[1], reg_rgb[2]
                if r is not None and g is not None and b is not None and np.all(np.isfinite([r, g, b])):
                    rgb_parsed = [float(r), float(g), float(b)]
                    is_valid_sample = True

        if is_valid_sample:
            valid_count += 1

        t_list.append(float(ts))
        rgb_list.append(rgb_parsed)
        motion_list.append(m_val)
        bright_list.append(b_val)
        track_list.append(is_valid_sample)

    t = np.asarray(t_list, dtype=np.float64)
    rgb = np.asarray(rgb_list, dtype=np.float64).reshape(-1, 3) if rgb_list else np.empty((0, 3), dtype=np.float64)
    motion = np.asarray(motion_list, dtype=np.float64)
    brightness = np.asarray(bright_list, dtype=np.float64)
    track_ok = np.asarray(track_list, dtype=bool)

    return t, rgb, motion, brightness, track_ok, len(t_list), valid_count


class QualityGate:
    """Evaluates measurement packets and multi-region rPPG signals."""

    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        self.thresholds = thresholds or QualityThresholds()

    def evaluate(
        self,
        data: Union[Sequence[Mapping[str, Any]], Mapping[str, Any]],
        hr_estimate: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Perform comprehensive quality evaluation."""
        is_packet_format = isinstance(data, Sequence)

        # 1. Parse global time, tracking, motion, and brightness
        if is_packet_format:
            if len(data) == 0:
                return self._empty_result(["Empty data sequence: no frames provided."])
            t_list, motion_list, track_list, bright_list = [], [], [], []
            for pkt in data:
                if not isinstance(pkt, Mapping):
                    continue
                ts = pkt.get("timestamp")
                if ts is None or not np.isfinite(ts):
                    continue
                t_list.append(float(ts))
                track_list.append(bool(pkt.get("track_ok", pkt.get("tracking_status", False))))
                m = pkt.get("motion", pkt.get("motion_displacement"))
                motion_list.append(float(m) if m is not None and np.isfinite(m) else 0.0)

                # Global brightness
                b = pkt.get("brightness")
                if isinstance(b, Mapping):
                    valid_b = [v for v in b.values() if v is not None and np.isfinite(v)]
                    b_val = float(np.mean(valid_b)) if valid_b else 0.0
                elif isinstance(b, (int, float, np.floating)) and np.isfinite(b):
                    b_val = float(b)
                elif pkt.get("overall_brightness") is not None:
                    b_val = float(pkt["overall_brightness"])
                else:
                    b_val = 0.0
                bright_list.append(b_val)

            t = np.asarray(t_list, dtype=np.float64)
            motion = np.asarray(motion_list, dtype=np.float64)
            track_ok = np.asarray(track_list, dtype=bool)
            brightness = np.asarray(bright_list, dtype=np.float64)
        else:
            t = np.asarray(data.get("t", []), dtype=np.float64)
            motion = np.asarray(data.get("motion", []), dtype=np.float64)
            track_ok = np.asarray(data.get("track_ok", []), dtype=bool)
            brightness = np.asarray(data.get("brightness", []), dtype=np.float64)

        if len(t) < 2:
            return self._empty_result(["Insufficient temporal samples (< 2 frames)."])

        total_duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0

        # Summaries
        track_summary = calculate_tracking_summary(track_ok)
        motion_summary = calculate_motion_summary(motion)
        bright_summary = calculate_brightness_summary(brightness)

        tracking_fraction = track_summary.get("tracking_fraction", 0.0)
        mean_motion_val = motion_summary.get("mean")
        mean_motion = float(mean_motion_val) if mean_motion_val is not None and np.isfinite(mean_motion_val) else 0.0
        peak_motion_val = motion_summary.get("max")
        peak_motion = float(peak_motion_val) if peak_motion_val is not None and np.isfinite(peak_motion_val) else 0.0
        mean_bright_val = bright_summary.get("mean")
        mean_brightness = float(mean_bright_val) if mean_bright_val is not None and np.isfinite(mean_bright_val) else 0.0

        # Check continuous segment duration
        continuous_segments = self._get_continuous_tracking_segments(t, track_ok)
        longest_continuous_s = max([seg[1] - seg[0] for seg in continuous_segments], default=0.0)

        # 2. Evaluate each region adaptively
        region_qualities: Dict[str, Dict[str, Any]] = {}
        selected_regions: List[str] = []
        rejected_regions: List[str] = []

        available_regions = list(ALL_REGIONS)
        if not is_packet_format and isinstance(data.get("rgb"), Mapping):
            available_regions = [r for r in ALL_REGIONS if r in data["rgb"]]

        for region in available_regions:
            r_eval = self._evaluate_single_region(
                data=data,
                region_name=region,
                is_packet_format=is_packet_format,
            )
            region_qualities[region] = r_eval

            if r_eval["usable"]:
                selected_regions.append(region)
            else:
                rejected_regions.append(region)

        # 3. Gating checks & reasons
        reasons: List[str] = []

        # Check A: Duration
        if longest_continuous_s < self.thresholds.min_duration_s:
            reasons.append(
                f"Continuous tracking duration ({longest_continuous_s:.1f}s) is below required "
                f"{self.thresholds.min_duration_s:.1f}s."
            )

        # Check B: Tracking Fraction
        if tracking_fraction < self.thresholds.min_tracking_fraction:
            reasons.append(
                f"Face tracking reliability ({tracking_fraction * 100:.1f}%) is below required "
                f"{self.thresholds.min_tracking_fraction * 100:.0f}%."
            )

        # Check C: Motion
        if mean_motion > self.thresholds.max_mean_motion:
            reasons.append(
                f"Excessive head motion: average displacement ({mean_motion:.1f}px) exceeds limit "
                f"({self.thresholds.max_mean_motion:.1f}px)."
            )
        if peak_motion > self.thresholds.max_peak_motion:
            reasons.append(
                f"Abrupt head movement spike ({peak_motion:.1f}px) exceeds limit "
                f"({self.thresholds.max_peak_motion:.1f}px)."
            )

        # Check D: Brightness
        if mean_brightness < self.thresholds.min_brightness:
            reasons.append(
                f"Insufficient lighting: average brightness ({mean_brightness:.1f}) is below minimum "
                f"({self.thresholds.min_brightness:.0f})."
            )
        elif mean_brightness > self.thresholds.max_brightness:
            reasons.append(
                f"Severe overexposure: average brightness ({mean_brightness:.1f}) exceeds maximum "
                f"({self.thresholds.max_brightness:.0f})."
            )

        # Check E: Usable Regions
        if len(selected_regions) < self.thresholds.min_valid_regions:
            reasons.append(
                f"Insufficient valid facial regions: {len(selected_regions)} accepted out of "
                f"{len(available_regions)} (minimum {self.thresholds.min_valid_regions} required)."
            )

        # Check F: Multi-Algorithm & Regional Agreement (if hr_estimate supplied)
        method_range = None
        roi_range = None
        if hr_estimate is not None:
            valid_methods_bpms = []
            for m in ("bpm_green", "bpm_chrom", "bpm_pos"):
                val = hr_estimate.get(m)
                if val is not None and np.isfinite(val):
                    valid_methods_bpms.append(float(val))

            if len(valid_methods_bpms) >= 2:
                method_range = float(np.max(valid_methods_bpms) - np.min(valid_methods_bpms))
                if method_range > self.thresholds.max_method_range_bpm:
                    reasons.append(
                        f"rPPG algorithm divergence: Green, CHROM, POS disagree by {method_range:.1f} BPM "
                        f"(max allowable {self.thresholds.max_method_range_bpm:.1f} BPM)."
                    )

            # Check region agreement
            reg_results = hr_estimate.get("region_results", {})
            roi_bpms = []
            for r_name in selected_regions:
                r_res = reg_results.get(r_name, {})
                bpm = r_res.get("bpm_aggregate", r_res.get("bpm"))
                if bpm is not None and np.isfinite(bpm):
                    roi_bpms.append(float(bpm))

            if len(roi_bpms) >= 2:
                roi_range = float(np.max(roi_bpms) - np.min(roi_bpms))
                if roi_range > self.thresholds.max_roi_range_bpm:
                    reasons.append(
                        f"Cross-region divergence: accepted regions disagree by {roi_range:.1f} BPM "
                        f"(max allowable {self.thresholds.max_roi_range_bpm:.1f} BPM)."
                    )

        # 4. Final Decision & Explainable Composite Score
        accepted = len(reasons) == 0

        score = self._compute_explainable_score(
            tracking_fraction=tracking_fraction,
            mean_motion=mean_motion,
            mean_brightness=mean_brightness,
            selected_regions=selected_regions,
            region_qualities=region_qualities,
            accepted=accepted,
        )

        confidence = "LOW"
        if accepted:
            confidence = "HIGH" if (score is not None and score >= 0.75) else "MODERATE"

        return {
            "score": score,
            "accepted": accepted,
            "reasons": reasons,
            "selected_regions": selected_regions,
            "rejected_regions": rejected_regions,
            "region_qualities": region_qualities,
            "confidence": confidence,
            "metrics_summary": {
                "total_duration_s": round(total_duration, 2),
                "continuous_duration_s": round(longest_continuous_s, 2),
                "tracking_fraction": round(tracking_fraction, 3),
                "mean_motion_px": round(mean_motion, 2),
                "mean_brightness": round(mean_brightness, 1),
                "method_range_bpm": round(method_range, 2) if method_range is not None else None,
                "roi_range_bpm": round(roi_range, 2) if roi_range is not None else None,
            },
        }

    def _evaluate_single_region(
        self,
        data: Union[Sequence[Mapping[str, Any]], Mapping[str, Any]],
        region_name: str,
        is_packet_format: bool,
    ) -> Dict[str, Any]:
        """Evaluate signal quality metrics for an individual facial ROI."""
        try:
            if is_packet_format:
                t, rgb, motion, brightness, track_ok, total_pkts, valid_count = _robust_extract_region_from_packets(
                    data, region_name
                )
            else:
                t, rgb, motion, brightness, track_ok, total_pkts, valid_count = _extract_region_data_from_timeseries(
                    data, region_name
                )

            if valid_count < 10:
                return {
                    "usable": False,
                    "reason": "Insufficient valid samples",
                    "snr_db": None,
                    "spectral_concentration": None,
                    "mean_brightness": None,
                }

            valid_brightness = brightness[track_ok]
            r_brightness = float(np.mean(valid_brightness)) if len(valid_brightness) > 0 else 0.0
            if r_brightness < self.thresholds.min_brightness:
                return {
                    "usable": False,
                    "reason": f"Underexposed ({r_brightness:.1f} < {self.thresholds.min_brightness:.0f})",
                    "snr_db": None,
                    "spectral_concentration": None,
                    "mean_brightness": round(r_brightness, 1),
                }
            if r_brightness > self.thresholds.max_brightness:
                return {
                    "usable": False,
                    "reason": f"Overexposed ({r_brightness:.1f} > {self.thresholds.max_brightness:.0f})",
                    "snr_db": None,
                    "spectral_concentration": None,
                    "mean_brightness": round(r_brightness, 1),
                }

            # Split valid samples into continuous segments
            t_valid = t[track_ok]
            rgb_valid = rgb[track_ok]
            motion_valid = motion[track_ok]
            brightness_valid = brightness[track_ok]

            if len(t_valid) < 10:
                return {
                    "usable": False,
                    "reason": "Insufficient valid tracked samples",
                    "snr_db": None,
                    "spectral_concentration": None,
                    "mean_brightness": round(r_brightness, 1),
                }

            segments = _split_into_continuous_segments(t_valid, rgb_valid, motion_valid, brightness_valid)
            if not segments:
                return {
                    "usable": False,
                    "reason": "No continuous tracking segments",
                    "snr_db": None,
                    "spectral_concentration": None,
                    "mean_brightness": round(r_brightness, 1),
                }

            # Pick longest segment
            longest_seg = max(segments, key=lambda s: s["t"][-1] - s["t"][0])
            dur = longest_seg["t"][-1] - longest_seg["t"][0]
            if dur < 5.0:  # Regional minimum window to evaluate spectrum
                return {
                    "usable": False,
                    "reason": f"Segment duration too short ({dur:.1f}s)",
                    "snr_db": None,
                    "spectral_concentration": None,
                    "mean_brightness": round(r_brightness, 1),
                }

            t_uni, rgb_uni, fs_resampled, _ = _resample_segment(
                longest_seg["t"],
                longest_seg["rgb"],
                longest_seg["motion"],
                longest_seg["brightness"],
            )
            green_res = estimate_green_hr(rgb_uni[:, 1], fs=fs_resampled, min_duration_s=5.0)

            if not green_res.get("valid", False):
                return {
                    "usable": False,
                    "reason": "Green channel spectrum invalid or flat",
                    "snr_db": None,
                    "spectral_concentration": None,
                    "mean_brightness": round(r_brightness, 1),
                }

            freqs = green_res.get("frequencies")
            psd = green_res.get("psd")
            peak_hz = green_res.get("frequency_hz")

            snr_info = calculate_snr(freqs, psd, peak_hz, signal_window_hz=0.25)
            conc_info = calculate_spectral_concentration(freqs, psd, peak_hz, concentration_window_hz=0.25)

            snr_db = snr_info.get("snr_db")
            concentration = conc_info.get("concentration")

            # Evaluate regional thresholds
            if snr_db is None or snr_db < self.thresholds.min_region_snr_db:
                return {
                    "usable": False,
                    "reason": f"Low SNR ({snr_db:.1f} dB < {self.thresholds.min_region_snr_db:.1f} dB)" if snr_db is not None else "Undefined SNR",
                    "snr_db": round(snr_db, 2) if snr_db is not None else None,
                    "spectral_concentration": round(concentration, 3) if concentration is not None else None,
                    "mean_brightness": round(r_brightness, 1),
                }

            if concentration is None or concentration < self.thresholds.min_region_concentration:
                return {
                    "usable": False,
                    "reason": f"Diffuse spectrum (concentration {concentration:.2f} < {self.thresholds.min_region_concentration:.2f})",
                    "snr_db": round(snr_db, 2),
                    "spectral_concentration": round(concentration, 3) if concentration is not None else None,
                    "mean_brightness": round(r_brightness, 1),
                }

            return {
                "usable": True,
                "reason": "Acceptable regional signal",
                "snr_db": round(snr_db, 2),
                "spectral_concentration": round(concentration, 3),
                "mean_brightness": round(r_brightness, 1),
            }

        except Exception as exc:
            return {
                "usable": False,
                "reason": f"Processing exception: {str(exc)}",
                "snr_db": None,
                "spectral_concentration": None,
                "mean_brightness": None,
            }

    def _get_continuous_tracking_segments(
        self, t: np.ndarray, track_ok: np.ndarray
    ) -> List[Tuple[float, float]]:
        """Identify start and end times of continuous tracked chunks."""
        segments = []
        if len(t) == 0:
            return segments

        in_seg = False
        start_t = 0.0

        for i in range(len(t)):
            if track_ok[i] and not in_seg:
                in_seg = True
                start_t = t[i]
            elif not track_ok[i] and in_seg:
                in_seg = False
                segments.append((start_t, t[i - 1]))

        if in_seg:
            segments.append((start_t, t[-1]))

        return segments

    def _compute_explainable_score(
        self,
        tracking_fraction: float,
        mean_motion: float,
        mean_brightness: float,
        selected_regions: List[str],
        region_qualities: Dict[str, Dict[str, Any]],
        accepted: bool,
    ) -> Optional[float]:
        """Derive an explainable composite quality score in [0.0, 1.0]."""
        if not accepted:
            return None

        # 1. Tracking score (0.0 to 1.0)
        s_track = float(np.clip(tracking_fraction, 0.0, 1.0))

        # 2. Motion score: 1.0 at 0px motion down to 0.0 at max_mean_motion
        s_motion = float(np.clip(1.0 - (mean_motion / max(self.thresholds.max_mean_motion, 1e-6)), 0.0, 1.0))

        # 3. Brightness score: optimal at mid-range (130), tapering off towards bounds
        mid = (self.thresholds.min_brightness + self.thresholds.max_brightness) / 2.0
        half_span = (self.thresholds.max_brightness - self.thresholds.min_brightness) / 2.0
        s_bright = float(np.clip(1.0 - (abs(mean_brightness - mid) / max(half_span, 1.0)), 0.0, 1.0))

        # 4. Regional SNR score: average of selected regions mapped from [0 dB, 10 dB] to [0.0, 1.0]
        snrs = [
            region_qualities[r]["snr_db"]
            for r in selected_regions
            if region_qualities[r].get("snr_db") is not None
        ]
        avg_snr = float(np.mean(snrs)) if snrs else 0.0
        s_snr = float(np.clip(avg_snr / 10.0, 0.0, 1.0))

        composite = 0.25 * s_track + 0.25 * s_motion + 0.25 * s_bright + 0.25 * s_snr
        return round(float(np.clip(composite, 0.0, 1.0)), 3)

    def _empty_result(self, reasons: List[str]) -> Dict[str, Any]:
        return {
            "score": None,
            "accepted": False,
            "reasons": reasons,
            "selected_regions": [],
            "rejected_regions": list(ALL_REGIONS),
            "region_qualities": {},
            "confidence": "LOW",
            "metrics_summary": {},
        }


def evaluate_quality(
    data: Union[Sequence[Mapping[str, Any]], Mapping[str, Any]],
    hr_estimate: Optional[Mapping[str, Any]] = None,
    thresholds: Optional[QualityThresholds] = None,
) -> Dict[str, Any]:
    """Public functional interface for Member 4 & team integration."""
    gate = QualityGate(thresholds=thresholds)
    return gate.evaluate(data=data, hr_estimate=hr_estimate)
