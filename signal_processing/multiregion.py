"""Multi-region rPPG heart-rate estimation from camera measurement packets.

This module bridges Member 1's camera pipeline output format to the existing
single-ROI heart-rate estimators (Green, CHROM, POS). It collects measurement
packets, extracts per-region RGB time series, handles missing/invalid data,
splits on tracking gaps, resamples each continuous segment independently,
and runs all three algorithms on the longest qualifying segment per region.

Camera packet format (Member 1):
    {
        "timestamp": float,
        "wall_time": str or None,
        "tracking_status": bool,
        "motion_displacement": float or None,
        "brightness": float or None,
        "regions": {
            "forehead_center": {
                "rgb": {"r": float, "g": float, "b": float},
                "brightness": float,
                "pixel_count": int
            },
            "forehead_left": { ... },
            "forehead_right": { ... },
            "left_cheek_upper": { ... },
            "left_cheek_lower": { ... },
            "right_cheek_upper": { ... },
            "right_cheek_lower": { ... },
            "nose_bridge": { ... }
        }
    }
When tracking fails, "regions" may be None and other fields may be None.

Also supports time-series format:
    {
        "t": timestamps,
        "rgb": {
            "forehead_center": RGB time series (N, 3),
            ...
        },
        "motion": motion measurements,
        "track_ok": tracking status,
        "brightness": brightness measurements
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .chrom import estimate_chrom_hr
from .green import estimate_green_hr
from .pos import estimate_pos_hr
from .quality_metrics import (
    calculate_snr,
    calculate_spectral_concentration,
    calculate_spectral_peak_prominence,
)
from .sampling import (
    estimate_sampling_rate,
    generate_uniform_timestamps,
    resample_nd_signal,
)

# All eight supported region names
ALL_REGIONS = (
    "forehead_center",
    "forehead_left",
    "forehead_right",
    "left_cheek_upper",
    "left_cheek_lower",
    "right_cheek_upper",
    "right_cheek_lower",
    "nose_bridge",
)

# Three-region baseline configuration (research comparison)
BASELINE_3_REGIONS = (
    "forehead_center",
    "left_cheek_upper",
    "right_cheek_upper",
)

# Gap threshold: 5x median dt. Tolerates normal webcam jitter (typically < 2-3x)
# while catching genuine tracking loss (seconds of missing frames).
_GAP_THRESHOLD_FACTOR = 5.0
# Small tolerance for duration checks after resampling to account for
# floating-point edge effects in uniform grid generation.
_DURATION_TOLERANCE_S = 0.1


@dataclass
class RegionConfig:
    """Configuration for region selection and processing."""
    regions: Tuple[str, ...] = ALL_REGIONS
    min_valid_regions: int = 2
    adaptive_mode: bool = False
    adaptive_min_snr_db: float = 0.0
    adaptive_min_concentration: float = 0.3
    adaptive_min_prominence: float = 0.0


@dataclass
class AggregationConfig:
    """Configuration for result aggregation."""
    method: str = "median"  # Only "median" supported currently
    min_valid_regions: int = 2


def _invalid_region_result(region: str, reason: str) -> Dict[str, Any]:
    """Build a standardized invalid result for a single region."""
    invalid_algo = {
        "valid": False,
        "bpm": None,
        "frequency_hz": None,
        "reason": reason,
    }
    return {
        region: {
            "green": dict(invalid_algo),
            "chrom": dict(invalid_algo),
            "pos": dict(invalid_algo),
            "aggregated_bpm": None,
            "metadata": {
                "valid_samples": 0,
                "missing_samples": 0,
                "sampling_rate_hz": None,
                "tracking_fraction": 0.0,
                "motion_mean": None,
                "brightness_mean": None,
                "num_continuous_segments": 0,
                "longest_segment_duration_s": 0.0,
                "selected_segment_duration_s": 0.0,
                "gap_warning": None,
            },
        }
    }


def _extract_region_data_from_packets(
    packets: Sequence[Mapping[str, Any]], region_name: str
) -> Tuple[
    np.ndarray,  # timestamps (all frames)
    np.ndarray,  # rgb (N, 3) with NaN for missing
    np.ndarray,  # motion
    np.ndarray,  # brightness
    np.ndarray,  # track_ok (bool)
    int,         # total packets
    int,         # valid samples for this region
]:
    """Extract per-region time series from camera packets (Member 1 format).

    Returns arrays aligned to ALL frames (including tracking failures).
    Valid samples have track_ok=True and finite RGB; others have NaN RGB and track_ok=False.
    """
    timestamps = []
    rgb_vals = []
    motion_vals = []
    brightness_vals = []
    track_ok_vals = []

    for pkt in packets:
        ts = pkt.get("timestamp")
        if ts is None or not np.isfinite(ts):
            continue

        tracking = bool(pkt.get("tracking_status", False))
        motion = pkt.get("motion_displacement")
        brightness = pkt.get("brightness")

        # Only collect if tracking is good and region data exists with valid RGB
        if tracking:
            regions = pkt.get("regions")
            if regions and region_name in regions:
                region_data = regions[region_name]
                rgb_dict = region_data.get("rgb")
                if rgb_dict and all(k in rgb_dict for k in ("r", "g", "b")):
                    r = rgb_dict["r"]
                    g = rgb_dict["g"]
                    b = rgb_dict["b"]
                    if all(np.isfinite(v) for v in (r, g, b)):
                        timestamps.append(float(ts))
                        rgb_vals.append([float(r), float(g), float(b)])
                        motion_vals.append(float(motion) if motion is not None and np.isfinite(motion) else 0.0)
                        brightness_vals.append(float(brightness) if brightness is not None and np.isfinite(brightness) else 0.0)
                        track_ok_vals.append(True)
                        continue

        # Tracking failed or region missing or invalid RGB - record as missing sample
        timestamps.append(float(ts))
        rgb_vals.append([np.nan, np.nan, np.nan])
        motion_vals.append(float(motion) if motion is not None and np.isfinite(motion) else 0.0)
        brightness_vals.append(float(brightness) if brightness is not None and np.isfinite(brightness) else 0.0)
        track_ok_vals.append(False)

    if not timestamps:
        return (
            np.array([], dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=bool),
            len(packets),
            0,
        )

    t_arr = np.array(timestamps, dtype=np.float64)
    rgb_arr = np.array(rgb_vals, dtype=np.float64)
    motion_arr = np.array(motion_vals, dtype=np.float64)
    brightness_arr = np.array(brightness_vals, dtype=np.float64)
    track_ok_arr = np.array(track_ok_vals, dtype=bool)

    valid_mask = track_ok_arr
    valid_count = int(np.sum(valid_mask))

    return t_arr, rgb_arr, motion_arr, brightness_arr, track_ok_arr, len(packets), valid_count


def _extract_region_data_from_timeseries(
    data: Mapping[str, Any], region_name: str
) -> Tuple[
    np.ndarray,  # timestamps
    np.ndarray,  # rgb (N, 3)
    np.ndarray,  # motion
    np.ndarray,  # brightness
    np.ndarray,  # track_ok (bool)
    int,         # total samples
    int,         # valid samples
]:
    """Extract per-region time series from time-series format."""
    t = np.asarray(data.get("t", []), dtype=np.float64)
    rgb_dict = data.get("rgb", {})
    motion = np.asarray(data.get("motion", []), dtype=np.float64)
    brightness = np.asarray(data.get("brightness", []), dtype=np.float64)
    track_ok = np.asarray(data.get("track_ok", []), dtype=bool)

    if t.size == 0:
        return (
            np.array([], dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=bool),
            0,
            0,
        )

    # Get RGB for this region
    region_rgb = rgb_dict.get(region_name)
    if region_rgb is None:
        # Region not present in data
        rgb_arr = np.full((t.size, 3), np.nan, dtype=np.float64)
        track_ok_arr = np.zeros(t.size, dtype=bool)
        valid_count = 0
    else:
        rgb_arr = np.asarray(region_rgb, dtype=np.float64)
        if rgb_arr.ndim != 2 or rgb_arr.shape[1] != 3:
            rgb_arr = np.full((t.size, 3), np.nan, dtype=np.float64)
            track_ok_arr = np.zeros(t.size, dtype=bool)
            valid_count = 0
        else:
            # Valid samples: track_ok=True and finite RGB
            finite_rgb = np.all(np.isfinite(rgb_arr), axis=1)
            track_ok_arr = track_ok & finite_rgb
            valid_count = int(np.sum(track_ok_arr))
            # Set invalid samples to NaN
            rgb_arr[~track_ok_arr] = np.nan

    # Ensure motion/brightness match length
    if motion.size != t.size:
        motion = np.zeros(t.size, dtype=np.float64)
    if brightness.size != t.size:
        brightness = np.zeros(t.size, dtype=np.float64)
    if track_ok.size != t.size:
        track_ok_arr = np.zeros(t.size, dtype=bool)

    return t, rgb_arr, motion, brightness, track_ok_arr, t.size, valid_count


def _split_into_continuous_segments(
    t_valid: np.ndarray,
    rgb_valid: np.ndarray,
    motion_valid: np.ndarray,
    brightness_valid: np.ndarray,
    gap_threshold_factor: float = _GAP_THRESHOLD_FACTOR,
) -> List[Dict[str, Any]]:
    """Split valid samples into continuous segments based on timestamp gaps.

    A new segment starts when the time difference between consecutive valid samples
    exceeds gap_threshold_factor * median_dt.

    Returns:
        List of segment dicts, each containing:
        - 't': timestamps for this segment
        - 'rgb': RGB values for this segment
        - 'motion': motion values for this segment
        - 'brightness': brightness values for this segment
        - 'duration_s': segment duration in seconds
        - 'num_samples': number of samples in segment
    """
    if t_valid.size < 2:
        return [{
            "t": t_valid,
            "rgb": rgb_valid,
            "motion": motion_valid,
            "brightness": brightness_valid,
            "duration_s": 0.0,
            "num_samples": t_valid.size,
        }]

    dt = np.diff(t_valid)
    median_dt = float(np.median(dt))
    if median_dt <= 0:
        # Fallback: treat as single segment if timestamps are problematic
        return [{
            "t": t_valid,
            "rgb": rgb_valid,
            "motion": motion_valid,
            "brightness": brightness_valid,
            "duration_s": float(t_valid[-1] - t_valid[0]) if t_valid.size > 1 else 0.0,
            "num_samples": t_valid.size,
        }]

    gap_threshold = gap_threshold_factor * median_dt
    gap_indices = np.where(dt > gap_threshold)[0]

    segments = []
    start_idx = 0
    for gap_idx in gap_indices:
        end_idx = gap_idx + 1  # gap is between gap_idx and gap_idx+1
        seg_t = t_valid[start_idx:end_idx]
        seg_rgb = rgb_valid[start_idx:end_idx]
        seg_motion = motion_valid[start_idx:end_idx]
        seg_brightness = brightness_valid[start_idx:end_idx]
        segments.append({
            "t": seg_t,
            "rgb": seg_rgb,
            "motion": seg_motion,
            "brightness": seg_brightness,
            "duration_s": float(seg_t[-1] - seg_t[0]) if seg_t.size > 1 else 0.0,
            "num_samples": seg_t.size,
        })
        start_idx = end_idx

    # Final segment
    seg_t = t_valid[start_idx:]
    seg_rgb = rgb_valid[start_idx:]
    seg_motion = motion_valid[start_idx:]
    seg_brightness = brightness_valid[start_idx:]
    segments.append({
        "t": seg_t,
        "rgb": seg_rgb,
        "motion": seg_motion,
        "brightness": seg_brightness,
        "duration_s": float(seg_t[-1] - seg_t[0]) if seg_t.size > 1 else 0.0,
        "num_samples": seg_t.size,
    })

    return segments


def _resample_segment(
    t_seg: np.ndarray,
    rgb_seg: np.ndarray,
    motion_seg: np.ndarray,
    brightness_seg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
    """Resample a single continuous segment to a uniform grid.

    Returns:
        (t_uniform, rgb_uniform, fs, metadata_dict)
    """
    fs = estimate_sampling_rate(t_seg, method="median")
    t_uniform = generate_uniform_timestamps(t_seg, fs)
    rgb_uniform = resample_nd_signal(t_seg, rgb_seg, t_uniform, kind="linear")
    motion_uniform = resample_nd_signal(t_seg, motion_seg, t_uniform, kind="linear")
    brightness_uniform = resample_nd_signal(t_seg, brightness_seg, t_uniform, kind="linear")

    metadata = {
        "motion_mean": float(np.mean(motion_uniform)) if motion_uniform.size > 0 else None,
        "brightness_mean": float(np.mean(brightness_uniform)) if brightness_uniform.size > 0 else None,
    }

    return t_uniform, rgb_uniform, fs, metadata


def _run_algorithms_on_segment(
    rgb_uniform: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    min_duration_s: float,
    nperseg: Optional[int],
) -> Dict[str, Any]:
    """Run Green, CHROM, and POS on a uniformly sampled RGB signal from one segment."""
    green_signal = rgb_uniform[:, 1]  # G channel

    green_result = estimate_green_hr(
        green_signal, fs, low_hz=low_hz, high_hz=high_hz,
        min_duration_s=min_duration_s, nperseg=nperseg
    )

    chrom_result = estimate_chrom_hr(
        rgb_uniform, fs, low_hz=low_hz, high_hz=high_hz,
        min_duration_s=min_duration_s, nperseg=nperseg
    )

    pos_result = estimate_pos_hr(
        rgb_uniform, fs, low_hz=low_hz, high_hz=high_hz,
        min_duration_s=min_duration_s, nperseg=nperseg
    )

    return {
        "green": green_result,
        "chrom": chrom_result,
        "pos": pos_result,
    }


def _aggregate_region_results(
    region_results: Dict[str, Dict[str, Any]],
    min_valid_regions: int = 2,
) -> Tuple[Optional[float], List[str], List[str]]:
    """Aggregate valid region-level estimates using median.

    Returns:
        (aggregated_bpm, regions_used, regions_excluded)
    """
    valid_bpms = []
    regions_used = []
    regions_excluded = []

    for region, result in region_results.items():
        # Get valid BPMs from all three methods for this region
        region_bpms = []
        for method in ("green", "chrom", "pos"):
            method_result = result.get(method)
            if isinstance(method_result, Mapping) and method_result.get("valid") is True:
                bpm = method_result.get("bpm")
                if isinstance(bpm, (int, float, np.floating)) and np.isfinite(bpm):
                    region_bpms.append(float(bpm))

        if region_bpms:
            # Use median of valid methods for this region
            region_median = float(np.median(region_bpms))
            valid_bpms.append(region_median)
            regions_used.append(region)
        else:
            regions_excluded.append(region)

    if len(valid_bpms) < min_valid_regions:
        # Insufficient valid regions - return None and mark all as excluded
        return None, [], regions_used + regions_excluded

    aggregated_bpm = float(np.median(valid_bpms))
    return aggregated_bpm, regions_used, regions_excluded


def _compute_method_comparison(
    region_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute method comparison across all regions."""
    method_bpms = {"green": [], "chrom": [], "pos": []}
    method_valid = {"green": 0, "chrom": 0, "pos": 0}

    for region, result in region_results.items():
        for method in ("green", "chrom", "pos"):
            method_result = result.get(method)
            if isinstance(method_result, Mapping) and method_result.get("valid") is True:
                bpm = method_result.get("bpm")
                if isinstance(bpm, (int, float, np.floating)) and np.isfinite(bpm):
                    method_bpms[method].append(float(bpm))
                    method_valid[method] += 1

    comparison = {}
    for method in ("green", "chrom", "pos"):
        bpms = method_bpms[method]
        if bpms:
            comparison[method] = {
                "valid_count": method_valid[method],
                "mean_bpm": float(np.mean(bpms)),
                "median_bpm": float(np.median(bpms)),
                "std_bpm": float(np.std(bpms, ddof=1)) if len(bpms) > 1 else 0.0,
                "min_bpm": float(np.min(bpms)),
                "max_bpm": float(np.max(bpms)),
            }
        else:
            comparison[method] = {
                "valid_count": 0,
                "mean_bpm": None,
                "median_bpm": None,
                "std_bpm": None,
                "min_bpm": None,
                "max_bpm": None,
            }

    return comparison


def _compute_region_comparison(
    region_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute region comparison across all methods."""
    comparison = {}

    for region, result in region_results.items():
        region_bpms = []
        for method in ("green", "chrom", "pos"):
            method_result = result.get(method)
            if isinstance(method_result, Mapping) and method_result.get("valid") is True:
                bpm = method_result.get("bpm")
                if isinstance(bpm, (int, float, np.floating)) and np.isfinite(bpm):
                    region_bpms.append(float(bpm))

        if region_bpms:
            comparison[region] = {
                "valid_methods": len(region_bpms),
                "mean_bpm": float(np.mean(region_bpms)),
                "median_bpm": float(np.median(region_bpms)),
                "std_bpm": float(np.std(region_bpms, ddof=1)) if len(region_bpms) > 1 else 0.0,
                "min_bpm": float(np.min(region_bpms)),
                "max_bpm": float(np.max(region_bpms)),
            }
        else:
            comparison[region] = {
                "valid_methods": 0,
                "mean_bpm": None,
                "median_bpm": None,
                "std_bpm": None,
                "min_bpm": None,
                "max_bpm": None,
            }

    return comparison


def _compute_quality_metrics_for_segment(
    rgb_uniform: np.ndarray,
    fs: float,
    hr_frequency_hz: float,
) -> Dict[str, Any]:
    """Compute quality metrics for a segment using the quality_metrics module."""
    from scipy.signal import welch

    # Compute Welch PSD
    freqs, psd = welch(rgb_uniform[:, 1], fs=fs, window="hann",
                       nperseg=min(256, len(rgb_uniform)//2), noverlap=None)

    metrics = {}
    if np.isfinite(hr_frequency_hz):
        metrics["snr"] = calculate_snr(freqs, psd, hr_frequency_hz)
        metrics["spectral_concentration"] = calculate_spectral_concentration(freqs, psd, hr_frequency_hz)
        metrics["spectral_peak_prominence"] = calculate_spectral_peak_prominence(freqs, psd, hr_frequency_hz)
    else:
        metrics["snr"] = {"snr_db": None, "status": "hr_unavailable"}
        metrics["spectral_concentration"] = {"concentration": None, "status": "hr_unavailable"}
        metrics["spectral_peak_prominence"] = {"prominence": None, "status": "hr_unavailable"}

    return metrics


def _adaptive_region_selection(
    region_results: Dict[str, Dict[str, Any]],
    config: RegionConfig,
) -> Tuple[List[str], List[str]]:
    """Experimental adaptive region selection based on signal quality.

    Selects regions that have at least one valid method with acceptable quality metrics.

    Returns:
        (selected_regions, excluded_regions)
    """
    selected = []
    excluded = []

    for region, result in region_results.items():
        # Check if region has any valid method
        has_valid = False
        best_snr = -np.inf
        best_concentration = -np.inf
        best_prominence = -np.inf

        for method in ("green", "chrom", "pos"):
            method_result = result.get(method)
            if isinstance(method_result, Mapping) and method_result.get("valid") is True:
                has_valid = True
                # Get quality metrics from metadata if available
                metadata = result.get("metadata", {})
                quality = metadata.get("quality_metrics", {})
                if quality:
                    snr = quality.get("snr", {}).get("snr_db")
                    if snr is not None and np.isfinite(snr):
                        best_snr = max(best_snr, snr)
                    conc = quality.get("spectral_concentration", {}).get("concentration")
                    if conc is not None and np.isfinite(conc):
                        best_concentration = max(best_concentration, conc)
                    prom = quality.get("spectral_peak_prominence", {}).get("prominence")
                    if prom is not None and np.isfinite(prom):
                        best_prominence = max(best_prominence, prom)

        if not has_valid:
            excluded.append(region)
            continue

        # Check quality thresholds
        snr_ok = best_snr >= config.adaptive_min_snr_db if best_snr > -np.inf else True
        conc_ok = best_concentration >= config.adaptive_min_concentration if best_concentration > -np.inf else True
        prom_ok = best_prominence >= config.adaptive_min_prominence if best_prominence > -np.inf else True

        if snr_ok and conc_ok and prom_ok:
            selected.append(region)
        else:
            excluded.append(region)

    return selected, excluded


def _process_single_region(
    packets_or_data: Union[Sequence[Mapping[str, Any]], Mapping[str, Any]],
    region_name: str,
    low_hz: float,
    high_hz: float,
    min_duration_s: float,
    nperseg: Optional[int],
    is_packet_format: bool,
) -> Dict[str, Any]:
    """Process a single region and return results with metadata."""
    try:
        if is_packet_format:
            t, rgb, motion, brightness, track_ok, total_pkts, valid_count = _extract_region_data_from_packets(
                packets_or_data, region_name
            )
        else:
            t, rgb, motion, brightness, track_ok, total_pkts, valid_count = _extract_region_data_from_timeseries(
                packets_or_data, region_name
            )

        if valid_count == 0:
            result = _invalid_region_result(region_name, "no valid tracking samples for region")[region_name]
            result["metadata"].update({
                "valid_samples": 0,
                "missing_samples": total_pkts,
                "tracking_fraction": 0.0,
                "num_continuous_segments": 0,
                "longest_segment_duration_s": 0.0,
                "selected_segment_duration_s": 0.0,
            })
            return result

        # Get valid samples only
        valid_mask = track_ok
        t_valid = t[valid_mask]
        rgb_valid = rgb[valid_mask]
        motion_valid = motion[valid_mask]
        brightness_valid = brightness[valid_mask]

        # Split into continuous segments at tracking gaps
        segments = _split_into_continuous_segments(
            t_valid, rgb_valid, motion_valid, brightness_valid
        )

        # Find qualifying segments (duration >= min_duration_s)
        qualifying_segments = []
        for seg in segments:
            if seg["duration_s"] + _DURATION_TOLERANCE_S >= min_duration_s and seg["num_samples"] >= 2:
                try:
                    t_uniform, rgb_uniform, fs, resample_meta = _resample_segment(
                        seg["t"], seg["rgb"], seg["motion"], seg["brightness"]
                    )
                    # Double-check duration after resampling (with tolerance)
                    resampled_duration = float(t_uniform[-1] - t_uniform[0]) if t_uniform.size > 1 else 0.0
                    if resampled_duration + _DURATION_TOLERANCE_S >= min_duration_s:
                        qualifying_segments.append({
                            "segment": seg,
                            "t_uniform": t_uniform,
                            "rgb_uniform": rgb_uniform,
                            "fs": fs,
                            "resample_meta": resample_meta,
                            "resampled_duration_s": resampled_duration,
                        })
                except Exception:
                    # Skip segments that fail resampling
                    continue

        # Build base metadata
        num_segments = len(segments)
        longest_duration = max((s["duration_s"] for s in segments), default=0.0)
        gap_warning = None
        if num_segments > 1:
            gap_warning = f"tracking split into {num_segments} continuous segments (max gap factor {_GAP_THRESHOLD_FACTOR}x median dt)"

        base_metadata = {
            "valid_samples": valid_count,
            "missing_samples": total_pkts - valid_count,
            "tracking_fraction": valid_count / total_pkts if total_pkts > 0 else 0.0,
            "num_continuous_segments": num_segments,
            "longest_segment_duration_s": longest_duration,
            "selected_segment_duration_s": 0.0,
            "gap_warning": gap_warning,
        }

        if not qualifying_segments:
            reason = f"insufficient continuous signal due to tracking gaps: longest segment {longest_duration:.2f}s < {min_duration_s:.2f}s (tolerance {_DURATION_TOLERANCE_S:.2f}s)"
            result = _invalid_region_result(region_name, reason)[region_name]
            result["metadata"].update(base_metadata)
            return result

        # Select the longest qualifying segment
        best = max(qualifying_segments, key=lambda x: x["resampled_duration_s"])
        seg = best["segment"]
        t_uniform = best["t_uniform"]
        rgb_uniform = best["rgb_uniform"]
        fs = best["fs"]
        resample_meta = best["resample_meta"]
        selected_duration = best["resampled_duration_s"]

        # Run all three algorithms on the selected segment
        algo_results = _run_algorithms_on_segment(
            rgb_uniform, fs, low_hz, high_hz, min_duration_s, nperseg
        )

        # Compute quality metrics for the selected segment
        # Use the median HR frequency from valid methods
        valid_freqs = []
        for method in ("green", "chrom", "pos"):
            res = algo_results[method]
            if res.get("valid") and res.get("frequency_hz") is not None:
                valid_freqs.append(res["frequency_hz"])
        hr_freq = float(np.median(valid_freqs)) if valid_freqs else None

        quality_metrics = _compute_quality_metrics_for_segment(rgb_uniform, fs, hr_freq) if hr_freq else {}

        metadata = dict(base_metadata)
        metadata.update({
            "sampling_rate_hz": fs,
            "selected_segment_duration_s": selected_duration,
            "motion_mean": resample_meta.get("motion_mean"),
            "brightness_mean": resample_meta.get("brightness_mean"),
            "quality_metrics": quality_metrics,
        })

        # Add aggregated BPM for this region (median of valid methods)
        region_bpms = []
        for method in ("green", "chrom", "pos"):
            res = algo_results[method]
            if res.get("valid") and res.get("bpm") is not None:
                region_bpms.append(res["bpm"])
        aggregated_bpm = float(np.median(region_bpms)) if region_bpms else None

        return {
            "green": algo_results["green"],
            "chrom": algo_results["chrom"],
            "pos": algo_results["pos"],
            "aggregated_bpm": aggregated_bpm,
            "metadata": metadata,
        }

    except Exception as exc:
        # Unexpected errors should not be silently swallowed, but we isolate per region
        result = _invalid_region_result(region_name, f"processing error: {exc}")[region_name]
        result["metadata"].update({
            "valid_samples": 0,
            "missing_samples": len(packets_or_data) if hasattr(packets_or_data, '__len__') else 0,
            "tracking_fraction": 0.0,
            "num_continuous_segments": 0,
            "longest_segment_duration_s": 0.0,
            "selected_segment_duration_s": 0.0,
            "gap_warning": None,
        })
        return result


def estimate_multiregion_hr(
    packets: Sequence[Mapping[str, Any]],
    low_hz: float = 0.75,
    high_hz: float = 3.0,
    min_duration_s: float = 10.0,
    nperseg: Optional[int] = None,
) -> Dict[str, Any]:
    """Estimate heart rate from camera measurement packets for all facial regions.

    This is the legacy API for backward compatibility. Uses the three-region baseline.

    Processing per region:
    1. Extract valid (tracking=True, valid RGB) samples
    2. Split into continuous segments at gaps > 5x median dt
    3. For each segment >= min_duration_s: resample, run Green/CHROM/POS
    4. Select the longest qualifying segment
    5. Return results from that segment, or invalid if none qualify

    Args:
        packets: Iterable of camera measurement dictionaries (Member 1 format).
        low_hz: Lower HR-band cutoff in Hz (default 0.75).
        high_hz: Upper HR-band cutoff in Hz (default 3.0).
        min_duration_s: Minimum required signal duration in seconds (default 10.0).
        nperseg: Optional Welch segment length for spectral estimation.

    Returns:
        Dictionary with results for each region (forehead, left_cheek, right_cheek).
    """
    # Map legacy region names to new region names
    legacy_to_new = {
        "forehead": "forehead_center",
        "left_cheek": "left_cheek_upper",
        "right_cheek": "right_cheek_upper",
    }

    # Process using new API with baseline 3 regions
    config = RegionConfig(regions=BASELINE_3_REGIONS)
    agg_config = AggregationConfig(min_valid_regions=2)
    result = estimate_multiregion_hr_advanced(
        packets, config, agg_config,
        low_hz=low_hz, high_hz=high_hz, min_duration_s=min_duration_s, nperseg=nperseg
    )

    # Map back to legacy region names for backward compatibility
    legacy_result = {}
    for legacy, new in legacy_to_new.items():
        if new in result["region_results"]:
            legacy_result[legacy] = result["region_results"][new]
        else:
            legacy_result[legacy] = _invalid_region_result(legacy, "region not processed")[legacy]

    return legacy_result


def estimate_multiregion_hr_advanced(
    data: Union[Sequence[Mapping[str, Any]], Mapping[str, Any]],
    region_config: Optional[RegionConfig] = None,
    agg_config: Optional[AggregationConfig] = None,
    low_hz: float = 0.75,
    high_hz: float = 3.0,
    min_duration_s: float = 10.0,
    nperseg: Optional[int] = None,
) -> Dict[str, Any]:
    """Advanced multi-region heart-rate estimation with configurable regions.

    This is the main public API for Member 4 and research comparisons.

    Args:
        data: Either camera packets (Member 1 format) or time-series format dict.
        region_config: RegionConfig specifying which regions to process and adaptive settings.
        agg_config: AggregationConfig specifying aggregation method and minimum regions.
        low_hz: Lower HR-band cutoff in Hz (default 0.75).
        high_hz: Upper HR-band cutoff in Hz (default 3.0).
        min_duration_s: Minimum required signal duration in seconds (default 10.0).
        nperseg: Optional Welch segment length for spectral estimation.

    Returns:
        Structured dictionary with:
        - aggregated_bpm: float or None
        - aggregation_method: str
        - regions_requested: list of region names
        - regions_available: list of region names with valid data
        - regions_used: list of region names used in aggregation
        - region_results: dict with per-region Green/CHROM/POS/aggregated_bpm
        - method_comparison: dict with cross-region method statistics
        - region_comparison: dict with cross-method region statistics
        - metadata: dict with sampling_rate_hz, recording_duration_s, region counts
    """
    if region_config is None:
        region_config = RegionConfig()
    if agg_config is None:
        agg_config = AggregationConfig()

    # Detect input format
    is_packet_format = isinstance(data, Sequence) and not isinstance(data, Mapping)

    # Process each requested region
    region_results = {}
    regions_available = []

    for region in region_config.regions:
        result = _process_single_region(
            data, region, low_hz, high_hz, min_duration_s, nperseg, is_packet_format
        )
        region_results[region] = result

        # Check if region has any valid method
        has_valid = any(
            result.get(m, {}).get("valid") is True
            for m in ("green", "chrom", "pos")
        )
        if has_valid:
            regions_available.append(region)

    # Adaptive region selection (experimental)
    regions_used = regions_available
    regions_excluded = []
    adaptive_failure_reason = None
    if region_config.adaptive_mode:
        regions_used, regions_excluded = _adaptive_region_selection(region_results, region_config)
        # If adaptive selection leaves too few regions, do NOT fall back.
        # Instead, mark as adaptive failure and return None for aggregated_bpm.
        if len(regions_used) < agg_config.min_valid_regions:
            adaptive_failure_reason = (
                f"adaptive selection: only {len(regions_used)} region(s) passed quality thresholds, "
                f"need at least {agg_config.min_valid_regions}"
            )
            # Keep regions_used as the ones that passed, regions_excluded as those that failed
            # aggregated_bpm will be None due to insufficient regions

    # Aggregate using only selected regions
    selected_results = {r: region_results[r] for r in regions_used}
    aggregated_bpm, final_used, final_excluded = _aggregate_region_results(
        selected_results, agg_config.min_valid_regions
    )

    # Combine excluded regions
    all_excluded = list(set(regions_excluded + final_excluded + [r for r in region_config.regions if r not in regions_available]))

    # Compute comparisons
    method_comparison = _compute_method_comparison(region_results)
    region_comparison = _compute_region_comparison(region_results)

    # Compute overall metadata
    # Estimate sampling rate from first available region
    sampling_rate_hz = None
    recording_duration_s = 0.0
    for region in regions_available:
        meta = region_results[region].get("metadata", {})
        if meta.get("sampling_rate_hz") is not None:
            sampling_rate_hz = meta["sampling_rate_hz"]
            recording_duration_s = meta.get("selected_segment_duration_s", 0.0)
            break

    return {
        "aggregated_bpm": aggregated_bpm,
        "aggregation_method": agg_config.method,
        "regions_requested": list(region_config.regions),
        "regions_available": regions_available,
        "regions_used": regions_used,
        "regions_excluded": all_excluded,
        "region_results": region_results,
        "method_comparison": method_comparison,
        "region_comparison": region_comparison,
        "metadata": {
            "sampling_rate_hz": sampling_rate_hz,
            "recording_duration_s": recording_duration_s,
            "num_regions_requested": len(region_config.regions),
            "num_regions_available": len(regions_available),
            "num_regions_used": len(regions_used),
            "adaptive_mode": region_config.adaptive_mode,
            "adaptive_failure_reason": adaptive_failure_reason,
        },
    }


# Convenience functions for common configurations
def get_baseline_3_region_config() -> RegionConfig:
    """Get RegionConfig for the three-region research baseline."""
    return RegionConfig(regions=BASELINE_3_REGIONS, min_valid_regions=2)


def get_8_region_config() -> RegionConfig:
    """Get RegionConfig for all eight regions."""
    return RegionConfig(regions=ALL_REGIONS, min_valid_regions=2)


def get_adaptive_8_region_config(
    min_snr_db: float = 0.0,
    min_concentration: float = 0.3,
    min_prominence: float = 0.0,
) -> RegionConfig:
    """Get RegionConfig for adaptive eight-region selection (experimental)."""
    return RegionConfig(
        regions=ALL_REGIONS,
        min_valid_regions=2,
        adaptive_mode=True,
        adaptive_min_snr_db=min_snr_db,
        adaptive_min_concentration=min_concentration,
        adaptive_min_prominence=min_prominence,
    )


# Example usage:
# from signal_processing.multiregion import (
#     estimate_multiregion_hr_advanced,
#     get_baseline_3_region_config,
#     get_8_region_config,
#     get_adaptive_8_region_config,
#     RegionConfig,
#     AggregationConfig,
# )
#
# # Three-region baseline
# result = estimate_multiregion_hr_advanced(packets, get_baseline_3_region_config())
#
# # Eight-region fixed
# result = estimate_multiregion_hr_advanced(packets, get_8_region_config())
#
# # Custom region selection
# config = RegionConfig(regions=("forehead_center", "nose_bridge"), min_valid_regions=1)
# result = estimate_multiregion_hr_advanced(packets, config)
#
# # Adaptive eight-region (experimental)
# result = estimate_multiregion_hr_advanced(packets, get_adaptive_8_region_config())
#
# # Time-series format
# ts_data = {"t": timestamps, "rgb": {...}, "motion": ..., "track_ok": ..., "brightness": ...}
# result = estimate_multiregion_hr_advanced(ts_data, get_8_region_config())