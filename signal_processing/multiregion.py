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
            "forehead": {
                "rgb": {"r": float, "g": float, "b": float},
                "brightness": float,
                "pixel_count": int
            },
            "left_cheek": { ... },
            "right_cheek": { ... }
        }
    }
When tracking fails, "regions" may be None and other fields may be None.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .chrom import estimate_chrom_hr
from .green import estimate_green_hr
from .pos import estimate_pos_hr
from .sampling import (
    estimate_sampling_rate,
    generate_uniform_timestamps,
    resample_nd_signal,
)

REQUIRED_REGIONS = ("forehead", "left_cheek", "right_cheek")
# Gap threshold: 5x median dt. Tolerates normal webcam jitter (typically < 2-3x)
# while catching genuine tracking loss (seconds of missing frames).
# Matches the warning threshold used in the original implementation.
_GAP_THRESHOLD_FACTOR = 5.0
# Small tolerance for duration checks after resampling to account for
# floating-point edge effects in uniform grid generation.
_DURATION_TOLERANCE_S = 0.1


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


def _extract_region_data(
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
    """Extract per-region time series from camera packets.

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


def estimate_multiregion_hr(
    packets: Sequence[Mapping[str, Any]],
    low_hz: float = 0.75,
    high_hz: float = 3.0,
    min_duration_s: float = 10.0,
    nperseg: Optional[int] = None,
) -> Dict[str, Any]:
    """Estimate heart rate from camera measurement packets for all three facial regions.

    Processing per region:
    1. Extract valid (tracking=True, valid RGB) samples
    2. Split into continuous segments at gaps > 3x median dt
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
        Dictionary with results for each region.
    """
    if not packets:
        return {
            region: _invalid_region_result(region, "empty packet list")[region]
            for region in REQUIRED_REGIONS
        }

    results = {}

    for region in REQUIRED_REGIONS:
        try:
            # Extract raw per-region data from packets
            t, rgb, motion, brightness, track_ok, total_pkts, valid_count = _extract_region_data(packets, region)

            if valid_count == 0:
                results[region] = _invalid_region_result(region, "no valid tracking samples for region")[region]
                results[region]["metadata"].update({
                    "valid_samples": 0,
                    "missing_samples": total_pkts,
                    "tracking_fraction": 0.0,
                    "num_continuous_segments": 0,
                    "longest_segment_duration_s": 0.0,
                    "selected_segment_duration_s": 0.0,
                })
                continue

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
                results[region] = _invalid_region_result(region, reason)[region]
                results[region]["metadata"].update(base_metadata)
                continue

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

            metadata = dict(base_metadata)
            metadata.update({
                "sampling_rate_hz": fs,
                "selected_segment_duration_s": selected_duration,
                "motion_mean": resample_meta.get("motion_mean"),
                "brightness_mean": resample_meta.get("brightness_mean"),
            })

            results[region] = {
                "green": algo_results["green"],
                "chrom": algo_results["chrom"],
                "pos": algo_results["pos"],
                "metadata": metadata,
            }

        except Exception as exc:
            # Unexpected errors should not be silently swallowed, but we isolate per region
            results[region] = _invalid_region_result(region, f"processing error: {exc}")[region]
            results[region]["metadata"].update({
                "valid_samples": 0,
                "missing_samples": len(packets),
                "tracking_fraction": 0.0,
                "num_continuous_segments": 0,
                "longest_segment_duration_s": 0.0,
                "selected_segment_duration_s": 0.0,
                "gap_warning": None,
            })

    return results