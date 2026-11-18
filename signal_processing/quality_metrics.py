"""Signal quality metrics for rPPG heart-rate estimation.

This module provides quantitative quality metrics for estimated heart-rate signals
and associated metadata. It does NOT make reliability decisions or produce a
combined quality score. Each metric is returned independently with clear status.

All functions are designed to be safe: they validate inputs, handle edge cases
(NaN, Inf, zero power, flat spectra), and return structured results with
explicit status/reason fields instead of raising exceptions for expected
invalid/insufficient data conditions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np


def _as_finite_array(x: Any, name: str = "x") -> np.ndarray:
    """Convert input to 1D float64 array and validate finiteness."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must not contain NaN or infinite values")
    return arr


def _safe_divide(num: float, den: float, default: float = 0.0) -> float:
    """Safe division with default for zero/non-finite denominator."""
    if not np.isfinite(den) or den == 0.0:
        return default
    return num / den


def calculate_snr(
    frequencies: Union[np.ndarray, Sequence[float]],
    psd: Union[np.ndarray, Sequence[float]],
    hr_frequency_hz: float,
    signal_window_hz: float = 0.1,
    hr_band: tuple[float, float] = (0.75, 3.0),
) -> Dict[str, Any]:
    """Calculate Signal-to-Noise Ratio (SNR) in dB from Welch PSD.

    Signal power: integrated PSD within ±signal_window_hz of hr_frequency_hz.
    Noise power: integrated PSD in the HR band excluding the signal window.

    Args:
        frequencies: Frequency array from Welch PSD (Hz).
        psd: Power spectral density array (same shape as frequencies).
        hr_frequency_hz: Detected heart-rate frequency in Hz.
        signal_window_hz: Half-width of signal window around HR peak (default 0.1 Hz).
        hr_band: (low_hz, high_hz) defining the allowed HR band for noise calculation.

    Returns:
        Dictionary with keys:
        - "snr_db": float or None (SNR in dB, None if undefined)
        - "signal_power": float
        - "noise_power": float
        - "status": "ok" | "zero_noise_power" | "zero_total_power" | "invalid_input"
        - "reason": str (explanation if not "ok")
    """
    try:
        freqs = _as_finite_array(frequencies, "frequencies")
        psd_arr = _as_finite_array(psd, "psd")
    except ValueError as exc:
        return {
            "snr_db": None,
            "signal_power": 0.0,
            "noise_power": 0.0,
            "status": "invalid_input",
            "reason": str(exc),
        }

    if freqs.shape != psd_arr.shape:
        return {
            "snr_db": None,
            "signal_power": 0.0,
            "noise_power": 0.0,
            "status": "invalid_input",
            "reason": "frequencies and psd must have the same shape",
        }

    if not np.isfinite(hr_frequency_hz):
        return {
            "snr_db": None,
            "signal_power": 0.0,
            "noise_power": 0.0,
            "status": "invalid_input",
            "reason": "hr_frequency_hz must be finite",
        }

    low_hz, high_hz = hr_band
    if not (low_hz < hr_frequency_hz < high_hz):
        return {
            "snr_db": None,
            "signal_power": 0.0,
            "noise_power": 0.0,
            "status": "invalid_input",
            "reason": f"hr_frequency_hz ({hr_frequency_hz:.3f}) outside HR band [{low_hz}, {high_hz}]",
        }

    # Signal window mask
    signal_mask = (freqs >= hr_frequency_hz - signal_window_hz) & (freqs <= hr_frequency_hz + signal_window_hz)
    # HR band mask (for noise calculation)
    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    # Noise mask: in HR band but outside signal window
    noise_mask = band_mask & ~signal_mask

    signal_power = float(np.trapezoid(psd_arr[signal_mask], freqs[signal_mask])) if np.any(signal_mask) else 0.0
    noise_power = float(np.trapezoid(psd_arr[noise_mask], freqs[noise_mask])) if np.any(noise_mask) else 0.0

    if signal_power == 0.0 and noise_power == 0.0:
        return {
            "snr_db": None,
            "signal_power": 0.0,
            "noise_power": 0.0,
            "status": "zero_total_power",
            "reason": "both signal and noise power are zero",
        }

    if noise_power == 0.0:
        return {
            "snr_db": None,
            "signal_power": signal_power,
            "noise_power": 0.0,
            "status": "zero_noise_power",
            "reason": "noise power is zero (infinite SNR)",
        }

    snr_db = 10.0 * np.log10(signal_power / noise_power)
    return {
        "snr_db": float(snr_db),
        "signal_power": signal_power,
        "noise_power": noise_power,
        "status": "ok",
        "reason": "",
    }


def calculate_spectral_peak_prominence(
    frequencies: Union[np.ndarray, Sequence[float]],
    psd: Union[np.ndarray, Sequence[float]],
    hr_frequency_hz: float,
    prominence_window_hz: float = 0.5,
) -> Dict[str, Any]:
    """Calculate spectral peak prominence of the HR peak.

    Prominence is defined as the peak PSD value minus the maximum of the
    surrounding baseline within ±prominence_window_hz (excluding the peak itself).

    Args:
        frequencies: Frequency array from Welch PSD (Hz).
        psd: Power spectral density array.
        hr_frequency_hz: Detected heart-rate frequency in Hz.
        prominence_window_hz: Half-width of window around peak for baseline estimation.

    Returns:
        Dictionary with keys:
        - "prominence": float or None
        - "peak_psd": float
        - "baseline_psd": float
        - "status": "ok" | "insufficient_data" | "invalid_input"
        - "reason": str
    """
    try:
        freqs = _as_finite_array(frequencies, "frequencies")
        psd_arr = _as_finite_array(psd, "psd")
    except ValueError as exc:
        return {
            "prominence": None,
            "peak_psd": 0.0,
            "baseline_psd": 0.0,
            "status": "invalid_input",
            "reason": str(exc),
        }

    if freqs.shape != psd_arr.shape:
        return {
            "prominence": None,
            "peak_psd": 0.0,
            "baseline_psd": 0.0,
            "status": "invalid_input",
            "reason": "frequencies and psd must have the same shape",
        }

    if not np.isfinite(hr_frequency_hz):
        return {
            "prominence": None,
            "peak_psd": 0.0,
            "baseline_psd": 0.0,
            "status": "invalid_input",
            "reason": "hr_frequency_hz must be finite",
        }

    # Find peak index (closest frequency bin to hr_frequency_hz)
    peak_idx = int(np.argmin(np.abs(freqs - hr_frequency_hz)))
    peak_psd = float(psd_arr[peak_idx])

    # Window around peak for baseline estimation (excluding peak itself)
    window_mask = (freqs >= hr_frequency_hz - prominence_window_hz) & (freqs <= hr_frequency_hz + prominence_window_hz)
    window_mask[peak_idx] = False  # exclude peak

    if not np.any(window_mask):
        return {
            "prominence": None,
            "peak_psd": peak_psd,
            "baseline_psd": 0.0,
            "status": "insufficient_data",
            "reason": "no baseline samples in prominence window",
        }

    baseline_psd = float(np.max(psd_arr[window_mask]))
    prominence = peak_psd - baseline_psd

    return {
        "prominence": float(prominence),
        "peak_psd": peak_psd,
        "baseline_psd": baseline_psd,
        "status": "ok",
        "reason": "",
    }


def calculate_spectral_concentration(
    frequencies: Union[np.ndarray, Sequence[float]],
    psd: Union[np.ndarray, Sequence[float]],
    hr_frequency_hz: float,
    concentration_window_hz: float = 0.2,
    hr_band: tuple[float, float] = (0.75, 3.0),
) -> Dict[str, Any]:
    """Calculate spectral concentration (fraction of HR-band power in peak neighborhood).

    Concentration = power in ±concentration_window_hz of HR peak / total power in HR band.

    Args:
        frequencies: Frequency array from Welch PSD (Hz).
        psd: Power spectral density array.
        hr_frequency_hz: Detected heart-rate frequency in Hz.
        concentration_window_hz: Half-width of peak neighborhood.
        hr_band: (low_hz, high_hz) defining the allowed HR band.

    Returns:
        Dictionary with keys:
        - "concentration": float in [0, 1] or None
        - "peak_power": float
        - "total_band_power": float
        - "status": "ok" | "zero_total_power" | "invalid_input"
        - "reason": str
    """
    try:
        freqs = _as_finite_array(frequencies, "frequencies")
        psd_arr = _as_finite_array(psd, "psd")
    except ValueError as exc:
        return {
            "concentration": None,
            "peak_power": 0.0,
            "total_band_power": 0.0,
            "status": "invalid_input",
            "reason": str(exc),
        }

    if freqs.shape != psd_arr.shape:
        return {
            "concentration": None,
            "peak_power": 0.0,
            "total_band_power": 0.0,
            "status": "invalid_input",
            "reason": "frequencies and psd must have the same shape",
        }

    if not np.isfinite(hr_frequency_hz):
        return {
            "concentration": None,
            "peak_power": 0.0,
            "total_band_power": 0.0,
            "status": "invalid_input",
            "reason": "hr_frequency_hz must be finite",
        }

    low_hz, high_hz = hr_band
    if not (low_hz < hr_frequency_hz < high_hz):
        return {
            "concentration": None,
            "peak_power": 0.0,
            "total_band_power": 0.0,
            "status": "invalid_input",
            "reason": f"hr_frequency_hz ({hr_frequency_hz:.3f}) outside HR band [{low_hz}, {high_hz}]",
        }

    # Peak neighborhood mask
    peak_mask = (freqs >= hr_frequency_hz - concentration_window_hz) & (freqs <= hr_frequency_hz + concentration_window_hz)
    # HR band mask
    band_mask = (freqs >= low_hz) & (freqs <= high_hz)

    peak_power = float(np.trapezoid(psd_arr[peak_mask], freqs[peak_mask])) if np.any(peak_mask) else 0.0
    total_band_power = float(np.trapezoid(psd_arr[band_mask], freqs[band_mask])) if np.any(band_mask) else 0.0

    if total_band_power == 0.0:
        return {
            "concentration": None,
            "peak_power": peak_power,
            "total_band_power": 0.0,
            "status": "zero_total_power",
            "reason": "total power in HR band is zero",
        }

    concentration = _safe_divide(peak_power, total_band_power, default=0.0)
    # Clamp to [0, 1] for numerical safety
    concentration = float(np.clip(concentration, 0.0, 1.0))

    return {
        "concentration": concentration,
        "peak_power": peak_power,
        "total_band_power": total_band_power,
        "status": "ok",
        "reason": "",
    }


def calculate_method_agreement(
    method_results: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Calculate agreement metrics across Green, CHROM, POS HR estimates.

    Args:
        method_results: Dict mapping method name ("green", "chrom", "pos") to
            result dict with keys "valid" (bool), "bpm" (float or None).

    Returns:
        Dictionary with keys:
        - "num_valid": int
        - "mean_bpm": float or None
        - "std_bpm": float or None
        - "range_bpm": float or None
        - "valid_methods": list of method names
        - "status": "ok" | "insufficient_valid_methods" | "invalid_input"
        - "reason": str
    """
    if not isinstance(method_results, Mapping):
        return {
            "num_valid": 0,
            "mean_bpm": None,
            "std_bpm": None,
            "range_bpm": None,
            "valid_methods": [],
            "status": "invalid_input",
            "reason": "method_results must be a mapping",
        }

    valid_bpms = []
    valid_methods = []

    for method, result in method_results.items():
        if not isinstance(result, Mapping):
            return {
                "num_valid": 0,
                "mean_bpm": None,
                "std_bpm": None,
                "range_bpm": None,
                "valid_methods": [],
                "status": "invalid_input",
                "reason": f"method result for '{method}' must be a mapping",
            }
        if result.get("valid") is True:
            bpm = result.get("bpm")
            if isinstance(bpm, (int, float, np.floating)) and np.isfinite(bpm):
                valid_bpms.append(float(bpm))
                valid_methods.append(str(method))

    num_valid = len(valid_bpms)

    if num_valid < 2:
        return {
            "num_valid": num_valid,
            "mean_bpm": float(valid_bpms[0]) if num_valid == 1 else None,
            "std_bpm": None,
            "range_bpm": None,
            "valid_methods": valid_methods,
            "status": "insufficient_valid_methods",
            "reason": f"need at least 2 valid methods for agreement, got {num_valid}",
        }

    bpms = np.array(valid_bpms, dtype=np.float64)
    mean_bpm = float(np.mean(bpms))
    std_bpm = float(np.std(bpms, ddof=1))
    range_bpm = float(np.max(bpms) - np.min(bpms))

    return {
        "num_valid": num_valid,
        "mean_bpm": mean_bpm,
        "std_bpm": std_bpm,
        "range_bpm": range_bpm,
        "valid_methods": valid_methods,
        "status": "ok",
        "reason": "",
    }


def calculate_roi_agreement(
    region_results: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Calculate agreement metrics across facial regions.

    For each region, uses the mean BPM across valid methods (Green/CHROM/POS).
    Then calculates agreement across region means.

    Args:
        region_results: Dict mapping region name ("forehead", "left_cheek", "right_cheek")
            to result dict with keys "green", "chrom", "pos" (each a method result dict).

    Returns:
        Dictionary with keys:
        - "num_valid_regions": int
        - "mean_bpm": float or None
        - "std_bpm": float or None
        - "range_bpm": float or None
        - "region_means": dict mapping region -> mean BPM
        - "status": "ok" | "insufficient_valid_regions" | "invalid_input"
        - "reason": str
    """
    if not isinstance(region_results, Mapping):
        return {
            "num_valid_regions": 0,
            "mean_bpm": None,
            "std_bpm": None,
            "range_bpm": None,
            "region_means": {},
            "status": "invalid_input",
            "reason": "region_results must be a mapping",
        }

    region_means = {}
    valid_region_bpms = []

    for region, result in region_results.items():
        if not isinstance(result, Mapping):
            return {
                "num_valid_regions": 0,
                "mean_bpm": None,
                "std_bpm": None,
                "range_bpm": None,
                "region_means": {},
                "status": "invalid_input",
                "reason": f"region result for '{region}' must be a mapping",
            }

        # Collect valid BPMs from all methods for this region
        region_bpms = []
        for method in ("green", "chrom", "pos"):
            method_result = result.get(method)
            if isinstance(method_result, Mapping) and method_result.get("valid") is True:
                bpm = method_result.get("bpm")
                if isinstance(bpm, (int, float, np.floating)) and np.isfinite(bpm):
                    region_bpms.append(float(bpm))

        if region_bpms:
            region_mean = float(np.mean(region_bpms))
            region_means[str(region)] = region_mean
            valid_region_bpms.append(region_mean)

    num_valid = len(valid_region_bpms)

    if num_valid < 2:
        return {
            "num_valid_regions": num_valid,
            "mean_bpm": float(valid_region_bpms[0]) if num_valid == 1 else None,
            "std_bpm": None,
            "range_bpm": None,
            "region_means": region_means,
            "status": "insufficient_valid_regions",
            "reason": f"need at least 2 valid regions for agreement, got {num_valid}",
        }

    bpms = np.array(valid_region_bpms, dtype=np.float64)
    mean_bpm = float(np.mean(bpms))
    std_bpm = float(np.std(bpms, ddof=1))
    range_bpm = float(np.max(bpms) - np.min(bpms))

    return {
        "num_valid_regions": num_valid,
        "mean_bpm": mean_bpm,
        "std_bpm": std_bpm,
        "range_bpm": range_bpm,
        "region_means": region_means,
        "status": "ok",
        "reason": "",
    }


def calculate_motion_summary(
    motion: Union[np.ndarray, Sequence[float], None],
) -> Dict[str, Any]:
    """Calculate summary statistics for motion measurements.

    Args:
        motion: 1D array of motion displacement values (may contain NaN/Inf/None).

    Returns:
        Dictionary with keys:
        - "mean": float or None
        - "median": float or None
        - "max": float or None
        - "num_valid": int
        - "num_total": int
        - "status": "ok" | "no_valid_data" | "invalid_input"
        - "reason": str
    """
    if motion is None:
        return {
            "mean": None,
            "median": None,
            "max": None,
            "num_valid": 0,
            "num_total": 0,
            "status": "no_valid_data",
            "reason": "motion input is None",
        }

    try:
        arr = np.asarray(motion, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        return {
            "mean": None,
            "median": None,
            "max": None,
            "num_valid": 0,
            "num_total": 0,
            "status": "invalid_input",
            "reason": f"motion must be array-like: {exc}",
        }

    if arr.ndim != 1:
        return {
            "mean": None,
            "median": None,
            "max": None,
            "num_valid": 0,
            "num_total": arr.size,
            "status": "invalid_input",
            "reason": f"motion must be 1D, got shape {arr.shape}",
        }

    num_total = arr.size
    finite_mask = np.isfinite(arr)
    num_valid = int(np.sum(finite_mask))

    if num_valid == 0:
        return {
            "mean": None,
            "median": None,
            "max": None,
            "num_valid": 0,
            "num_total": num_total,
            "status": "no_valid_data",
            "reason": "no finite motion values",
        }

    valid_arr = arr[finite_mask]
    return {
        "mean": float(np.mean(valid_arr)),
        "median": float(np.median(valid_arr)),
        "max": float(np.max(valid_arr)),
        "num_valid": num_valid,
        "num_total": num_total,
        "status": "ok",
        "reason": "",
    }


def calculate_tracking_summary(
    track_ok: Union[np.ndarray, Sequence[bool], Sequence[int], None],
) -> Dict[str, Any]:
    """Calculate tracking status summary.

    Args:
        track_ok: 1D array of boolean or integer tracking status (True/1 = tracked).

    Returns:
        Dictionary with keys:
        - "tracking_fraction": float in [0, 1]
        - "total_samples": int
        - "tracked_samples": int
        - "lost_samples": int
        - "status": "ok" | "invalid_input"
        - "reason": str
    """
    if track_ok is None:
        return {
            "tracking_fraction": 0.0,
            "total_samples": 0,
            "tracked_samples": 0,
            "lost_samples": 0,
            "status": "invalid_input",
            "reason": "track_ok input is None",
        }

    try:
        arr = np.asarray(track_ok)
    except (ValueError, TypeError) as exc:
        return {
            "tracking_fraction": 0.0,
            "total_samples": 0,
            "tracked_samples": 0,
            "lost_samples": 0,
            "status": "invalid_input",
            "reason": f"track_ok must be array-like: {exc}",
        }

    if arr.ndim != 1:
        return {
            "tracking_fraction": 0.0,
            "total_samples": 0,
            "tracked_samples": 0,
            "lost_samples": 0,
            "status": "invalid_input",
            "reason": f"track_ok must be 1D, got shape {arr.shape}",
        }

    total = arr.size
    if total == 0:
        return {
            "tracking_fraction": 0.0,
            "total_samples": 0,
            "tracked_samples": 0,
            "lost_samples": 0,
            "status": "ok",
            "reason": "",
        }

    # Convert to boolean (True/1/1.0 -> True, False/0/0.0 -> False)
    tracked = np.sum(arr.astype(bool))
    lost = total - int(tracked)
    fraction = _safe_divide(float(tracked), float(total), default=0.0)

    return {
        "tracking_fraction": fraction,
        "total_samples": total,
        "tracked_samples": int(tracked),
        "lost_samples": lost,
        "status": "ok",
        "reason": "",
    }


def calculate_brightness_summary(
    brightness: Union[np.ndarray, Sequence[float], None],
) -> Dict[str, Any]:
    """Calculate summary statistics for brightness measurements.

    Args:
        brightness: 1D array of brightness values (may contain NaN/Inf/None).

    Returns:
        Dictionary with keys:
        - "mean": float or None
        - "median": float or None
        - "min": float or None
        - "max": float or None
        - "num_valid": int
        - "num_total": int
        - "status": "ok" | "no_valid_data" | "invalid_input"
        - "reason": str
    """
    if brightness is None:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "num_valid": 0,
            "num_total": 0,
            "status": "no_valid_data",
            "reason": "brightness input is None",
        }

    try:
        arr = np.asarray(brightness, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "num_valid": 0,
            "num_total": 0,
            "status": "invalid_input",
            "reason": f"brightness must be array-like: {exc}",
        }

    if arr.ndim != 1:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "num_valid": 0,
            "num_total": arr.size,
            "status": "invalid_input",
            "reason": f"brightness must be 1D, got shape {arr.shape}",
        }

    num_total = arr.size
    finite_mask = np.isfinite(arr)
    num_valid = int(np.sum(finite_mask))

    if num_valid == 0:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "num_valid": 0,
            "num_total": num_total,
            "status": "no_valid_data",
            "reason": "no finite brightness values",
        }

    valid_arr = arr[finite_mask]
    return {
        "mean": float(np.mean(valid_arr)),
        "median": float(np.median(valid_arr)),
        "min": float(np.min(valid_arr)),
        "max": float(np.max(valid_arr)),
        "num_valid": num_valid,
        "num_total": num_total,
        "status": "ok",
        "reason": "",
    }