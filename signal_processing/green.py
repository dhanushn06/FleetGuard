"""Green-channel rPPG heart-rate estimation.

This module implements a modular Green Method estimator for uniformly sampled
webcam rPPG signals. The pipeline is:

    1D green signal + fs
      -> input/range validation
      -> condition_signal(...)  (detrend + Butterworth SOS + sosfiltfilt)
      -> Welch PSD (scipy.signal.welch, Hann window)
      -> peak selection within configured HR band
      -> BPM estimate

Spectral estimation helpers are kept separate from Green-specific orchestration
so they can be reused by CHROM/POS implementations later.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy import signal

from .conditioning import DEFAULT_HIGHHZ, DEFAULT_LOWHZ, condition_signal

DEFAULT_MIN_DURATION_S = 10.0
DEFAULT_WELCH_SEGMENT_S = 8.0
_MIN_WELCH_NPERSEG = 64


def _invalid_result(reason: str, conditioned_signal: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Build a standardized invalid Green-method result.

    Purpose:
        Invalid/insufficient inputs must not produce a fabricated BPM. This
        helper returns the required failure structure with a descriptive reason.
    """
    result: Dict[str, Any] = {
        "method": "green",
        "bpm": None,
        "valid": False,
        "frequency_hz": None,
        "reason": str(reason),
    }
    if conditioned_signal is not None:
        result["conditioned_signal"] = conditioned_signal
    return result


def _validate_basic_inputs(
    green_signal: Any, fs: Any, low_hz: Any, high_hz: Any, min_duration_s: Any
) -> tuple[np.ndarray, float, float, float, float]:
    """Validate estimator-level inputs before conditioning/spectral analysis.

    Purpose:
        Enforce the public API contract and reject clearly invalid inputs with
        meaningful errors that can be converted into structured failure results.
    """
    x = np.asarray(green_signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"green_signal must be a 1D array, got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("green_signal must not contain NaN or infinite values")
    if x.size < 2:
        raise ValueError("green_signal must contain at least 2 samples")

    fs_val = float(fs)
    if not np.isfinite(fs_val) or fs_val <= 0:
        raise ValueError("fs must be a finite sampling rate > 0 Hz")

    low = float(low_hz)
    high = float(high_hz)
    if not (np.isfinite(low) and np.isfinite(high)):
        raise ValueError("low_hz and high_hz must be finite")
    if low <= 0 or low >= high:
        raise ValueError("frequency band must satisfy 0 < low_hz < high_hz")
    if high >= 0.5 * fs_val:
        raise ValueError(
            f"high_hz must be < Nyquist frequency ({0.5 * fs_val:.3f} Hz for fs={fs_val:.3f} Hz)"
        )

    min_dur = float(min_duration_s)
    if not np.isfinite(min_dur) or min_dur <= 0:
        raise ValueError("min_duration_s must be a finite value > 0")

    min_samples = int(np.ceil(min_dur * fs_val))
    if x.size < min_samples:
        raise ValueError(
            f"green_signal is too short: need at least {min_samples} samples "
            f"({min_dur:.2f} s at fs={fs_val:.3f} Hz), got {x.size}"
        )

    # Reject constant/near-constant signals early (conditioning also enforces).
    if np.all(x == x[0]) or float(np.std(x)) <= 1e-12:
        raise ValueError("green_signal must not be constant or degenerate")

    return x, fs_val, low, high, min_dur


def _resolve_welch_params(n_samples: int, fs: float, nperseg: Optional[int]) -> tuple[int, int]:
    """Choose and validate Welch segment/overlap parameters.

    Purpose:
        Balance spectral averaging and frequency resolution for short rPPG
        windows while ensuring parameters are valid for the available samples.

    Rationale:
        - Hann window is used (via scipy.signal.welch default window handling
          when window='hann') to reduce spectral leakage.
        - Default segment length targets ~8 s (capped by signal length) to
          provide reasonable HR-band resolution on ~10 s windows without
          collapsing to a single-segment periodogram unless necessary.
        - 50% overlap is used as a standard variance/resolution trade-off.
        - No nfft zero-padding is applied by default, so reported frequency bins
          reflect the Welch segment DFT grid (not increased true physiological
          resolution).

    Returns:
        (nperseg_resolved, noverlap)
    """
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2 for Welch PSD")

    if nperseg is None:
        target = int(np.round(fs * DEFAULT_WELCH_SEGMENT_S))
        seg = min(n_samples, max(_MIN_WELCH_NPERSEG, target))
    else:
        if isinstance(nperseg, bool) or not isinstance(nperseg, (int, np.integer)):
            raise ValueError("nperseg must be a positive integer or None")
        seg = int(nperseg)
        if seg < 1:
            raise ValueError("nperseg must be >= 1")

    if seg > n_samples:
        raise ValueError("nperseg must be <= signal length")
    if seg < 8:
        raise ValueError("nperseg is too small for a meaningful Welch estimate")

    noverlap = seg // 2
    if noverlap >= seg:
        noverlap = seg - 1
    if noverlap < 0:
        noverlap = 0
    return seg, noverlap


def compute_welch_psd(
    x: np.ndarray,
    fs: float,
    nperseg: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Compute Welch PSD for a conditioned 1D signal.

    Purpose:
        Provide a reusable spectral-density estimator (Green now, CHROM/POS
        later) with consistent windowing and parameter validation.

    Returns:
        (frequencies_hz, psd, nperseg_resolved, noverlap)
    """
    seg, noverlap = _resolve_welch_params(x.size, fs, nperseg)
    freqs, psd = signal.welch(
        x,
        fs=fs,
        window="hann",
        nperseg=seg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
    )
    return np.asarray(freqs, dtype=np.float64), np.asarray(psd, dtype=np.float64), seg, noverlap


def select_peak_in_band(
    freqs: np.ndarray, psd: np.ndarray, low_hz: float, high_hz: float
) -> tuple[float, int]:
    """Select the dominant spectral peak within the configured HR band.

    Purpose:
        Restrict heart-rate estimation to physiologically/configured bounds and
        return the maximum Welch PSD bin frequency (no sub-bin interpolation).

    Returns:
        (peak_frequency_hz, index_in_full_arrays)
    """
    if freqs.ndim != 1 or psd.ndim != 1 or freqs.shape != psd.shape:
        raise ValueError("freqs and psd must be 1D arrays of the same shape")
    if freqs.size == 0:
        raise ValueError("empty spectrum")

    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        raise ValueError(
            f"no frequency bins available in configured band [{low_hz:.3f}, {high_hz:.3f}] Hz"
        )

    band_indices = np.where(mask)[0]
    band_psd = psd[band_indices]
    if not np.all(np.isfinite(band_psd)):
        raise ValueError("PSD contains non-finite values in the target band")

    rel_idx = int(np.argmax(band_psd))
    peak_idx = int(band_indices[rel_idx])
    return float(freqs[peak_idx]), peak_idx


def estimate_green_hr(
    green_signal: Any,
    fs: Any,
    low_hz: float = DEFAULT_LOWHZ,
    high_hz: float = DEFAULT_HIGHHZ,
    min_duration_s: float = DEFAULT_MIN_DURATION_S,
    nperseg: Optional[int] = None,
) -> Dict[str, Any]:
    """Estimate heart rate from a 1D green-channel rPPG signal.

    Pipeline:
        validate -> condition_signal(...) -> Welch PSD (Hann, 50% overlap)
        -> band-limited peak selection -> BPM conversion.

    Args:
        green_signal: 1D green-channel signal (uniformly sampled).
        fs: Sampling rate in Hz.
        low_hz: Lower HR-band cutoff in Hz (default 0.75 Hz).
        high_hz: Upper HR-band cutoff in Hz (default 3.0 Hz).
        min_duration_s: Minimum required signal duration in seconds (default 10.0).
        nperseg: Optional Welch segment length. If None, a duration-based value
            (~8 s, capped by signal length, with a sensible minimum) is used.

    Returns:
        A structured dictionary. For valid estimates:
            {
              "method": "green",
              "bpm": float,
              "valid": True,
              "frequency_hz": float,
              "frequencies": np.ndarray,
              "psd": np.ndarray,
              "conditioned_signal": np.ndarray,
            }
        For invalid inputs/conditions (no fabricated BPM):
            {
              "method": "green",
              "bpm": None,
              "valid": False,
              "frequency_hz": None,
              "reason": str,
              ... (optional fields only if meaningful/available)
            }

    Notes:
        - The dominant frequency is the maximum Welch PSD bin within the
          configured band (no sub-bin interpolation). Frequency resolution is
          limited by segment length and sampling rate; zero-padding would not
          increase true physiological resolution.
        - Expected input/conditioning/spectral domain issues are reported as
          invalid results; unexpected programming errors are not swallowed.
    """
    conditioned: Optional[np.ndarray] = None

    # Stage 1: basic input/domain validation (expected failures -> invalid result)
    try:
        x, fs_val, low, high, min_dur = _validate_basic_inputs(
            green_signal, fs, low_hz, high_hz, min_duration_s
        )
    except ValueError as exc:
        return _invalid_result(str(exc))

    # Stage 2: conditioning (reuses existing pipeline; expected failures -> invalid result)
    try:
        conditioned = condition_signal(
            x,
            fs_val,
            low_hz=low,
            high_hz=high,
            order=4,
            detrend_method="linear",
            min_duration_s=min_dur,
        )
    except ValueError as exc:
        return _invalid_result(str(exc), conditioned_signal=conditioned)

    # Stage 3: Welch PSD and peak selection
    try:
        freqs, psd, _, _ = compute_welch_psd(conditioned, fs_val, nperseg=nperseg)
        peak_hz, _ = select_peak_in_band(freqs, psd, low, high)
    except ValueError as exc:
        result = _invalid_result(str(exc), conditioned_signal=conditioned)
        # Include spectral arrays if they were computed before peak-selection failure.
        if "freqs" in locals() and "psd" in locals():
            result["frequencies"] = freqs
            result["psd"] = psd
        return result

    bpm = float(peak_hz * 60.0)
    return {
        "method": "green",
        "bpm": bpm,
        "valid": True,
        "frequency_hz": peak_hz,
        "frequencies": freqs,
        "psd": psd,
        "conditioned_signal": conditioned,
    }
