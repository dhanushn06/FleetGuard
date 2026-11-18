"""CHROM (Chrominance-based) rPPG heart-rate estimation.

This module implements a single-ROI CHROM estimator for uniformly sampled RGB
signals with shape ``(N, 3)`` in channel order ``[R, G, B]``.

Pipeline:
    1. Validate RGB input and parameters
    2. Temporally normalize each channel by its mean
    3. Build CHROM projections (X, Y)
    4. Condition X and Y with ``condition_signal`` (detrend + bandpass)
    5. Compute adaptive scaling ``alpha = std(Xf) / std(Yf)``
    6. Combine ``S = Xf - alpha * Yf``
    7. Estimate HR from Welch PSD peak in the configured band

Spectral helpers and conditioning are reused from existing modules to avoid
duplication and keep CHROM/POS implementations modular.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .conditioning import DEFAULT_HIGHHZ, DEFAULT_LOWHZ, condition_signal
from .green import compute_welch_psd, select_peak_in_band

DEFAULT_MIN_DURATION_S = 10.0
_EPS = 1e-12
_MEAN_EPS = 1e-12


def _invalid_result(reason: str, conditioned_signal: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Build a standardized invalid CHROM result.

    Purpose:
        Expected input/domain failures must not produce a fabricated BPM.
    """
    result: Dict[str, Any] = {
        "method": "chrom",
        "bpm": None,
        "valid": False,
        "frequency_hz": None,
        "reason": str(reason),
    }
    if conditioned_signal is not None:
        result["conditioned_signal"] = conditioned_signal
    return result


def _validate_rgb_inputs(
    rgb_signal: Any,
    fs: Any,
    low_hz: Any,
    high_hz: Any,
    min_duration_s: Any,
) -> Tuple[np.ndarray, float, float, float, float]:
    """Validate CHROM estimator inputs and basic RGB signal integrity.

    Purpose:
        Enforce API contract and reject invalid inputs with meaningful errors
        suitable for conversion to structured failure results.

    Returns:
        (rgb_array[N,3], fs, low_hz, high_hz, min_duration_s)
    """
    rgb = np.asarray(rgb_signal, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"rgb_signal must be a 2D array of shape (N, 3), got shape {rgb.shape}")
    if not np.all(np.isfinite(rgb)):
        raise ValueError("rgb_signal must not contain NaN or infinite values")
    if rgb.shape[0] < 2:
        raise ValueError("rgb_signal must contain at least 2 samples")

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
    if rgb.shape[0] < min_samples:
        raise ValueError(
            f"rgb_signal is too short: need at least {min_samples} samples "
            f"({min_dur:.2f} s at fs={fs_val:.3f} Hz), got {rgb.shape[0]}"
        )

    # Channel means must be valid for temporal normalization.
    means = np.mean(rgb, axis=0)
    if not np.all(np.isfinite(means)):
        raise ValueError("RGB channel means must be finite")
    if np.any(np.abs(means) <= _MEAN_EPS):
        raise ValueError("RGB channel means must be non-zero (invalid for normalization)")

    # Channels must not be degenerate (constant/near-constant), as they cannot
    # carry a reliable pulsatile chrominance signal.
    stds = np.std(rgb, axis=0)
    if not np.all(np.isfinite(stds)):
        raise ValueError("RGB channel standard deviations must be finite")
    if np.any(stds <= _EPS):
        raise ValueError("RGB channels must not be constant or degenerate")

    return rgb, fs_val, low, high, min_dur


def _normalize_by_mean(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize R,G,B channels by their temporal means.

    Purpose:
        CHROM operates on mean-normalized color traces to reduce dependency on
        absolute illumination/intensity level.
    """
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]
    mean_r = float(np.mean(r))
    mean_g = float(np.mean(g))
    mean_b = float(np.mean(b))

    if not np.all(np.isfinite([mean_r, mean_g, mean_b])):
        raise ValueError("RGB channel means must be finite")
    if abs(mean_r) <= _MEAN_EPS or abs(mean_g) <= _MEAN_EPS or abs(mean_b) <= _MEAN_EPS:
        raise ValueError("RGB channel means must be non-zero (invalid for normalization)")

    rn = r / mean_r
    gn = g / mean_g
    bn = b / mean_b
    if not (np.all(np.isfinite(rn)) and np.all(np.isfinite(gn)) and np.all(np.isfinite(bn))):
        raise ValueError("normalized RGB channels contain non-finite values")
    return rn, gn, bn


def estimate_chrom_hr(
    rgb_signal: Any,
    fs: Any,
    low_hz: float = DEFAULT_LOWHZ,
    high_hz: float = DEFAULT_HIGHHZ,
    min_duration_s: float = DEFAULT_MIN_DURATION_S,
    nperseg: Optional[int] = None,
) -> Dict[str, Any]:
    """Estimate heart rate from a single uniformly sampled RGB ROI using CHROM.

    Args:
        rgb_signal: RGB signal array of shape ``(N, 3)`` with columns [R, G, B].
        fs: Sampling rate in Hz.
        low_hz: Lower HR-band cutoff in Hz (default 0.75 Hz).
        high_hz: Upper HR-band cutoff in Hz (default 3.0 Hz).
        min_duration_s: Minimum required duration in seconds (default 10.0 s).
        nperseg: Optional Welch segment length passed to spectral estimation.

    Returns:
        Structured result dictionary.
        Valid:
            {
              "method": "chrom",
              "bpm": float,
              "valid": True,
              "frequency_hz": float,
              "frequencies": np.ndarray,
              "psd": np.ndarray,
              "conditioned_signal": np.ndarray,  # final CHROM signal S
              "alpha": float,
            }
        Invalid (expected input/domain errors, no fabricated BPM):
            {
              "method": "chrom",
              "bpm": None,
              "valid": False,
              "frequency_hz": None,
              "reason": str,
              ... (optional fields only if meaningful/available)
            }

    Notes:
        - Conditioning (detrend + Butterworth SOS + zero-phase) is applied to
          CHROM projections X and Y via ``condition_signal``.
        - Spectral peak selection uses Welch PSD maximum bin within the
          configured band (no sub-bin interpolation).
        - Unexpected programming errors are not silently suppressed.
    """
    # Stage 1: validate inputs
    try:
        rgb, fs_val, low, high, min_dur = _validate_rgb_inputs(
            rgb_signal, fs, low_hz, high_hz, min_duration_s
        )
    except ValueError as exc:
        return _invalid_result(str(exc))

    # Stage 2: normalize and construct CHROM projections
    try:
        rn, gn, bn = _normalize_by_mean(rgb)
        x_proj = 3.0 * rn - 2.0 * gn
        y_proj = 1.5 * rn + gn - 1.5 * bn
        if not (np.all(np.isfinite(x_proj)) and np.all(np.isfinite(y_proj))):
            raise ValueError("CHROM projection signals contain non-finite values")
    except ValueError as exc:
        return _invalid_result(str(exc))

    # Stage 3: condition X and Y (reuse existing conditioning pipeline)
    try:
        x_filt = condition_signal(
            x_proj,
            fs_val,
            low_hz=low,
            high_hz=high,
            order=4,
            detrend_method="linear",
            min_duration_s=min_dur,
        )
        y_filt = condition_signal(
            y_proj,
            fs_val,
            low_hz=low,
            high_hz=high,
            order=4,
            detrend_method="linear",
            min_duration_s=min_dur,
        )
    except ValueError as exc:
        return _invalid_result(str(exc))

    # Stage 4: adaptive scaling and combination
    try:
        std_x = float(np.std(x_filt))
        std_y = float(np.std(y_filt))
        if not (np.isfinite(std_x) and np.isfinite(std_y)):
            raise ValueError("conditioned CHROM signal standard deviations must be finite")
        if std_x <= _EPS or std_y <= _EPS:
            raise ValueError("conditioned CHROM signals must not be degenerate")

        alpha = std_x / std_y
        if not np.isfinite(alpha) or abs(alpha) <= _EPS:
            raise ValueError("invalid CHROM alpha scaling factor")

        chrom_signal = x_filt - alpha * y_filt
        if chrom_signal.ndim != 1 or chrom_signal.shape[0] != rgb.shape[0]:
            raise ValueError("combined CHROM signal has invalid shape")
        if not np.all(np.isfinite(chrom_signal)):
            raise ValueError("combined CHROM signal contains non-finite values")
        if float(np.std(chrom_signal)) <= _EPS:
            raise ValueError("combined CHROM signal is degenerate")
    except ValueError as exc:
        return _invalid_result(str(exc))

    # Stage 5: spectral analysis (reuse Green helpers)
    try:
        freqs, psd, _, _ = compute_welch_psd(chrom_signal, fs_val, nperseg=nperseg)
        peak_hz, _ = select_peak_in_band(freqs, psd, low, high)
    except ValueError as exc:
        result = _invalid_result(str(exc), conditioned_signal=chrom_signal)
        if "freqs" in locals() and "psd" in locals():
            result["frequencies"] = freqs
            result["psd"] = psd
        if "alpha" in locals():
            result["alpha"] = alpha
        return result

    bpm = float(peak_hz * 60.0)
    return {
        "method": "chrom",
        "bpm": bpm,
        "valid": True,
        "frequency_hz": peak_hz,
        "frequencies": freqs,
        "psd": psd,
        "conditioned_signal": chrom_signal,
        "alpha": float(alpha),
    }
