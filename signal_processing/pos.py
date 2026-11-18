"""POS (Plane-Orthogonal-to-Skin) rPPG heart-rate estimation.

This module implements a single-ROI POS estimator for uniformly sampled RGB
signals with shape ``(N, 3)`` in channel order ``[R, G, B]``.

Formulation (standard POS projection on temporally mean-normalized channels):
    rn = R / mean(R),  gn = G / mean(G),  bn = B / mean(B)
    S1 = gn - bn
    S2 = gn + bn - 2*rn
    alpha = std(S1) / std(S2)
    h = S1 + alpha*S2

The pulse signal ``h`` is then conditioned using the existing detrend/bandpass
pipeline and its heart rate is estimated from the dominant Welch PSD peak
within the configured HR band.
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
    """Build a standardized invalid POS result.

    Purpose:
        Expected input/domain failures must not produce a fabricated BPM.
    """
    result: Dict[str, Any] = {
        "method": "pos",
        "bpm": None,
        "valid": False,
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
    """Validate POS estimator inputs and basic RGB signal integrity.

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

    means = np.mean(rgb, axis=0)
    if not np.all(np.isfinite(means)):
        raise ValueError("RGB channel means must be finite")
    if np.any(np.abs(means) <= _MEAN_EPS):
        raise ValueError("RGB channel means must be non-zero (invalid for normalization)")

    stds = np.std(rgb, axis=0)
    if not np.all(np.isfinite(stds)):
        raise ValueError("RGB channel standard deviations must be finite")
    if np.any(stds <= _EPS):
        raise ValueError("RGB channels must not be constant or degenerate")

    return rgb, fs_val, low, high, min_dur


def _normalize_by_mean(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize R,G,B channels by their temporal means.

    Purpose:
        POS operates on temporally normalized color traces so projections are
        less sensitive to absolute illumination/intensity levels.
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


def estimate_pos_hr(
    rgb_signal: Any,
    fs: Any,
    low_hz: float = DEFAULT_LOWHZ,
    high_hz: float = DEFAULT_HIGHHZ,
    min_duration_s: float = DEFAULT_MIN_DURATION_S,
    nperseg: Optional[int] = None,
) -> Dict[str, Any]:
    """Estimate heart rate from a single uniformly sampled RGB ROI using POS.

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
              "method": "pos",
              "valid": True,
              "bpm": float,
              "frequency_hz": float,
              "frequencies": np.ndarray,
              "psd": np.ndarray,
              "conditioned_signal": np.ndarray,  # conditioned POS pulse signal h
              "alpha": float,
            }
        Invalid (expected input/domain errors, no fabricated BPM):
            {
              "method": "pos",
              "valid": False,
              "bpm": None,
              "reason": str,
              ... (optional fields only if meaningful/available)
            }

    Notes:
        - Standard POS projection is applied to temporally mean-normalized RGB:
          ``S1 = G - B``, ``S2 = G + B - 2R``, ``h = S1 + alpha*S2`` with
          ``alpha = std(S1)/std(S2)`` (guarded for degenerate/invalid cases).
        - The existing conditioning pipeline (detrend + Butterworth SOS +
          zero-phase bandpass) is applied to ``h`` via ``condition_signal``.
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

    # Stage 2: normalize and construct POS projections
    try:
        rn, gn, bn = _normalize_by_mean(rgb)
        s1 = gn - bn
        s2 = gn + bn - 2.0 * rn
        if not (np.all(np.isfinite(s1)) and np.all(np.isfinite(s2))):
            raise ValueError("POS projection signals contain non-finite values")
    except ValueError as exc:
        return _invalid_result(str(exc))

    # Stage 3: adaptive scaling and pulse construction
    try:
        std_s1 = float(np.std(s1))
        std_s2 = float(np.std(s2))
        if not (np.isfinite(std_s1) and np.isfinite(std_s2)):
            raise ValueError("POS projection standard deviations must be finite")
        if std_s1 <= _EPS or std_s2 <= _EPS:
            raise ValueError("POS projection standard deviations must be non-zero/non-degenerate")

        alpha = std_s1 / std_s2
        if not np.isfinite(alpha) or abs(alpha) <= _EPS:
            raise ValueError("invalid POS alpha scaling factor")

        h = s1 + alpha * s2
        if h.ndim != 1 or h.shape[0] != rgb.shape[0]:
            raise ValueError("POS pulse signal has invalid shape")
        if not np.all(np.isfinite(h)):
            raise ValueError("POS pulse signal contains non-finite values")
        if float(np.std(h)) <= _EPS:
            raise ValueError("POS pulse signal is degenerate")
    except ValueError as exc:
        return _invalid_result(str(exc))

    # Stage 4: condition POS pulse signal (reuse existing conditioning pipeline)
    try:
        conditioned = condition_signal(
            h,
            fs_val,
            low_hz=low,
            high_hz=high,
            order=4,
            detrend_method="linear",
            min_duration_s=min_dur,
        )
    except ValueError as exc:
        return _invalid_result(str(exc))

    # Stage 5: spectral analysis (reuse Green helpers)
    try:
        freqs, psd, _, _ = compute_welch_psd(conditioned, fs_val, nperseg=nperseg)
        peak_hz, _ = select_peak_in_band(freqs, psd, low, high)
    except ValueError as exc:
        result = _invalid_result(str(exc), conditioned_signal=conditioned)
        if "freqs" in locals() and "psd" in locals():
            result["frequencies"] = freqs
            result["psd"] = psd
        if "alpha" in locals():
            result["alpha"] = float(alpha)
        return result

    bpm = float(peak_hz * 60.0)
    return {
        "method": "pos",
        "valid": True,
        "bpm": bpm,
        "frequency_hz": peak_hz,
        "frequencies": freqs,
        "psd": psd,
        "conditioned_signal": conditioned,
        "alpha": float(alpha),
    }
