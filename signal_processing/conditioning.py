"""Signal conditioning utilities for webcam rPPG preprocessing.

This module provides array-level conditioning steps applied after validation and
uniform resampling, and before rPPG-specific algorithms (Green/CHROM/POS):

    detrend -> Butterworth bandpass (SOS, zero-phase)

The implementation targets an offline/windowed hackathon pipeline and favors
clear, deterministic behavior over silent best-effort processing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal

DEFAULT_LOWHZ = 0.75
DEFAULT_HIGHHZ = 3.0
DEFAULT_FILTER_ORDER = 4
DEFAULT_DETREND_METHOD = "linear"
# Minimum duration helps ensure a few cardiac cycles and stable zero-phase IIR
# filtering in the default heart-rate band (45-180 BPM).
MIN_CONDITION_DURATION_S = 4.0
_EPS_STD = 1e-12


def _as_1d_signal(x: Any, name: str = "x") -> np.ndarray:
    """Convert input to a 1D float64 NumPy array and validate basic integrity.

    Purpose:
        Centralize input normalization so conditioning functions reject invalid
        shapes/types early with consistent error messages.

    Raises:
        ValueError: If the signal is not 1D or contains non-finite values.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must not contain NaN or infinite values")
    return arr


def _validate_fs(fs: Any) -> float:
    """Validate sampling rate for conditioning operations.

    Purpose:
        Ensure filter design and duration checks use a meaningful, positive,
        finite sampling rate in Hz.
    """
    try:
        fs_val = float(fs)
    except (TypeError, ValueError) as exc:
        raise ValueError("fs must be a finite numeric sampling rate in Hz") from exc
    if not np.isfinite(fs_val) or fs_val <= 0:
        raise ValueError("fs must be a finite sampling rate > 0 Hz")
    return fs_val


def _validate_min_length(n_samples: int, fs: float, min_duration_s: float = MIN_CONDITION_DURATION_S) -> None:
    """Validate that a signal is long enough for reliable conditioning.

    Purpose:
        Prevent unstable or misleading zero-phase bandpass filtering on very
        short windows by requiring a minimum duration (several cardiac cycles).

    Raises:
        ValueError: If the signal is shorter than the required minimum.
    """
    if n_samples < 2:
        raise ValueError("x must contain at least 2 samples")
    min_samples = max(16, int(np.ceil(fs * min_duration_s)))
    if n_samples < min_samples:
        raise ValueError(
            "x is too short for reliable conditioning: "
            f"need at least {min_samples} samples ({min_duration_s:.2f} s at fs={fs:.3f} Hz), "
            f"got {n_samples}"
        )


def _validate_not_constant(x: np.ndarray, name: str = "x") -> None:
    """Reject constant or near-constant (degenerate) signals.

    Purpose:
        Signals without meaningful variation cannot carry a pulsatile component;
        processing them would produce misleading outputs and should fail fast.
    """
    if x.size == 0:
        raise ValueError(f"{name} must not be empty")
    # Fast path for exact constants, robust path for near-constants.
    if np.all(x == x[0]) or float(np.std(x)) <= _EPS_STD:
        raise ValueError(f"{name} must not be constant or near-constant")


def detrend_signal(x: Any, method: str = DEFAULT_DETREND_METHOD) -> np.ndarray:
    """Remove slow trends from a 1D signal.

    Purpose:
        Suppress slow illumination/sensor drift prior to bandpass filtering so
        the cardiac band is less contaminated by low-frequency components.

    Implementation:
        Uses ``scipy.signal.detrend`` for supported methods:
        - ``"linear"``: remove best-fit line (slope + offset)
        - ``"constant"``: remove mean (DC)

    Args:
        x: 1D signal array.
        method: Detrending method, either ``"linear"`` or ``"constant"``.

    Returns:
        Detrended 1D float64 NumPy array.

    Raises:
        ValueError: If input is invalid or method is unsupported.
    """
    arr = _as_1d_signal(x, "x")
    if arr.size < 2:
        raise ValueError("x must contain at least 2 samples for detrending")
    if method not in {"linear", "constant"}:
        raise ValueError("method must be 'linear' or 'constant'")

    # Do not silently process degenerate signals.
    _validate_not_constant(arr, "x")

    detrended = signal.detrend(arr, type=method, overwrite_data=False)
    return np.asarray(detrended, dtype=np.float64)


def design_bandpass_filter(
    fs: Any,
    low_hz: float = DEFAULT_LOWHZ,
    high_hz: float = DEFAULT_HIGHHZ,
    order: int = DEFAULT_FILTER_ORDER,
) -> np.ndarray:
    """Design a Butterworth bandpass filter in SOS (second-order sections) form.

    Purpose:
        Create a numerically stable IIR bandpass for isolating the heart-rate
        band (default 0.75-3.0 Hz / 45-180 BPM) in uniformly sampled signals.

    Args:
        fs: Sampling rate in Hz.
        low_hz: Lower cutoff frequency in Hz (must be > 0).
        high_hz: Upper cutoff frequency in Hz (must be < Nyquist).
        order: Butterworth filter order (positive integer).

    Returns:
        SOS coefficient array suitable for ``scipy.signal.sosfiltfilt``.

    Raises:
        ValueError: If sampling rate, cutoffs, Nyquist constraints, or order are
            invalid.
    """
    fs_val = _validate_fs(fs)

    try:
        low = float(low_hz)
        high = float(high_hz)
    except (TypeError, ValueError) as exc:
        raise ValueError("low_hz and high_hz must be finite numeric values") from exc

    if not (np.isfinite(low) and np.isfinite(high)):
        raise ValueError("low_hz and high_hz must be finite")
    if low <= 0:
        raise ValueError("low_hz must be > 0")
    if low >= high:
        raise ValueError("low_hz must be < high_hz")

    nyquist = 0.5 * fs_val
    if high >= nyquist:
        raise ValueError(
            f"high_hz must be < Nyquist frequency ({nyquist:.3f} Hz for fs={fs_val:.3f} Hz)"
        )

    if not isinstance(order, (int, np.integer)) or isinstance(order, bool):
        raise ValueError("order must be a positive integer")
    order_int = int(order)
    if order_int < 1:
        raise ValueError("order must be >= 1")

    sos = signal.butter(
        order_int,
        [low, high],
        btype="bandpass",
        fs=fs_val,
        output="sos",
    )
    return np.asarray(sos, dtype=np.float64)


def bandpass_filter(x: Any, sos: Any) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter (SOS) to a 1D signal.

    Purpose:
        Isolate the cardiac frequency band with zero phase distortion for an
        offline/windowed pipeline using forward-backward filtering.

    Args:
        x: 1D signal array.
        sos: Second-order-sections filter coefficients (e.g. from
            ``design_bandpass_filter``).

    Returns:
        Bandpass-filtered 1D float64 NumPy array with the same length as input.

    Raises:
        ValueError: If signal or SOS inputs are invalid, or filtering cannot be
            performed safely.
    """
    arr = _as_1d_signal(x, "x")
    _validate_not_constant(arr, "x")

    sos_arr = np.asarray(sos, dtype=np.float64)
    if sos_arr.ndim != 2 or sos_arr.shape[1] != 6:
        raise ValueError("sos must be a 2D array with shape (n_sections, 6)")
    if not np.all(np.isfinite(sos_arr)):
        raise ValueError("sos must contain finite coefficients")

    # Require enough samples for stable zero-phase filtering (SciPy pad length).
    # Default padlen for sosfiltfilt is 3 * (n_sections - 1) in SOS form? In
    # SciPy it uses 3*ntaps-like padding; for SOS, padlen defaults to
    # 3 * (2 * n_sections - 1) in recent implementations. Use a conservative
    # check that also enforces a practical minimum.
    n_sections = sos_arr.shape[0]
    min_for_pad = 3 * (2 * n_sections - 1) + 1
    if arr.size <= min_for_pad:
        raise ValueError(
            "x is too short for stable zero-phase SOS filtering: "
            f"need more than {min_for_pad} samples for this filter, got {arr.size}"
        )

    try:
        filtered = signal.sosfiltfilt(sos_arr, arr)
    except ValueError as exc:
        # Re-raise with clear context (e.g. SciPy pad/length issues).
        raise ValueError(f"bandpass filtering failed: {exc}") from exc

    filtered = np.asarray(filtered, dtype=np.float64)
    if filtered.shape != arr.shape:
        raise ValueError("filtered signal shape does not match input shape")
    if not np.all(np.isfinite(filtered)):
        raise ValueError("filtered signal contains non-finite values")
    return filtered


def condition_signal(
    x: Any,
    fs: Any,
    low_hz: float = DEFAULT_LOWHZ,
    high_hz: float = DEFAULT_HIGHHZ,
    order: int = DEFAULT_FILTER_ORDER,
    detrend_method: str = DEFAULT_DETREND_METHOD,
    min_duration_s: float = MIN_CONDITION_DURATION_S,
) -> np.ndarray:
    """Condition a 1D uniformly sampled signal for rPPG analysis.

    Purpose:
        Apply the approved preprocessing chain for offline/windowed analysis:
        validate -> detrend -> Butterworth SOS bandpass -> zero-phase filtering.

    Args:
        x: 1D signal array (uniformly sampled).
        fs: Sampling rate in Hz.
        low_hz: Lower bandpass cutoff in Hz (default 0.75 Hz / 45 BPM).
        high_hz: Upper bandpass cutoff in Hz (default 3.0 Hz / 180 BPM).
        order: Butterworth filter order (default 4).
        detrend_method: ``"linear"`` or ``"constant"`` for
            ``scipy.signal.detrend``.
        min_duration_s: Minimum signal duration required for reliable
            conditioning (default 4.0 s).

    Returns:
        Conditioned 1D float64 NumPy array with the same length as input.

    Raises:
        ValueError: If inputs are invalid (shape, non-finite values, sampling
            rate, Nyquist/cutoffs, insufficient length, or constant/degenerate
            signal) or if filtering cannot be performed safely.
    """
    arr = _as_1d_signal(x, "x")
    fs_val = _validate_fs(fs)

    if not isinstance(min_duration_s, (int, float, np.floating)) or not np.isfinite(
        float(min_duration_s)
    ) or float(min_duration_s) <= 0:
        raise ValueError("min_duration_s must be a finite value > 0")
    min_duration_s_val = float(min_duration_s)

    _validate_min_length(arr.size, fs_val, min_duration_s=min_duration_s_val)
    _validate_not_constant(arr, "x")

    detrended = detrend_signal(arr, method=detrend_method)
    if not np.all(np.isfinite(detrended)):
        raise ValueError("detrended signal contains non-finite values")
    _validate_not_constant(detrended, "detrended signal")

    sos = design_bandpass_filter(fs_val, low_hz=low_hz, high_hz=high_hz, order=order)
    conditioned = bandpass_filter(detrended, sos)

    if conditioned.shape != arr.shape:
        raise ValueError("conditioned signal shape does not match input shape")
    if not np.all(np.isfinite(conditioned)):
        raise ValueError("conditioned signal contains non-finite values")
    return conditioned
