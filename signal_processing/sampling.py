"""Sampling-rate estimation and resampling utilities for rPPG signals.

Purpose:
    Webcam frame timestamps are often irregular due to OS scheduling and camera
    jitter. Classical rPPG pipelines typically assume (or benefit from) a
    uniform sampling grid before filtering and spectral analysis. This module
    provides utilities to estimate an effective sampling rate from timestamps
    and to resample time-aligned signals/packets onto a uniform time base.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np
from scipy import interpolate

from .validation import REQUIRED_RGB_REGIONS, validate_signal_packet


def _as_1d_time_array(t: Any) -> np.ndarray:
    """Normalize a timestamp-like input to a 1D float array for sampling ops.

    Purpose:
        Provides lightweight validation for utilities that operate directly on
        timestamps (without requiring a full packet).
    """
    arr = np.asarray(t, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"t must be a 1D array, got shape {arr.shape}")
    if arr.size < 2:
        raise ValueError("t must contain at least 2 samples to estimate sampling rate")
    if not np.all(np.isfinite(arr)):
        raise ValueError("t must not contain NaN or infinite values")
    dt = np.diff(arr)
    if np.any(dt <= 0):
        raise ValueError("t must be strictly increasing")
    return arr


def estimate_sampling_rate(t: Any, method: str = "median") -> float:
    """Estimate the effective sampling rate (Hz) from timestamps.

    Purpose:
        Convert irregular frame timestamps into a single sampling-rate value
        suitable for uniform resampling and frequency-domain analysis.

    Method:
        - ``median`` (default): ``fs = 1 / median(diff(t))``. Robust to jitter
          and occasional dropped/late frames.
        - ``mean``: ``fs = 1 / mean(diff(t))``. Uses average interval.

    Args:
        t: Timestamps in seconds (1D, strictly increasing).
        method: Estimation strategy, either ``"median"`` or ``"mean"``.

    Returns:
        Estimated sampling rate in Hz as a float.

    Raises:
        ValueError: If timestamps are invalid or method is unknown.
    """
    arr = _as_1d_time_array(t)
    dt = np.diff(arr)

    if method == "median":
        dt_est = float(np.median(dt))
    elif method == "mean":
        dt_est = float(np.mean(dt))
    else:
        raise ValueError("method must be 'median' or 'mean'")

    if dt_est <= 0:
        raise ValueError("estimated time step must be positive")
    return 1.0 / dt_est


def generate_uniform_timestamps(t: Any, fs: float) -> np.ndarray:
    """Generate a uniform timestamp grid spanning the input time range.

    Purpose:
        Create the target time base used for resampling irregularly sampled
        signals to a constant frame rate.

    The grid starts at ``t[0]`` and advances by ``1/fs`` while remaining within
    the original time span ``[t[0], t[-1]]`` (inclusive of the endpoint when it
    lands exactly on the grid).

    Args:
        t: Original timestamps in seconds (1D, strictly increasing).
        fs: Target sampling rate in Hz (must be > 0).

    Returns:
        1D NumPy float64 array of uniform timestamps.

    Raises:
        ValueError: If inputs are invalid or the resulting grid has < 2 samples.
    """
    if fs <= 0:
        raise ValueError("fs must be > 0")

    arr = _as_1d_time_array(t)
    t_start = float(arr[0])
    t_end = float(arr[-1])
    duration = t_end - t_start
    if duration <= 0:
        raise ValueError("timestamp duration must be positive")

    dt = 1.0 / float(fs)
    # Number of steps that fit within [t_start, t_end] (inclusive tolerance)
    n_steps = int(np.floor(duration / dt + 1e-12))
    n_samples = n_steps + 1
    if n_samples < 2:
        raise ValueError("uniform grid must contain at least 2 samples")

    t_uniform = t_start + np.arange(n_samples, dtype=np.float64) * dt
    # Numerical safety: clamp any tiny overshoot due to floating-point error
    if t_uniform[-1] > t_end:
        t_uniform[-1] = t_end
    return t_uniform


def resample_nd_signal(t: Any, values: Any, t_uniform: Any, kind: str = "linear") -> np.ndarray:
    """Resample a 1D or ND time series from ``t`` onto ``t_uniform``.

    Purpose:
        Provide a generic interpolation utility for continuous signals (e.g.
        RGB channels, motion, brightness) and, with ``kind='nearest'`` or
        ``kind='previous'``/``'next'`` where supported, for discrete metadata.

    Args:
        t: Original timestamps, shape (N,), strictly increasing.
        values: Signal values aligned with ``t``. Supported shapes are (N,) or
            (N, D) where interpolation is applied independently per column.
        t_uniform: Target uniform timestamps, shape (M,), must lie within the
            range of ``t`` to avoid extrapolation.
        kind: Interpolation kind passed to ``scipy.interpolate.interp1d``.

    Returns:
        Resampled NumPy array with shape (M,) or (M, D).

    Raises:
        ValueError: If shapes/ranges are incompatible or interpolation fails.
    """
    t_arr = _as_1d_time_array(t)
    x = np.asarray(values)
    tu = np.asarray(t_uniform, dtype=np.float64)

    if tu.ndim != 1 or tu.size < 1:
        raise ValueError("t_uniform must be a 1D non-empty array")
    if not np.all(np.isfinite(tu)):
        raise ValueError("t_uniform must not contain NaN or infinite values")
    if tu[0] < t_arr[0] - 1e-12 or tu[-1] > t_arr[-1] + 1e-12:
        raise ValueError("t_uniform must lie within the range of t (no extrapolation)")

    if x.ndim == 1:
        if x.shape[0] != t_arr.size:
            raise ValueError("1D values length must match t")
        f = interpolate.interp1d(
            t_arr,
            x.astype(np.float64, copy=False),
            kind=kind,
            axis=0,
            bounds_error=True,
            assume_sorted=True,
        )
        return np.asarray(f(tu), dtype=np.float64)

    if x.ndim == 2:
        if x.shape[0] != t_arr.size:
            raise ValueError("2D values first dimension must match t")
        f = interpolate.interp1d(
            t_arr,
            x.astype(np.float64, copy=False),
            kind=kind,
            axis=0,
            bounds_error=True,
            assume_sorted=True,
        )
        return np.asarray(f(tu), dtype=np.float64)

    raise ValueError("values must be a 1D or 2D array aligned with t")


def resample_signal_packet(packet: Mapping[str, Any], target_fs: Optional[float] = None) -> Dict[str, Any]:
    """Validate and resample a full signal packet onto a uniform time base.

    Purpose:
        Convenience utility that prepares packet-aligned RGB regions and
        metadata for downstream algorithms that expect uniformly sampled input.

    Behavior:
        - Validates the input packet using ``validate_signal_packet``.
        - Estimates sampling rate from ``t`` when ``target_fs`` is None.
        - Generates uniform timestamps spanning the original duration.
        - Linearly interpolates RGB regions, motion, and brightness.
        - Resamples ``track_ok`` using nearest-neighbor interpolation and
          returns a boolean array.

    Args:
        packet: Signal packet in the shared FleetGuard format.
        target_fs: Desired uniform sampling rate in Hz. If None, uses the
            median-estimated rate from the input timestamps.

    Returns:
        A new normalized packet dictionary with uniformly sampled ``t`` and
        resampled ``rgb``, ``motion``, ``track_ok``, and ``brightness``.

    Raises:
        TypeError/ValueError: If validation, sampling-rate estimation, or
            resampling fails.
    """
    validated = validate_signal_packet(packet, min_samples=2)
    t = validated["t"]

    fs = float(target_fs) if target_fs is not None else estimate_sampling_rate(t, method="median")
    t_uniform = generate_uniform_timestamps(t, fs)

    resampled_rgb: Dict[str, np.ndarray] = {}
    for region in REQUIRED_RGB_REGIONS:
        resampled_rgb[region] = resample_nd_signal(t, validated["rgb"][region], t_uniform, kind="linear")

    motion = resample_nd_signal(t, validated["motion"], t_uniform, kind="linear")
    brightness = resample_nd_signal(t, validated["brightness"], t_uniform, kind="linear")

    # Nearest-neighbor for boolean tracking flags, then cast back to bool.
    track_ok_float = validated["track_ok"].astype(np.float64)
    track_ok_resampled = resample_nd_signal(t, track_ok_float, t_uniform, kind="nearest")
    track_ok = track_ok_resampled >= 0.5

    return {
        "t": t_uniform,
        "rgb": resampled_rgb,
        "motion": motion,
        "track_ok": track_ok,
        "brightness": brightness,
    }
