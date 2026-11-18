"""Signal validation utilities for FleetGuard rPPG inputs.

This module validates and normalizes the shared data packet format used by the
signal-processing pipeline:

    {
        "t": timestamps_in_seconds,
        "rgb": {
            "forehead": numpy_array_of_shape_Nx3,
            "left_cheek": numpy_array_of_shape_Nx3,
            "right_cheek": numpy_array_of_shape_Nx3
        },
        "motion": motion_values,
        "track_ok": boolean_values,
        "brightness": brightness_values
    }

Validation is intentionally strict where correctness matters for downstream
sampling-rate estimation and resampling (e.g. finite, strictly increasing
timestamps and consistent sample counts across all channels/metadata).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

REQUIRED_RGB_REGIONS = ("forehead", "left_cheek", "right_cheek")


def _as_numpy_array(values: Any, name: str) -> np.ndarray:
    """Convert input to a NumPy array or raise a clear TypeError/ValueError.

    Purpose:
        Centralizes conversion so validation errors consistently identify the
        offending field name.
    """
    try:
        arr = np.asarray(values)
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError(f"{name} must be array-like and convertible to numpy.ndarray") from exc
    return arr


def validate_timestamps(t: Any, *, min_samples: int = 2) -> np.ndarray:
    """Validate and normalize the timestamp vector.

    Purpose:
        Ensure timestamps form a valid time base for sampling-rate estimation
        and resampling.

    Requirements:
        - 1D array-like sequence
        - at least ``min_samples`` samples
        - numeric, finite values
        - strictly increasing (no duplicates, no backwards time)

    Args:
        t: Timestamps in seconds.
        min_samples: Minimum number of samples required.

    Returns:
        A 1D NumPy float64 array of timestamps.

    Raises:
        TypeError: If ``t`` is not array-like or not numeric.
        ValueError: If shape, length, finiteness, or monotonicity is invalid.
    """
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1")

    arr = _as_numpy_array(t, "t")
    if arr.ndim != 1:
        raise ValueError(f"t must be a 1D array, got shape {arr.shape}")
    if arr.size < min_samples:
        raise ValueError(
            f"t must contain at least {min_samples} samples, got {arr.size}"
        )
    if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_):
        raise TypeError("t must contain numeric timestamp values")

    arr = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError("t must not contain NaN or infinite values")

    dt = np.diff(arr)
    if np.any(dt <= 0):
        raise ValueError("t must be strictly increasing (all consecutive differences > 0)")

    return arr


def validate_rgb_region(rgb: Any, name: str, expected_samples: int) -> np.ndarray:
    """Validate and normalize a single RGB region signal.

    Purpose:
        Ensure each facial-region trace has the expected temporal length and
        channel layout (R, G, B per sample) before any rPPG processing.

    Requirements:
        - 2D array-like sequence with shape (N, 3)
        - N == ``expected_samples``
        - numeric, finite values

    Args:
        rgb: RGB samples for one region.
        name: Human-readable region/key name for error messages.
        expected_samples: Required number of temporal samples.

    Returns:
        A NumPy float64 array of shape (N, 3).

    Raises:
        TypeError: If values are not numeric.
        ValueError: If shape, length, or finiteness is invalid.
    """
    if expected_samples < 0:
        raise ValueError("expected_samples must be >= 0")

    arr = _as_numpy_array(rgb, name)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array of shape (N, 3), got shape {arr.shape}")
    if arr.shape[1] != 3:
        raise ValueError(f"{name} must have exactly 3 channels (R,G,B), got shape {arr.shape}")
    if arr.shape[0] != expected_samples:
        raise ValueError(
            f"{name} length mismatch: expected {expected_samples} samples, got {arr.shape[0]}"
        )
    if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_):
        raise TypeError(f"{name} must contain numeric RGB values")

    arr = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must not contain NaN or infinite values")

    return arr


def validate_1d_metadata(
    values: Any,
    name: str,
    expected_samples: int,
    *,
    boolean: bool = False,
) -> np.ndarray:
    """Validate and normalize per-frame metadata aligned with timestamps.

    Purpose:
        Ensure auxiliary signals (motion, tracking flags, brightness) are
        temporally aligned with ``t`` and safe to use in filtering/quality
        checks.

    Args:
        values: Metadata sequence.
        name: Field name for error messages.
        expected_samples: Required number of samples.
        boolean: If True, normalize values to a boolean array (for ``track_ok``).
            If False, require numeric finite values.

    Returns:
        A 1D NumPy array (bool or float64 depending on ``boolean``).

    Raises:
        TypeError/ValueError: If type, shape, length, or finiteness is invalid.
    """
    arr = _as_numpy_array(values, name)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {arr.shape}")
    if arr.size != expected_samples:
        raise ValueError(
            f"{name} length mismatch: expected {expected_samples} samples, got {arr.size}"
        )

    if boolean:
        # Accept typical boolean-like inputs (bool, 0/1). Normalize explicitly.
        if arr.dtype == object:
            raise TypeError(f"{name} must contain boolean-like values")
        return arr.astype(bool, copy=False)

    if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_):
        raise TypeError(f"{name} must contain numeric values")

    arr = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must not contain NaN or infinite values")
    return arr


def validate_signal_packet(packet: Mapping[str, Any], *, min_samples: int = 2) -> Dict[str, Any]:
    """Validate and normalize a complete signal packet.

    Purpose:
        Provide a single entry point that enforces the shared data contract and
        returns a normalized, NumPy-backed packet for downstream utilities
        (sampling-rate calculation, resampling, and future rPPG methods).

    Args:
        packet: Mapping containing ``t``, ``rgb``, ``motion``, ``track_ok``,
            and ``brightness``.
        min_samples: Minimum number of temporal samples required.

    Returns:
        A new dictionary with validated/normalized fields:
            - ``t``: float64 array of shape (N,)
            - ``rgb``: dict of region -> float64 array of shape (N, 3)
            - ``motion``: float64 array of shape (N,)
            - ``track_ok``: bool array of shape (N,)
            - ``brightness``: float64 array of shape (N,)

    Raises:
        TypeError: If packet/fields have incorrect types.
        ValueError: If required keys are missing or any value fails validation.
    """
    if not isinstance(packet, Mapping):
        raise TypeError("packet must be a mapping (e.g. dict)")

    required_keys = {"t", "rgb", "motion", "track_ok", "brightness"}
    missing = required_keys - set(packet.keys())
    if missing:
        raise ValueError(f"packet is missing required keys: {sorted(missing)}")

    t = validate_timestamps(packet["t"], min_samples=min_samples)
    n_samples = t.size

    rgb = packet["rgb"]
    if not isinstance(rgb, Mapping):
        raise TypeError("rgb must be a mapping of region names to Nx3 arrays")

    missing_regions = set(REQUIRED_RGB_REGIONS) - set(rgb.keys())
    if missing_regions:
        raise ValueError(f"rgb is missing required regions: {sorted(missing_regions)}")

    validated_rgb: Dict[str, np.ndarray] = {}
    for region in REQUIRED_RGB_REGIONS:
        validated_rgb[region] = validate_rgb_region(rgb[region], f"rgb['{region}']", n_samples)

    motion = validate_1d_metadata(packet["motion"], "motion", n_samples)
    track_ok = validate_1d_metadata(packet["track_ok"], "track_ok", n_samples, boolean=True)
    brightness = validate_1d_metadata(packet["brightness"], "brightness", n_samples)

    return {
        "t": t,
        "rgb": validated_rgb,
        "motion": motion,
        "track_ok": track_ok,
        "brightness": brightness,
    }
