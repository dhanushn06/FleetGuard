import numpy as np
import pytest

from signal_processing.validation import (
    REQUIRED_RGB_REGIONS,
    validate_rgb_region,
    validate_signal_packet,
    validate_timestamps,
)


def make_valid_packet(n_samples: int = 30, fps: float = 30.0):
    """Create a synthetic, valid signal packet for tests."""
    t = np.arange(n_samples, dtype=np.float64) / fps
    rng = np.random.default_rng(42)
    rgb = {
        region: rng.normal(loc=120.0, scale=2.0, size=(n_samples, 3))
        for region in REQUIRED_RGB_REGIONS
    }
    motion = rng.uniform(0.0, 1.0, size=n_samples)
    track_ok = np.ones(n_samples, dtype=bool)
    track_ok[::10] = False
    brightness = rng.uniform(80.0, 160.0, size=n_samples)
    return {
        "t": t,
        "rgb": rgb,
        "motion": motion,
        "track_ok": track_ok,
        "brightness": brightness,
    }


def test_validate_timestamps_accepts_strictly_increasing():
    t = np.array([0.0, 0.033, 0.067, 0.101], dtype=np.float64)
    out = validate_timestamps(t)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    assert np.allclose(out, t)


@pytest.mark.parametrize(
    "bad_t",
    [
        np.array([0.0, 0.1, 0.1, 0.2]),  # duplicate
        np.array([0.0, 0.2, 0.1, 0.3]),  # decreasing
        np.array([0.0, np.nan, 0.2]),
        np.array([0.0, np.inf, 0.2]),
    ],
)
def test_validate_timestamps_rejects_invalid_sequences(bad_t):
    with pytest.raises(ValueError):
        validate_timestamps(bad_t)


def test_validate_timestamps_rejects_bad_shape_or_length():
    with pytest.raises(ValueError):
        validate_timestamps(np.array([[0.0, 1.0], [2.0, 3.0]]))
    with pytest.raises(ValueError):
        validate_timestamps(np.array([0.5]))  # min_samples default is 2


def test_validate_rgb_region_accepts_valid_shape():
    arr = np.ones((10, 3), dtype=np.float32)
    out = validate_rgb_region(arr, "rgb['forehead']", expected_samples=10)
    assert out.shape == (10, 3)
    assert out.dtype == np.float64


@pytest.mark.parametrize(
    "bad,expected_samples",
    [
        (np.ones((10, 4)), 10),  # wrong channels
        (np.ones((9, 3)), 10),   # length mismatch
        (np.ones((10,)), 10),    # wrong ndim
        (np.array([[1.0, 2.0, np.nan]]), 1),
    ],
)
def test_validate_rgb_region_rejects_invalid_inputs(bad, expected_samples):
    with pytest.raises(ValueError):
        validate_rgb_region(bad, "rgb['forehead']", expected_samples=expected_samples)


def test_validate_signal_packet_valid_normalizes_types_and_values():
    packet = make_valid_packet(n_samples=25)
    packet["track_ok"] = np.array([1, 0, 1] + [1] * 22)  # boolean-like input

    out = validate_signal_packet(packet)

    assert set(out.keys()) == {"t", "rgb", "motion", "track_ok", "brightness"}
    assert out["t"].shape == (25,)
    assert out["motion"].shape == (25,)
    assert out["brightness"].shape == (25,)
    assert out["track_ok"].shape == (25,)
    assert out["track_ok"].dtype == bool
    assert out["track_ok"][0] is np.bool_(True)
    assert out["track_ok"][1] is np.bool_(False)
    for region in REQUIRED_RGB_REGIONS:
        assert out["rgb"][region].shape == (25, 3)
        assert np.issubdtype(out["rgb"][region].dtype, np.floating)


def test_validate_signal_packet_rejects_missing_keys():
    packet = make_valid_packet()
    del packet["motion"]
    with pytest.raises(ValueError, match="missing required keys"):
        validate_signal_packet(packet)


def test_validate_signal_packet_rejects_missing_regions():
    packet = make_valid_packet()
    del packet["rgb"]["left_cheek"]
    with pytest.raises(ValueError, match="missing required regions"):
        validate_signal_packet(packet)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("motion", np.ones(10)),       # length mismatch
        ("brightness", np.ones(10)),
        ("track_ok", np.ones(10, dtype=bool)),
    ],
)
def test_validate_signal_packet_rejects_metadata_length_mismatch(field, bad_value):
    packet = make_valid_packet(n_samples=12)
    packet[field] = bad_value
    with pytest.raises(ValueError, match="length mismatch"):
        validate_signal_packet(packet)


def test_validate_signal_packet_rejects_rgb_length_mismatch():
    packet = make_valid_packet(n_samples=12)
    packet["rgb"]["forehead"] = np.ones((11, 3))
    with pytest.raises(ValueError, match="length mismatch"):
        validate_signal_packet(packet)


def test_validate_signal_packet_rejects_non_mapping_packet():
    with pytest.raises(TypeError):
        validate_signal_packet([1, 2, 3])  # type: ignore[arg-type]
