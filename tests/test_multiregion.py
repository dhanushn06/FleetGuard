import numpy as np
import pytest

from signal_processing.multiregion import estimate_multiregion_hr


FS = 30.0
DURATION_S = 12.0  # >= default min_duration_s (10 s)
N_SAMPLES = int(FS * DURATION_S)


def make_packets(
    bpm: float = 72.0,
    fs: float = FS,
    duration_s: float = DURATION_S,
    noise_std: float = 0.0,
    seed: int = 0,
    tracking_failures: Optional[list] = None,
    missing_regions: Optional[dict] = None,
    out_of_band: bool = False,
) -> list:
    """Generate synthetic camera packets in Member 1's format.

    Args:
        bpm: Heart rate in BPM for the pulsatile signal.
        fs: Sampling frequency in Hz.
        duration_s: Duration of the signal in seconds.
        noise_std: Standard deviation of Gaussian noise added to RGB.
        seed: Random seed for reproducibility.
        tracking_failures: List of frame indices where tracking_status=False.
        missing_regions: Dict mapping region name to list of frame indices where region is missing.
        out_of_band: If True, add strong out-of-band components.

    Returns:
        List of packet dictionaries.
    """
    rng = np.random.default_rng(seed)
    n = int(np.round(duration_s * fs))
    t = np.arange(n, dtype=np.float64) / fs
    f0 = bpm / 60.0

    # Base signals with pulsatile component
    pulse = np.sin(2 * np.pi * f0 * t + 0.2)
    r_base = 140.0 * (1.0 + 0.004 * pulse) + 0.01 * t
    g_base = 120.0 * (1.0 + 0.010 * np.sin(2 * np.pi * f0 * t + 0.05)) + 0.015 * t
    b_base = 100.0 * (1.0 + 0.003 * np.sin(2 * np.pi * f0 * t + 0.35)) + 0.008 * t

    if out_of_band:
        drift = 2.5 * np.sin(2 * np.pi * 0.15 * t)
        hf = 0.9 * np.sin(2 * np.pi * 4.5 * t)
        r_base += 0.8 * drift + 0.4 * hf
        g_base += 1.2 * drift + 0.3 * hf
        b_base += 1.0 * drift + 0.5 * hf

    if noise_std > 0:
        r_base += rng.normal(0.0, noise_std, size=n)
        g_base += rng.normal(0.0, noise_std, size=n)
        b_base += rng.normal(0.0, noise_std, size=n)

    tracking_failures = set(tracking_failures or [])
    missing_regions = missing_regions or {}

    packets = []
    for i in range(n):
        ts = float(t[i])
        tracking_ok = i not in tracking_failures

        pkt = {
            "timestamp": ts,
            "wall_time": f"2024-01-01T00:00:{int(ts):02d}.{int((ts%1)*1000):03d}Z",
            "tracking_status": tracking_ok,
            "motion_displacement": 0.1 * np.sin(2 * np.pi * 0.5 * ts) if tracking_ok else None,
            "brightness": 150.0 + 2.0 * ts if tracking_ok else None,
            "regions": None,
        }

        if tracking_ok:
            regions = {}
            for region_name in ("forehead", "left_cheek", "right_cheek"):
                if region_name in missing_regions and i in missing_regions[region_name]:
                    continue  # region missing for this frame

                # Add small per-region variation
                region_seed = hash((region_name, i, seed)) % 2**32
                region_rng = np.random.default_rng(region_seed)
                r = r_base[i] + region_rng.normal(0.0, 0.5)
                g = g_base[i] + region_rng.normal(0.0, 0.5)
                b = b_base[i] + region_rng.normal(0.0, 0.5)

                regions[region_name] = {
                    "rgb": {"r": float(r), "g": float(g), "b": float(b)},
                    "brightness": float(150.0 + 2.0 * ts + region_rng.normal(0.0, 1.0)),
                    "pixel_count": 1000 + region_rng.integers(-50, 50),
                }
            pkt["regions"] = regions

        packets.append(pkt)

    return packets


def make_packets_with_gap(
    bpm: float = 72.0,
    fs: float = FS,
    gap_start_s: float = 10.0,
    gap_duration_s: float = 5.0,
    total_duration_s: float = 25.0,
    seed: int = 0,
) -> list:
    """Create packets with a specific tracking gap.

    Args:
        gap_start_s: When the gap starts (seconds from start).
        gap_duration_s: How long the gap lasts.
        total_duration_s: Total duration of the signal.
    """
    n = int(np.round(total_duration_s * fs))
    t = np.arange(n, dtype=np.float64) / fs
    f0 = bpm / 60.0

    pulse = np.sin(2 * np.pi * f0 * t + 0.2)
    r_base = 140.0 * (1.0 + 0.004 * pulse) + 0.01 * t
    g_base = 120.0 * (1.0 + 0.010 * np.sin(2 * np.pi * f0 * t + 0.05)) + 0.015 * t
    b_base = 100.0 * (1.0 + 0.003 * np.sin(2 * np.pi * f0 * t + 0.35)) + 0.008 * t

    gap_start_idx = int(gap_start_s * fs)
    gap_end_idx = int((gap_start_s + gap_duration_s) * fs)

    packets = []
    for i in range(n):
        ts = float(t[i])
        tracking_ok = not (gap_start_idx <= i < gap_end_idx)

        pkt = {
            "timestamp": ts,
            "wall_time": f"2024-01-01T00:00:{int(ts):02d}.{int((ts%1)*1000):03d}Z",
            "tracking_status": tracking_ok,
            "motion_displacement": 0.1 * np.sin(2 * np.pi * 0.5 * ts) if tracking_ok else None,
            "brightness": 150.0 + 2.0 * ts if tracking_ok else None,
            "regions": None,
        }

        if tracking_ok:
            regions = {}
            for region_name in ("forehead", "left_cheek", "right_cheek"):
                region_seed = hash((region_name, i, seed)) % 2**32
                region_rng = np.random.default_rng(region_seed)
                r = r_base[i] + region_rng.normal(0.0, 0.5)
                g = g_base[i] + region_rng.normal(0.0, 0.5)
                b = b_base[i] + region_rng.normal(0.0, 0.5)

                regions[region_name] = {
                    "rgb": {"r": float(r), "g": float(g), "b": float(b)},
                    "brightness": float(150.0 + 2.0 * ts + region_rng.normal(0.0, 1.0)),
                    "pixel_count": 1000 + region_rng.integers(-50, 50),
                }
            pkt["regions"] = regions

        packets.append(pkt)

    return packets


def assert_region_result_valid(region_result: dict, expected_bpm: float, tol_bpm: float = 7.0):
    """Assert that a region's three algorithms produce valid results near expected BPM."""
    for algo in ("green", "chrom", "pos"):
        res = region_result[algo]
        assert res["valid"] is True, f"{algo} should be valid, got reason: {res.get('reason')}"
        assert isinstance(res["bpm"], float)
        assert abs(res["bpm"] - expected_bpm) <= tol_bpm, f"{algo}: {res['bpm']} vs {expected_bpm}"


def assert_region_result_invalid(region_result: dict, reason_substring: str = ""):
    """Assert that all three algorithms in a region are invalid."""
    for algo in ("green", "chrom", "pos"):
        res = region_result[algo]
        assert res["valid"] is False, f"{algo} should be invalid"
        assert res["bpm"] is None
        assert res["frequency_hz"] is None
        if reason_substring:
            assert reason_substring.lower() in res.get("reason", "").lower()


def test_estimate_multiregion_hr_valid_packets_all_regions():
    """All three regions have valid data, all three algorithms run."""
    packets = make_packets(bpm=72.0, seed=1, noise_std=0.05)
    result = estimate_multiregion_hr(packets)

    for region in ("forehead", "left_cheek", "right_cheek"):
        assert region in result
        assert_region_result_valid(result[region], expected_bpm=72.0)
        meta = result[region]["metadata"]
        assert meta["valid_samples"] > 0
        assert meta["sampling_rate_hz"] == pytest.approx(FS, rel=0.1)
        assert 0.9 <= meta["tracking_fraction"] <= 1.0
        assert meta["num_continuous_segments"] == 1
        assert meta["longest_segment_duration_s"] == pytest.approx(DURATION_S, rel=0.1)
        assert meta["selected_segment_duration_s"] == pytest.approx(DURATION_S, rel=0.1)
        assert meta["gap_warning"] is None


@pytest.mark.parametrize("expected_bpm", [60.0, 72.0, 90.0])
def test_estimate_multiregion_hr_known_heart_rates(expected_bpm):
    """Test known heart rates at 60, 72, 90 BPM."""
    packets = make_packets(bpm=expected_bpm, seed=42, noise_std=0.02)
    result = estimate_multiregion_hr(packets)

    for region in ("forehead", "left_cheek", "right_cheek"):
        assert_region_result_valid(result[region], expected_bpm=expected_bpm, tol_bpm=6.0)


def test_estimate_multiregion_hr_noisy_signal():
    """Test with moderate noise."""
    packets = make_packets(bpm=72.0, seed=2, noise_std=0.3)
    result = estimate_multiregion_hr(packets)

    for region in ("forehead", "left_cheek", "right_cheek"):
        assert_region_result_valid(result[region], expected_bpm=72.0, tol_bpm=8.0)


def test_estimate_multiregion_hr_out_of_band_components():
    """Test with strong out-of-band components."""
    packets = make_packets(bpm=78.0, seed=3, noise_std=0.05, out_of_band=True)
    result = estimate_multiregion_hr(packets)

    for region in ("forehead", "left_cheek", "right_cheek"):
        assert_region_result_valid(result[region], expected_bpm=78.0, tol_bpm=8.0)


def test_estimate_multiregion_hr_missing_tracking_packets():
    """Test with intermittent tracking failures (short gaps that don't split)."""
    # Fail tracking at frames 100-102 (3 frames = 0.1s at 30Hz) - well below 3x median dt
    tracking_failures = list(range(100, 103))
    packets = make_packets(bpm=72.0, seed=4, tracking_failures=tracking_failures)
    result = estimate_multiregion_hr(packets)

    for region in ("forehead", "left_cheek", "right_cheek"):
        assert_region_result_valid(result[region], expected_bpm=72.0, tol_bpm=7.0)
        meta = result[region]["metadata"]
        assert meta["missing_samples"] == len(tracking_failures)
        assert meta["tracking_fraction"] < 1.0
        # Short gap should NOT split into multiple segments
        assert meta["num_continuous_segments"] == 1
        assert meta["gap_warning"] is None


def test_estimate_multiregion_hr_missing_region_measurements():
    """Test with specific region missing on some frames (short gap that doesn't split)."""
    # 10 frames missing = 0.33s at 30Hz. With 5x median dt threshold (0.167s),
    # the dt across gap is 11 * 0.033 = 0.36s > 0.167s, so it WILL split.
    # Use a smaller gap: 3 frames = 0.1s, dt across = 4 * 0.033 = 0.13s < 0.167s (5x threshold)
    missing_regions = {"forehead": list(range(50, 53))}  # forehead missing for 3 frames
    packets = make_packets(bpm=72.0, seed=5, missing_regions=missing_regions)
    result = estimate_multiregion_hr(packets)

    # Forehead should have some missing samples but still work (short gap doesn't split)
    assert_region_result_valid(result["forehead"], expected_bpm=72.0, tol_bpm=7.0)
    assert result["forehead"]["metadata"]["missing_samples"] >= 3
    assert result["forehead"]["metadata"]["num_continuous_segments"] == 1

    # Other regions should be unaffected
    assert_region_result_valid(result["left_cheek"], expected_bpm=72.0)
    assert_region_result_valid(result["right_cheek"], expected_bpm=72.0)


def test_estimate_multiregion_hr_invalid_rgb_values():
    """Test with NaN/Inf in RGB values."""
    packets = make_packets(bpm=72.0, seed=6)
    # Inject NaN in forehead RGB at frame 10
    packets[10]["regions"]["forehead"]["rgb"]["r"] = np.nan
    packets[11]["regions"]["forehead"]["rgb"]["g"] = np.inf
    packets[12]["regions"]["forehead"]["rgb"]["b"] = -np.inf

    result = estimate_multiregion_hr(packets)

    # Forehead should handle invalid values gracefully (those frames become missing)
    assert_region_result_valid(result["forehead"], expected_bpm=72.0, tol_bpm=7.0)
    assert result["forehead"]["metadata"]["missing_samples"] >= 3


def test_estimate_multiregion_hr_empty_input():
    """Test with empty packet list."""
    result = estimate_multiregion_hr([])

    for region in ("forehead", "left_cheek", "right_cheek"):
        assert_region_result_invalid(result[region], "empty packet list")
        assert result[region]["metadata"]["valid_samples"] == 0


def test_estimate_multiregion_hr_insufficient_duration():
    """Test with signal duration less than min_duration_s."""
    packets = make_packets(bpm=72.0, duration_s=5.0, seed=7)  # 5s < 10s default
    result = estimate_multiregion_hr(packets)

    for region in ("forehead", "left_cheek", "right_cheek"):
        assert_region_result_invalid(result[region], "insufficient continuous signal")


def test_estimate_multiregion_hr_large_tracking_gap_splits_signal():
    """Test that a large tracking gap (5s) splits the signal into segments."""
    # 10.1s valid, 5s gap, 10.1s valid = 25.2s total
    # Each segment is 10.1s (303 frames at 30Hz) to ensure >= 10s duration
    packets = make_packets_with_gap(
        bpm=72.0, gap_start_s=10.1, gap_duration_s=5.0, total_duration_s=25.2, seed=8
    )
    result = estimate_multiregion_hr(packets)

    for region in ("forehead", "left_cheek", "right_cheek"):
        meta = result[region]["metadata"]
        # Should detect 2 continuous segments
        assert meta["num_continuous_segments"] == 2
        assert meta["longest_segment_duration_s"] == pytest.approx(10.1, abs=0.5)
        assert meta["gap_warning"] is not None
        assert "split into 2 continuous segments" in meta["gap_warning"]

        # Should select the longest segment (both are 10.1s, so either works)
        assert meta["selected_segment_duration_s"] == pytest.approx(10.1, abs=0.5)
        # Algorithms should run on the selected 10.1s segment
        assert_region_result_valid(result[region], expected_bpm=72.0, tol_bpm=7.0)


def test_estimate_multiregion_hr_two_segments_neither_long_enough():
    """Test with two segments, neither meeting min_duration_s."""
    # 5s valid, 5s gap, 5s valid = 15s total, but each segment only 5s < 10s default
    packets = make_packets_with_gap(
        bpm=72.0, gap_start_s=5.0, gap_duration_s=5.0, total_duration_s=15.0, seed=9
    )
    result = estimate_multiregion_hr(packets)

    for region in ("forehead", "left_cheek", "right_cheek"):
        assert_region_result_invalid(result[region], "insufficient continuous signal")
        meta = result[region]["metadata"]
        assert meta["num_continuous_segments"] == 2
        assert meta["longest_segment_duration_s"] == pytest.approx(5.0, abs=0.5)
        assert meta["selected_segment_duration_s"] == 0.0


def test_estimate_multiregion_hr_one_long_one_short_segment():
    """Test with one long segment (12.1s) and one short segment (3s)."""
    # 12.1s valid, 5s gap, 3s valid = 20.1s total
    packets = make_packets_with_gap(
        bpm=72.0, gap_start_s=12.1, gap_duration_s=5.0, total_duration_s=20.1, seed=10
    )
    result = estimate_multiregion_hr(packets)

    for region in ("forehead", "left_cheek", "right_cheek"):
        meta = result[region]["metadata"]
        assert meta["num_continuous_segments"] == 2
        assert meta["longest_segment_duration_s"] == pytest.approx(12.1, abs=0.5)
        # Should select the 12.1s segment
        assert meta["selected_segment_duration_s"] == pytest.approx(12.1, abs=0.5)
        assert_region_result_valid(result[region], expected_bpm=72.0, tol_bpm=7.0)


def test_estimate_multiregion_hr_multiple_usable_segments_selects_longest():
    """Test with three segments: 8.1s, 12.1s, 15.1s - should select 15.1s."""
    # Create custom packets: 8.1s valid, 2s gap, 12.1s valid, 2s gap, 15.1s valid
    fs = FS
    n1 = int(8.1 * fs)
    n_gap = int(2 * fs)
    n2 = int(12.1 * fs)
    n3 = int(15.1 * fs)
    total_n = n1 + n_gap + n2 + n_gap + n3
    t = np.arange(total_n, dtype=np.float64) / fs
    f0 = 72.0 / 60.0

    pulse = np.sin(2 * np.pi * f0 * t + 0.2)
    r_base = 140.0 * (1.0 + 0.004 * pulse) + 0.01 * t
    g_base = 120.0 * (1.0 + 0.010 * np.sin(2 * np.pi * f0 * t + 0.05)) + 0.015 * t
    b_base = 100.0 * (1.0 + 0.003 * np.sin(2 * np.pi * f0 * t + 0.35)) + 0.008 * t

    packets = []
    for i in range(total_n):
        ts = float(t[i])
        # Gap regions
        in_gap1 = n1 <= i < n1 + n_gap
        in_gap2 = n1 + n_gap + n2 <= i < n1 + n_gap + n2 + n_gap
        tracking_ok = not (in_gap1 or in_gap2)

        pkt = {
            "timestamp": ts,
            "wall_time": f"2024-01-01T00:00:{int(ts):02d}.{int((ts%1)*1000):03d}Z",
            "tracking_status": tracking_ok,
            "motion_displacement": 0.1 * np.sin(2 * np.pi * 0.5 * ts) if tracking_ok else None,
            "brightness": 150.0 + 2.0 * ts if tracking_ok else None,
            "regions": None,
        }

        if tracking_ok:
            regions = {}
            for region_name in ("forehead", "left_cheek", "right_cheek"):
                region_seed = hash((region_name, i, 11)) % 2**32
                region_rng = np.random.default_rng(region_seed)
                r = r_base[i] + region_rng.normal(0.0, 0.5)
                g = g_base[i] + region_rng.normal(0.0, 0.5)
                b = b_base[i] + region_rng.normal(0.0, 0.5)

                regions[region_name] = {
                    "rgb": {"r": float(r), "g": float(g), "b": float(b)},
                    "brightness": float(150.0 + 2.0 * ts + region_rng.normal(0.0, 1.0)),
                    "pixel_count": 1000 + region_rng.integers(-50, 50),
                }
            pkt["regions"] = regions

        packets.append(pkt)

    result = estimate_multiregion_hr(packets)

    for region in ("forehead", "left_cheek", "right_cheek"):
        meta = result[region]["metadata"]
        assert meta["num_continuous_segments"] == 3
        assert meta["longest_segment_duration_s"] == pytest.approx(15.1, abs=0.5)
        assert meta["selected_segment_duration_s"] == pytest.approx(15.1, abs=0.5)
        assert_region_result_valid(result[region], expected_bpm=72.0, tol_bpm=7.0)


def test_estimate_multiregion_hr_no_interpolation_across_gaps():
    """Verify that interpolation does NOT bridge tracking gaps.

    Create a signal with a known step change across a gap.
    If interpolation bridged the gap, the step would be smoothed.
    """
    fs = FS
    # 5s at baseline, 5s gap, 5s at elevated level
    n_seg = int(5 * fs)
    n_gap = int(5 * fs)
    total_n = n_seg + n_gap + n_seg
    t = np.arange(total_n, dtype=np.float64) / fs

    # Baseline level
    r_base = np.concatenate([
        np.full(n_seg, 140.0),
        np.full(n_gap, np.nan),  # gap
        np.full(n_seg, 160.0),   # step up by 20
    ])
    g_base = np.concatenate([
        np.full(n_seg, 120.0),
        np.full(n_gap, np.nan),
        np.full(n_seg, 140.0),
    ])
    b_base = np.concatenate([
        np.full(n_seg, 100.0),
        np.full(n_gap, np.nan),
        np.full(n_seg, 120.0),
    ])

    packets = []
    for i in range(total_n):
        ts = float(t[i])
        in_gap = n_seg <= i < n_seg + n_gap
        tracking_ok = not in_gap

        pkt = {
            "timestamp": ts,
            "wall_time": f"2024-01-01T00:00:{int(ts):02d}.{int((ts%1)*1000):03d}Z",
            "tracking_status": tracking_ok,
            "motion_displacement": 0.0 if tracking_ok else None,
            "brightness": 150.0 if tracking_ok else None,
            "regions": None,
        }

        if tracking_ok:
            regions = {}
            for region_name in ("forehead", "left_cheek", "right_cheek"):
                regions[region_name] = {
                    "rgb": {"r": float(r_base[i]), "g": float(g_base[i]), "b": float(b_base[i])},
                    "brightness": 150.0,
                    "pixel_count": 1000,
                }
            pkt["regions"] = regions

        packets.append(pkt)

    result = estimate_multiregion_hr(packets)

    for region in ("forehead", "left_cheek", "right_cheek"):
        meta = result[region]["metadata"]
        assert meta["num_continuous_segments"] == 2
        # Each segment is 5s < 10s min_duration_s, so should be invalid
        assert_region_result_invalid(result[region], "insufficient continuous signal")
        assert meta["selected_segment_duration_s"] == 0.0


def test_estimate_multiregion_hr_invalid_data_no_fabricated_bpm():
    """Invalid data must not produce fabricated BPM values."""
    # All tracking failed
    packets = make_packets(bpm=72.0, seed=14, tracking_failures=list(range(N_SAMPLES)))
    result = estimate_multiregion_hr(packets)

    for region in ("forehead", "left_cheek", "right_cheek"):
        for algo in ("green", "chrom", "pos"):
            res = result[region][algo]
            assert res["valid"] is False
            assert res["bpm"] is None
            assert res["frequency_hz"] is None
            assert "reason" in res


def test_estimate_multiregion_hr_non_monotonic_timestamps_handled():
    """Test handling of non-monotonic timestamps (should be caught by sampling)."""
    packets = make_packets(bpm=72.0, seed=15)
    # Make timestamps non-monotonic by swapping two
    packets[10]["timestamp"], packets[11]["timestamp"] = packets[11]["timestamp"], packets[10]["timestamp"]

    result = estimate_multiregion_hr(packets)

    # Should handle gracefully - either invalid result or still work if valid samples are monotonic
    for region in ("forehead", "left_cheek", "right_cheek"):
        assert region in result
        assert "green" in result[region]
        assert "chrom" in result[region]
        assert "pos" in result[region]
        assert "metadata" in result[region]


def test_estimate_multiregion_hr_custom_frequency_band():
    """Test with custom HR frequency band."""
    packets = make_packets(bpm=66.0, seed=16)  # 66 BPM = 1.1 Hz
    result = estimate_multiregion_hr(packets, low_hz=0.9, high_hz=1.4)

    for region in ("forehead", "left_cheek", "right_cheek"):
        assert_region_result_valid(result[region], expected_bpm=66.0, tol_bpm=6.0)
        for algo in ("green", "chrom", "pos"):
            freq = result[region][algo]["frequency_hz"]
            assert 0.9 <= freq <= 1.4


def test_estimate_multiregion_hr_all_tracking_failed():
    """Test when tracking_status is False for all packets."""
    packets = []
    for i in range(N_SAMPLES):
        packets.append({
            "timestamp": float(i / FS),
            "wall_time": None,
            "tracking_status": False,
            "motion_displacement": None,
            "brightness": None,
            "regions": None,
        })

    result = estimate_multiregion_hr(packets)

    for region in ("forehead", "left_cheek", "right_cheek"):
        assert_region_result_invalid(result[region])
        assert result[region]["metadata"]["valid_samples"] == 0
        assert result[region]["metadata"]["tracking_fraction"] == 0.0
        assert result[region]["metadata"]["num_continuous_segments"] == 0


def test_estimate_multiregion_hr_one_region_failing_does_not_crash_others():
    """If one region has no valid data, others should still process."""
    # Make forehead completely missing (tracking fails for all frames)
    packets = make_packets(bpm=72.0, seed=12, tracking_failures=list(range(N_SAMPLES)))
    result = estimate_multiregion_hr(packets)

    # Forehead should be invalid
    assert_region_result_invalid(result["forehead"])

    # But other regions should also be invalid since tracking failed for all
    # Let's test a different scenario: only forehead region missing
    packets2 = make_packets(bpm=72.0, seed=13, missing_regions={"forehead": list(range(N_SAMPLES))})
    result2 = estimate_multiregion_hr(packets2)

    # Forehead has no valid samples
    assert_region_result_invalid(result2["forehead"], "no valid tracking samples")

    # Other regions should work fine
    assert_region_result_valid(result2["left_cheek"], expected_bpm=72.0)
    assert_region_result_valid(result2["right_cheek"], expected_bpm=72.0)


def test_estimate_multiregion_hr_metadata_fields_complete():
    """Test that all required metadata fields are present."""
    packets = make_packets(bpm=72.0, seed=17, noise_std=0.05)
    result = estimate_multiregion_hr(packets)

    for region in ("forehead", "left_cheek", "right_cheek"):
        meta = result[region]["metadata"]
        required_fields = {
            "valid_samples", "missing_samples", "sampling_rate_hz",
            "tracking_fraction", "motion_mean", "brightness_mean",
            "num_continuous_segments", "longest_segment_duration_s",
            "selected_segment_duration_s", "gap_warning"
        }
        assert required_fields.issubset(meta.keys())
        assert isinstance(meta["valid_samples"], int)
        assert isinstance(meta["missing_samples"], int)
        assert isinstance(meta["num_continuous_segments"], int)
        assert isinstance(meta["longest_segment_duration_s"], float)
        assert isinstance(meta["selected_segment_duration_s"], float)


# Need to import Optional for type hints
from typing import Optional