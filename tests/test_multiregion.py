"""Tests for signal_processing.multiregion module."""

import numpy as np
import pytest

from signal_processing.multiregion import (
    estimate_multiregion_hr,
    estimate_multiregion_hr_advanced,
    get_baseline_3_region_config,
    get_8_region_config,
    get_adaptive_8_region_config,
    RegionConfig,
    AggregationConfig,
    ALL_REGIONS,
    BASELINE_3_REGIONS,
    _extract_region_data_from_packets,
    _extract_region_data_from_timeseries,
    _split_into_continuous_segments,
    _aggregate_region_results,
    _compute_method_comparison,
    _compute_region_comparison,
    _adaptive_region_selection,
)


# =============================================================================
# Fixtures and synthetic data generators
# =============================================================================

FS = 30.0
DURATION_S = 12.0  # >= default min_duration_s (10 s)
N_SAMPLES = int(FS * DURATION_S)

ALL_8_REGIONS = list(ALL_REGIONS)
BASELINE_3 = list(BASELINE_3_REGIONS)


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
    """Generate synthetic camera packets in Member 1's format (8 regions)."""
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
            for region_name in ALL_8_REGIONS:
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


def make_timeseries_data(
    bpm: float = 72.0,
    fs: float = FS,
    duration_s: float = DURATION_S,
    noise_std: float = 0.0,
    seed: int = 0,
    regions: Optional[list] = None,
) -> dict:
    """Generate synthetic time-series format data."""
    rng = np.random.default_rng(seed)
    n = int(np.round(duration_s * fs))
    t = np.arange(n, dtype=np.float64) / fs
    f0 = bpm / 60.0

    pulse = np.sin(2 * np.pi * f0 * t + 0.2)
    r_base = 140.0 * (1.0 + 0.004 * pulse) + 0.01 * t
    g_base = 120.0 * (1.0 + 0.010 * np.sin(2 * np.pi * f0 * t + 0.05)) + 0.015 * t
    b_base = 100.0 * (1.0 + 0.003 * np.sin(2 * np.pi * f0 * t + 0.35)) + 0.008 * t

    if noise_std > 0:
        r_base += rng.normal(0.0, noise_std, size=n)
        g_base += rng.normal(0.0, noise_std, size=n)
        b_base += rng.normal(0.0, noise_std, size=n)

    region_list = regions or ALL_8_REGIONS
    rgb_dict = {}
    for region_name in region_list:
        region_seed = hash((region_name, seed)) % 2**32
        region_rng = np.random.default_rng(region_seed)
        r = r_base + region_rng.normal(0.0, 0.5, size=n)
        g = g_base + region_rng.normal(0.0, 0.5, size=n)
        b = b_base + region_rng.normal(0.0, 0.5, size=n)
        rgb_dict[region_name] = np.column_stack([r, g, b])

    motion = 0.1 * np.sin(2 * np.pi * 0.5 * t)
    brightness = 150.0 + 2.0 * t
    track_ok = np.ones(n, dtype=bool)

    return {
        "t": t,
        "rgb": rgb_dict,
        "motion": motion,
        "brightness": brightness,
        "track_ok": track_ok,
    }


def make_packets_with_gap(
    bpm: float = 72.0,
    fs: float = FS,
    gap_start_s: float = 10.0,
    gap_duration_s: float = 5.0,
    total_duration_s: float = 25.0,
    seed: int = 0,
) -> list:
    """Create packets with a specific tracking gap."""
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
            for region_name in ALL_8_REGIONS:
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


# =============================================================================
# Tests for helper functions
# =============================================================================

class TestHelperFunctions:
    def test_extract_region_data_from_packets(self):
        packets = make_packets(bpm=72.0, seed=1)
        t, rgb, motion, brightness, track_ok, total, valid = _extract_region_data_from_packets(
            packets, "forehead_center"
        )
        assert t.size == N_SAMPLES
        assert rgb.shape == (N_SAMPLES, 3)
        assert valid == N_SAMPLES
        assert np.all(track_ok)

    def test_extract_region_data_from_timeseries(self):
        data = make_timeseries_data(bpm=72.0, seed=1)
        t, rgb, motion, brightness, track_ok, total, valid = _extract_region_data_from_timeseries(
            data, "forehead_center"
        )
        assert t.size == N_SAMPLES
        assert rgb.shape == (N_SAMPLES, 3)
        assert valid == N_SAMPLES
        assert np.all(track_ok)

    def test_extract_region_data_missing_region(self):
        data = make_timeseries_data(bpm=72.0, seed=1, regions=["forehead_center"])
        t, rgb, motion, brightness, track_ok, total, valid = _extract_region_data_from_timeseries(
            data, "left_cheek_upper"
        )
        assert valid == 0
        assert np.all(np.isnan(rgb))

    def test_split_into_continuous_segments_no_gap(self):
        t = np.arange(100) / 30.0
        rgb = np.random.rand(100, 3)
        motion = np.random.rand(100)
        brightness = np.random.rand(100)
        segments = _split_into_continuous_segments(t, rgb, motion, brightness)
        assert len(segments) == 1
        assert segments[0]["num_samples"] == 100

    def test_split_into_continuous_segments_with_gap(self):
        t = np.concatenate([np.arange(100), np.arange(200, 300)]) / 30.0
        rgb = np.random.rand(200, 3)
        motion = np.random.rand(200)
        brightness = np.random.rand(200)
        segments = _split_into_continuous_segments(t, rgb, motion, brightness)
        assert len(segments) == 2
        assert segments[0]["num_samples"] == 100
        assert segments[1]["num_samples"] == 100

    def test_aggregate_region_results_median(self):
        region_results = {
            "r1": {"green": {"valid": True, "bpm": 70.0}, "chrom": {"valid": True, "bpm": 72.0}, "pos": {"valid": True, "bpm": 74.0}},
            "r2": {"green": {"valid": True, "bpm": 71.0}, "chrom": {"valid": True, "bpm": 73.0}, "pos": {"valid": True, "bpm": 75.0}},
            "r3": {"green": {"valid": True, "bpm": 69.0}, "chrom": {"valid": True, "bpm": 71.0}, "pos": {"valid": True, "bpm": 73.0}},
        }
        agg, used, excluded = _aggregate_region_results(region_results, min_valid_regions=2)
        assert agg == pytest.approx(72.0, abs=1.0)  # median of region medians
        assert len(used) == 3
        assert len(excluded) == 0

    def test_aggregate_region_results_insufficient(self):
        region_results = {
            "r1": {"green": {"valid": True, "bpm": 70.0}, "chrom": {"valid": False}, "pos": {"valid": False}},
        }
        agg, used, excluded = _aggregate_region_results(region_results, min_valid_regions=2)
        assert agg is None
        assert len(used) == 0
        assert "r1" in excluded

    def test_compute_method_comparison(self):
        region_results = {
            "r1": {"green": {"valid": True, "bpm": 70.0}, "chrom": {"valid": True, "bpm": 72.0}, "pos": {"valid": True, "bpm": 74.0}},
            "r2": {"green": {"valid": True, "bpm": 71.0}, "chrom": {"valid": True, "bpm": 73.0}, "pos": {"valid": True, "bpm": 75.0}},
        }
        comp = _compute_method_comparison(region_results)
        assert comp["green"]["valid_count"] == 2
        assert comp["green"]["mean_bpm"] == pytest.approx(70.5)
        assert comp["chrom"]["median_bpm"] == pytest.approx(72.5)

    def test_compute_region_comparison(self):
        region_results = {
            "r1": {"green": {"valid": True, "bpm": 70.0}, "chrom": {"valid": True, "bpm": 72.0}, "pos": {"valid": True, "bpm": 74.0}},
            "r2": {"green": {"valid": True, "bpm": 71.0}, "chrom": {"valid": True, "bpm": 73.0}, "pos": {"valid": True, "bpm": 75.0}},
        }
        comp = _compute_region_comparison(region_results)
        assert comp["r1"]["valid_methods"] == 3
        assert comp["r1"]["median_bpm"] == pytest.approx(72.0)

    def test_adaptive_region_selection(self):
        region_results = {
            "good": {
                "green": {"valid": True, "bpm": 72.0},
                "chrom": {"valid": True, "bpm": 73.0},
                "pos": {"valid": True, "bpm": 71.0},
                "metadata": {"quality_metrics": {
                    "snr": {"snr_db": 5.0},
                    "spectral_concentration": {"concentration": 0.5},
                    "spectral_peak_prominence": {"prominence": 1.0},
                }}
            },
            "bad_snr": {
                "green": {"valid": True, "bpm": 72.0},
                "chrom": {"valid": True, "bpm": 73.0},
                "pos": {"valid": True, "bpm": 71.0},
                "metadata": {"quality_metrics": {
                    "snr": {"snr_db": -5.0},
                    "spectral_concentration": {"concentration": 0.5},
                    "spectral_peak_prominence": {"prominence": 1.0},
                }}
            },
            "no_valid": {
                "green": {"valid": False, "bpm": None},
                "chrom": {"valid": False, "bpm": None},
                "pos": {"valid": False, "bpm": None},
            },
        }
        config = RegionConfig(adaptive_mode=True, adaptive_min_snr_db=0.0, adaptive_min_concentration=0.3)
        selected, excluded = _adaptive_region_selection(region_results, config)
        assert "good" in selected
        assert "bad_snr" in excluded
        assert "no_valid" in excluded


# =============================================================================
# Legacy API tests (backward compatibility)
# =============================================================================

class TestLegacyAPI:
    def test_estimate_multiregion_hr_valid_packets_all_regions(self):
        """All three legacy regions have valid data, all three algorithms run."""
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
    def test_estimate_multiregion_hr_known_heart_rates(self, expected_bpm):
        """Test known heart rates at 60, 72, 90 BPM."""
        packets = make_packets(bpm=expected_bpm, seed=42, noise_std=0.02)
        result = estimate_multiregion_hr(packets)

        for region in ("forehead", "left_cheek", "right_cheek"):
            assert_region_result_valid(result[region], expected_bpm=expected_bpm, tol_bpm=6.0)

    def test_estimate_multiregion_hr_noisy_signal(self):
        """Test with moderate noise."""
        packets = make_packets(bpm=72.0, seed=2, noise_std=0.3)
        result = estimate_multiregion_hr(packets)

        for region in ("forehead", "left_cheek", "right_cheek"):
            assert_region_result_valid(result[region], expected_bpm=72.0, tol_bpm=8.0)

    def test_estimate_multiregion_hr_out_of_band_components(self):
        """Test with strong out-of-band components."""
        packets = make_packets(bpm=78.0, seed=3, noise_std=0.05, out_of_band=True)
        result = estimate_multiregion_hr(packets)

        for region in ("forehead", "left_cheek", "right_cheek"):
            assert_region_result_valid(result[region], expected_bpm=78.0, tol_bpm=8.0)

    def test_estimate_multiregion_hr_missing_tracking_packets(self):
        """Test with intermittent tracking failures (short gaps that don't split)."""
        tracking_failures = list(range(100, 103))
        packets = make_packets(bpm=72.0, seed=4, tracking_failures=tracking_failures)
        result = estimate_multiregion_hr(packets)

        for region in ("forehead", "left_cheek", "right_cheek"):
            assert_region_result_valid(result[region], expected_bpm=72.0, tol_bpm=7.0)
            meta = result[region]["metadata"]
            assert meta["missing_samples"] == len(tracking_failures)
            assert meta["tracking_fraction"] < 1.0
            assert meta["num_continuous_segments"] == 1
            assert meta["gap_warning"] is None

    def test_estimate_multiregion_hr_missing_region_measurements(self):
        """Test with specific region missing on some frames."""
        missing_regions = {"forehead_center": list(range(50, 53))}
        packets = make_packets(bpm=72.0, seed=5, missing_regions=missing_regions)
        result = estimate_multiregion_hr(packets)

        assert_region_result_valid(result["forehead"], expected_bpm=72.0, tol_bpm=7.0)
        assert result["forehead"]["metadata"]["missing_samples"] >= 3
        assert result["forehead"]["metadata"]["num_continuous_segments"] == 1

        assert_region_result_valid(result["left_cheek"], expected_bpm=72.0)
        assert_region_result_valid(result["right_cheek"], expected_bpm=72.0)

    def test_estimate_multiregion_hr_invalid_rgb_values(self):
        """Test with NaN/Inf in RGB values."""
        packets = make_packets(bpm=72.0, seed=6)
        packets[10]["regions"]["forehead_center"]["rgb"]["r"] = np.nan
        packets[11]["regions"]["forehead_center"]["rgb"]["g"] = np.inf
        packets[12]["regions"]["forehead_center"]["rgb"]["b"] = -np.inf

        result = estimate_multiregion_hr(packets)

        assert_region_result_valid(result["forehead"], expected_bpm=72.0, tol_bpm=7.0)
        assert result["forehead"]["metadata"]["missing_samples"] >= 3

    def test_estimate_multiregion_hr_empty_input(self):
        """Test with empty packet list."""
        result = estimate_multiregion_hr([])

        for region in ("forehead", "left_cheek", "right_cheek"):
            assert_region_result_invalid(result[region], "no valid tracking samples")
            assert result[region]["metadata"]["valid_samples"] == 0

    def test_estimate_multiregion_hr_insufficient_duration(self):
        """Test with signal duration less than min_duration_s."""
        packets = make_packets(bpm=72.0, duration_s=5.0, seed=7)
        result = estimate_multiregion_hr(packets)

        for region in ("forehead", "left_cheek", "right_cheek"):
            assert_region_result_invalid(result[region], "insufficient continuous signal")

    def test_estimate_multiregion_hr_large_tracking_gap_splits_signal(self):
        """Test that a large tracking gap (5s) splits the signal into segments."""
        packets = make_packets_with_gap(
            bpm=72.0, gap_start_s=10.1, gap_duration_s=5.0, total_duration_s=25.2, seed=8
        )
        result = estimate_multiregion_hr(packets)

        for region in ("forehead", "left_cheek", "right_cheek"):
            meta = result[region]["metadata"]
            assert meta["num_continuous_segments"] == 2
            assert meta["longest_segment_duration_s"] == pytest.approx(10.1, abs=0.5)
            assert meta["gap_warning"] is not None
            assert "split into 2 continuous segments" in meta["gap_warning"]
            assert meta["selected_segment_duration_s"] == pytest.approx(10.1, abs=0.5)
            assert_region_result_valid(result[region], expected_bpm=72.0, tol_bpm=7.0)

    def test_estimate_multiregion_hr_two_segments_neither_long_enough(self):
        """Test with two segments, neither meeting min_duration_s."""
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

    def test_estimate_multiregion_hr_one_long_one_short_segment(self):
        """Test with one long segment (12.1s) and one short segment (3s)."""
        packets = make_packets_with_gap(
            bpm=72.0, gap_start_s=12.1, gap_duration_s=5.0, total_duration_s=20.1, seed=10
        )
        result = estimate_multiregion_hr(packets)

        for region in ("forehead", "left_cheek", "right_cheek"):
            meta = result[region]["metadata"]
            assert meta["num_continuous_segments"] == 2
            assert meta["longest_segment_duration_s"] == pytest.approx(12.1, abs=0.5)
            assert meta["selected_segment_duration_s"] == pytest.approx(12.1, abs=0.5)
            assert_region_result_valid(result[region], expected_bpm=72.0, tol_bpm=7.0)

    def test_estimate_multiregion_hr_multiple_usable_segments_selects_longest(self):
        """Test with three segments: 8.1s, 12.1s, 15.1s - should select 15.1s."""
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
                for region_name in ALL_8_REGIONS:
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

    def test_estimate_multiregion_hr_no_interpolation_across_gaps(self):
        """Verify that interpolation does NOT bridge tracking gaps."""
        fs = FS
        n_seg = int(5 * fs)
        n_gap = int(5 * fs)
        total_n = n_seg + n_gap + n_seg
        t = np.arange(total_n, dtype=np.float64) / fs

        r_base = np.concatenate([
            np.full(n_seg, 140.0),
            np.full(n_gap, np.nan),
            np.full(n_seg, 160.0),
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
                for region_name in ALL_8_REGIONS:
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
            assert_region_result_invalid(result[region], "insufficient continuous signal")
            assert meta["selected_segment_duration_s"] == 0.0

    def test_estimate_multiregion_hr_invalid_data_no_fabricated_bpm(self):
        """Invalid data must not produce fabricated BPM values."""
        packets = make_packets(bpm=72.0, seed=14, tracking_failures=list(range(N_SAMPLES)))
        result = estimate_multiregion_hr(packets)

        for region in ("forehead", "left_cheek", "right_cheek"):
            for algo in ("green", "chrom", "pos"):
                res = result[region][algo]
                assert res["valid"] is False
                assert res["bpm"] is None
                assert res["frequency_hz"] is None
                assert "reason" in res


# =============================================================================
# Advanced API tests (new functionality)
# =============================================================================

class TestAdvancedAPI:
    def test_estimate_multiregion_hr_advanced_baseline_3_region(self):
        """Three-region baseline configuration."""
        packets = make_packets(bpm=72.0, seed=1, noise_std=0.05)
        config = get_baseline_3_region_config()
        result = estimate_multiregion_hr_advanced(packets, config)

        assert result["aggregated_bpm"] is not None
        assert result["aggregation_method"] == "median"
        assert set(result["regions_requested"]) == set(BASELINE_3)
        assert len(result["regions_available"]) == 3
        assert len(result["regions_used"]) == 3
        assert "region_results" in result
        assert "method_comparison" in result
        assert "region_comparison" in result
        assert "metadata" in result

        for region in BASELINE_3:
            assert region in result["region_results"]
            assert_region_result_valid(result["region_results"][region], expected_bpm=72.0)

    def test_estimate_multiregion_hr_advanced_8_region(self):
        """Eight-region configuration."""
        packets = make_packets(bpm=72.0, seed=1, noise_std=0.05)
        config = get_8_region_config()
        result = estimate_multiregion_hr_advanced(packets, config)

        assert result["aggregated_bpm"] is not None
        assert set(result["regions_requested"]) == set(ALL_8_REGIONS)
        assert len(result["regions_available"]) == 8
        assert len(result["regions_used"]) == 8

        for region in ALL_8_REGIONS:
            assert region in result["region_results"]
            assert_region_result_valid(result["region_results"][region], expected_bpm=72.0)

    def test_estimate_multiregion_hr_advanced_custom_region_selection(self):
        """Custom region selection."""
        packets = make_packets(bpm=72.0, seed=1, noise_std=0.05)
        config = RegionConfig(regions=("forehead_center", "nose_bridge"), min_valid_regions=1)
        result = estimate_multiregion_hr_advanced(packets, config)

        assert result["aggregated_bpm"] is not None
        assert set(result["regions_requested"]) == {"forehead_center", "nose_bridge"}
        assert len(result["regions_available"]) == 2
        assert len(result["regions_used"]) == 2

    def test_estimate_multiregion_hr_advanced_timeseries_format(self):
        """Time-series format input."""
        data = make_timeseries_data(bpm=72.0, seed=1, noise_std=0.05)
        config = get_8_region_config()
        result = estimate_multiregion_hr_advanced(data, config)

        assert result["aggregated_bpm"] is not None
        assert len(result["regions_available"]) == 8

    def test_estimate_multiregion_hr_advanced_missing_regions(self):
        """Some regions missing from data."""
        data = make_timeseries_data(bpm=72.0, seed=1, regions=["forehead_center", "left_cheek_upper"])
        config = get_8_region_config()
        result = estimate_multiregion_hr_advanced(data, config)

        assert len(result["regions_available"]) == 2
        assert len(result["regions_used"]) == 2
        assert "forehead_center" in result["regions_available"]
        assert "left_cheek_upper" in result["regions_available"]
        assert "nose_bridge" in result["regions_excluded"]

    def test_estimate_multiregion_hr_advanced_none_rgb_values(self):
        """None RGB values in time-series data."""
        data = make_timeseries_data(bpm=72.0, seed=1)
        data["rgb"]["forehead_center"] = None
        config = get_8_region_config()
        result = estimate_multiregion_hr_advanced(data, config)

        assert "forehead_center" in result["regions_excluded"]
        assert len(result["regions_available"]) == 7

    def test_estimate_multiregion_hr_advanced_nan_inf_rgb(self):
        """NaN/Inf RGB values in time-series data."""
        data = make_timeseries_data(bpm=72.0, seed=1)
        data["rgb"]["forehead_center"][10, 0] = np.nan
        data["rgb"]["forehead_center"][11, 1] = np.inf
        data["rgb"]["forehead_center"][12, 2] = -np.inf
        config = get_8_region_config()
        result = estimate_multiregion_hr_advanced(data, config)

        # Should handle gracefully - those frames become missing
        assert "forehead_center" in result["regions_available"]

    def test_estimate_multiregion_hr_advanced_invalid_timestamps(self):
        """Non-monotonic or invalid timestamps."""
        data = make_timeseries_data(bpm=72.0, seed=1)
        data["t"][100] = data["t"][99]  # non-monotonic
        data["t"][101] = np.nan
        config = get_8_region_config()
        result = estimate_multiregion_hr_advanced(data, config)

        # Should handle gracefully
        assert "aggregated_bpm" in result

    def test_estimate_multiregion_hr_advanced_tracking_failures(self):
        """Tracking failures in time-series data."""
        data = make_timeseries_data(bpm=72.0, seed=1)
        data["track_ok"][100:103] = False  # 3 frames tracking failure (0.1s at 30Hz) - below 5x median dt threshold
        config = get_8_region_config()
        result = estimate_multiregion_hr_advanced(data, config)

        # Should still work with short gap
        assert result["aggregated_bpm"] is not None

    def test_estimate_multiregion_hr_advanced_long_tracking_gaps(self):
        """Long tracking gaps that split signal."""
        data = make_timeseries_data(bpm=72.0, seed=1, duration_s=25.0)
        # Create 5s gap in middle
        gap_start = int(10.0 * FS)
        gap_end = int(15.0 * FS)
        data["track_ok"][gap_start:gap_end] = False
        config = get_8_region_config()
        result = estimate_multiregion_hr_advanced(data, config)

        # Should split into segments, select longest
        for region in result["regions_available"]:
            meta = result["region_results"][region]["metadata"]
            assert meta["num_continuous_segments"] >= 2

    def test_estimate_multiregion_hr_advanced_insufficient_continuous_duration(self):
        """Signal duration less than min_duration_s."""
        data = make_timeseries_data(bpm=72.0, seed=1, duration_s=5.0)
        config = get_8_region_config()
        result = estimate_multiregion_hr_advanced(data, config)

        assert result["aggregated_bpm"] is None
        for region in ALL_8_REGIONS:
            assert_region_result_invalid(result["region_results"][region], "insufficient continuous signal")

    def test_estimate_multiregion_hr_advanced_one_region_fails_others_succeed(self):
        """One region failing while others succeed."""
        data = make_timeseries_data(bpm=72.0, seed=1)
        # Make forehead_center all NaN
        data["rgb"]["forehead_center"][:] = np.nan
        config = get_8_region_config()
        result = estimate_multiregion_hr_advanced(data, config)

        assert "forehead_center" in result["regions_excluded"]
        assert len(result["regions_available"]) == 7
        assert result["aggregated_bpm"] is not None

    def test_estimate_multiregion_hr_advanced_median_aggregation(self):
        """Median aggregation of valid region estimates."""
        packets = make_packets(bpm=72.0, seed=1, noise_std=0.05)
        config = get_8_region_config()
        agg_config = AggregationConfig(method="median", min_valid_regions=2)
        result = estimate_multiregion_hr_advanced(packets, config, agg_config)

        assert result["aggregated_bpm"] is not None
        assert result["aggregation_method"] == "median"
        assert result["metadata"]["num_regions_used"] == 8

    def test_estimate_multiregion_hr_advanced_minimum_region_count(self):
        """Minimum valid region count for aggregation."""
        packets = make_packets(bpm=72.0, seed=1, noise_std=0.05)
        config = get_8_region_config()
        agg_config = AggregationConfig(min_valid_regions=5)
        result = estimate_multiregion_hr_advanced(packets, config, agg_config)

        assert result["aggregated_bpm"] is not None
        assert result["metadata"]["num_regions_used"] >= 5

    def test_estimate_multiregion_hr_advanced_no_valid_regions(self):
        """No valid regions - all tracking failed."""
        packets = make_packets(bpm=72.0, seed=1, tracking_failures=list(range(N_SAMPLES)))
        config = get_8_region_config()
        result = estimate_multiregion_hr_advanced(packets, config)

        assert result["aggregated_bpm"] is None
        assert len(result["regions_available"]) == 0
        assert len(result["regions_used"]) == 0

    def test_estimate_multiregion_hr_advanced_adaptive_selection(self):
        """Adaptive region selection (experimental)."""
        packets = make_packets(bpm=72.0, seed=1, noise_std=0.05)
        config = get_adaptive_8_region_config(min_snr_db=0.0, min_concentration=0.3)
        result = estimate_multiregion_hr_advanced(packets, config)

        assert result["metadata"]["adaptive_mode"] is True
        assert "regions_excluded" in result
        assert result["aggregated_bpm"] is not None

    def test_estimate_multiregion_hr_advanced_adaptive_selected_excluded_reporting(self):
        """Adaptive selection reports selected and excluded regions."""
        packets = make_packets(bpm=72.0, seed=1, noise_std=0.05)
        config = get_adaptive_8_region_config(min_snr_db=100.0)  # Impossible threshold
        result = estimate_multiregion_hr_advanced(packets, config)

        # With impossible threshold, adaptive excludes all, NO fallback
        assert "regions_excluded" in result
        assert len(result["regions_used"]) == 0  # No fallback - regions_used empty
        assert result["aggregated_bpm"] is None  # No HR fabricated
        assert result["metadata"]["adaptive_failure_reason"] is not None
        assert "adaptive selection" in result["metadata"]["adaptive_failure_reason"]
        # All 8 regions should be in excluded
        assert len(result["regions_excluded"]) == 8

    def test_estimate_multiregion_hr_advanced_adaptive_all_regions_fail(self):
        """All regions fail adaptive selection - aggregated_bpm=None, excluded preserved."""
        packets = make_packets(bpm=72.0, seed=1, noise_std=0.05)
        config = get_adaptive_8_region_config(min_snr_db=50.0, min_concentration=0.9)
        result = estimate_multiregion_hr_advanced(packets, config)

        assert result["aggregated_bpm"] is None
        assert len(result["regions_used"]) == 0
        assert len(result["regions_excluded"]) == 8
        assert result["metadata"]["adaptive_failure_reason"] is not None

    def test_estimate_multiregion_hr_advanced_adaptive_one_region_passes_min_two(self):
        """Only one region passes adaptive selection when min_valid_regions=2."""
        packets = make_packets(bpm=72.0, seed=1, noise_std=0.05)
        # Use very high thresholds to ensure only 1 region passes
        config = get_adaptive_8_region_config(min_snr_db=10.0, min_concentration=0.8)
        agg_config = AggregationConfig(min_valid_regions=2)
        result = estimate_multiregion_hr_advanced(packets, config, agg_config)

        assert result["aggregated_bpm"] is None
        assert len(result["regions_used"]) <= 1  # At most one passed
        assert len(result["regions_excluded"]) >= 7
        assert result["metadata"]["adaptive_failure_reason"] is not None

    def test_estimate_multiregion_hr_advanced_adaptive_exactly_min_regions_pass(self):
        """Exactly min_valid_regions pass adaptive selection - should succeed."""
        packets = make_packets(bpm=72.0, seed=1, noise_std=0.05)
        # Set threshold so exactly 2 regions pass
        config = get_adaptive_8_region_config(min_snr_db=1.0, min_concentration=0.2)
        agg_config = AggregationConfig(min_valid_regions=2)
        result = estimate_multiregion_hr_advanced(packets, config, agg_config)

        # Should succeed with exactly 2 regions
        assert result["aggregated_bpm"] is not None
        assert len(result["regions_used"]) >= 2
        assert result["metadata"]["adaptive_failure_reason"] is None

    def test_estimate_multiregion_hr_advanced_adaptive_failed_regions_not_in_aggregation(self):
        """Regions failing adaptive selection are excluded from aggregation."""
        packets = make_packets(bpm=72.0, seed=1, noise_std=0.05)
        config = get_adaptive_8_region_config(min_snr_db=100.0)  # All fail
        result = estimate_multiregion_hr_advanced(packets, config)

        # No regions used, so no aggregation
        assert result["aggregated_bpm"] is None
        assert len(result["regions_used"]) == 0
        # Excluded regions should be listed
        assert len(result["regions_excluded"]) == 8
        # Verify excluded regions are the ones that failed
        for region in ALL_8_REGIONS:
            assert region in result["regions_excluded"]

    def test_estimate_multiregion_hr_advanced_non_adaptive_still_works(self):
        """Non-adaptive mode (default) still works normally."""
        packets = make_packets(bpm=72.0, seed=1, noise_std=0.05)
        config = get_8_region_config()  # adaptive_mode=False
        result = estimate_multiregion_hr_advanced(packets, config)

        assert result["aggregated_bpm"] is not None
        assert len(result["regions_used"]) == 8
        assert result["metadata"]["adaptive_mode"] is False
        assert result["metadata"]["adaptive_failure_reason"] is None

    def test_estimate_multiregion_hr_advanced_method_comparison(self):
        """Method comparison across regions."""
        packets = make_packets(bpm=72.0, seed=1, noise_std=0.05)
        config = get_8_region_config()
        result = estimate_multiregion_hr_advanced(packets, config)

        comp = result["method_comparison"]
        assert "green" in comp
        assert "chrom" in comp
        assert "pos" in comp
        for method in ("green", "chrom", "pos"):
            assert comp[method]["valid_count"] == 8
            assert comp[method]["mean_bpm"] is not None

    def test_estimate_multiregion_hr_advanced_region_comparison(self):
        """Region comparison across methods."""
        packets = make_packets(bpm=72.0, seed=1, noise_std=0.05)
        config = get_8_region_config()
        result = estimate_multiregion_hr_advanced(packets, config)

        comp = result["region_comparison"]
        for region in ALL_8_REGIONS:
            assert region in comp
            assert comp[region]["valid_methods"] == 3
            assert comp[region]["median_bpm"] is not None

    def test_estimate_multiregion_hr_advanced_metadata(self):
        """Metadata includes sampling rate, duration, region counts."""
        packets = make_packets(bpm=72.0, seed=1, noise_std=0.05)
        config = get_8_region_config()
        result = estimate_multiregion_hr_advanced(packets, config)

        meta = result["metadata"]
        assert meta["sampling_rate_hz"] == pytest.approx(FS, rel=0.1)
        assert meta["recording_duration_s"] > 0
        assert meta["num_regions_requested"] == 8
        assert meta["num_regions_available"] == 8
        assert meta["num_regions_used"] == 8

    def test_estimate_multiregion_hr_advanced_region_results_structure(self):
        """Region results contain Green/CHROM/POS/aggregated_bpm."""
        packets = make_packets(bpm=72.0, seed=1, noise_std=0.05)
        config = get_8_region_config()
        result = estimate_multiregion_hr_advanced(packets, config)

        for region in ALL_8_REGIONS:
            r = result["region_results"][region]
            assert "green" in r
            assert "chrom" in r
            assert "pos" in r
            assert "aggregated_bpm" in r
            assert "metadata" in r
            assert r["aggregated_bpm"] is not None

    def test_estimate_multiregion_hr_advanced_invalid_input_handling(self):
        """Invalid input handling."""
        config = get_8_region_config()
        result = estimate_multiregion_hr_advanced([], config)

        assert result["aggregated_bpm"] is None
        assert len(result["regions_available"]) == 0

    def test_convenience_configs(self):
        """Test convenience configuration functions."""
        baseline = get_baseline_3_region_config()
        assert baseline.regions == BASELINE_3_REGIONS
        assert baseline.adaptive_mode is False

        eight = get_8_region_config()
        assert eight.regions == ALL_REGIONS
        assert eight.adaptive_mode is False

        adaptive = get_adaptive_8_region_config(min_snr_db=5.0)
        assert adaptive.regions == ALL_REGIONS
        assert adaptive.adaptive_mode is True
        assert adaptive.adaptive_min_snr_db == 5.0


# =============================================================================
# Integration tests
# =============================================================================

class TestIntegration:
    def test_three_region_vs_eight_region_comparison(self):
        """Research comparison: 3-region baseline vs 8-region."""
        packets = make_packets(bpm=72.0, seed=1, noise_std=0.05)

        baseline_result = estimate_multiregion_hr_advanced(packets, get_baseline_3_region_config())
        eight_result = estimate_multiregion_hr_advanced(packets, get_8_region_config())

        # Both should produce valid aggregated BPM
        assert baseline_result["aggregated_bpm"] is not None
        assert eight_result["aggregated_bpm"] is not None

        # 8-region should have more regions available
        assert eight_result["metadata"]["num_regions_available"] >= baseline_result["metadata"]["num_regions_available"]

    def test_api_usage_example(self):
        """Test the documented API usage example."""
        packets = make_packets(bpm=72.0, seed=1, noise_std=0.05)

        # Three-region baseline
        result = estimate_multiregion_hr_advanced(packets, get_baseline_3_region_config())
        assert result["aggregated_bpm"] is not None

        # Eight-region fixed
        result = estimate_multiregion_hr_advanced(packets, get_8_region_config())
        assert result["aggregated_bpm"] is not None

        # Custom region selection
        config = RegionConfig(regions=("forehead_center", "nose_bridge"), min_valid_regions=1)
        result = estimate_multiregion_hr_advanced(packets, config)
        assert result["aggregated_bpm"] is not None

        # Adaptive eight-region (experimental)
        result = estimate_multiregion_hr_advanced(packets, get_adaptive_8_region_config())
        assert result["aggregated_bpm"] is not None

        # Time-series format
        ts_data = make_timeseries_data(bpm=72.0, seed=1, noise_std=0.05)
        result = estimate_multiregion_hr_advanced(ts_data, get_8_region_config())
        assert result["aggregated_bpm"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])