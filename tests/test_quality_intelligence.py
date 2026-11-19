"""Tests for signal_processing.quality_intelligence module."""

import numpy as np
import pytest

from signal_processing.quality_intelligence import (
    QualityGate,
    QualityThresholds,
    evaluate_quality,
)
from signal_processing.multiregion import ALL_REGIONS


def make_synthetic_packet_stream(
    duration_s: float = 12.0,
    fs: float = 30.0,
    bpm: float = 72.0,
    motion_mean: float = 1.0,
    brightness_val: float = 120.0,
    tracking_prob: float = 1.0,
    corrupted_regions: list = None,
    seed: int = 42,
) -> list:
    """Generate synthetic camera frame packets for 8 regions."""
    rng = np.random.default_rng(seed)
    n_frames = int(duration_s * fs)
    t = np.arange(n_frames, dtype=np.float64) / fs
    f0 = bpm / 60.0
    pulse = np.sin(2 * np.pi * f0 * t)

    packets = []
    for i in range(n_frames):
        is_tracked = bool(rng.uniform() < tracking_prob)
        m_val = float(motion_mean + rng.normal(0, 0.2))

        regions_dict = {}
        rgb_dict = {}
        bright_dict = {}

        for r_name in ALL_REGIONS:
            if not is_tracked:
                regions_dict[r_name] = {"rgb": None, "brightness": None, "pixel_count": 0}
                rgb_dict[r_name] = None
                bright_dict[r_name] = None
            elif corrupted_regions and r_name in corrupted_regions:
                # Add intense noise to corrupted regions
                noise = rng.normal(0, 50.0)
                r_rgb = {"r": 100.0, "g": 100.0 + noise, "b": 100.0}
                regions_dict[r_name] = {
                    "rgb": r_rgb,
                    "brightness": 20.0,  # too dark
                    "pixel_count": 500,
                }
                rgb_dict[r_name] = r_rgb
                bright_dict[r_name] = 20.0
            else:
                g_val = brightness_val * (1.0 + 0.02 * pulse[i]) + rng.normal(0, 0.1)
                r_rgb = {"r": brightness_val, "g": g_val, "b": brightness_val}
                regions_dict[r_name] = {
                    "rgb": r_rgb,
                    "brightness": brightness_val,
                    "pixel_count": 500,
                }
                rgb_dict[r_name] = r_rgb
                bright_dict[r_name] = brightness_val

        packets.append({
            "timestamp": float(t[i]),
            "wall_time": "2026-11-19T00:00:00Z",
            "track_ok": is_tracked,
            "tracking_status": is_tracked,
            "motion": m_val,
            "motion_displacement": m_val,
            "brightness": brightness_val,
            "overall_brightness": brightness_val,
            "rgb": rgb_dict,
            "regions": regions_dict,
        })

    return packets


class TestQualityIntelligence:

    def test_clean_signal_accepted(self):
        """Test that clean 8-region synthetic data passes the quality gate."""
        packets = make_synthetic_packet_stream(duration_s=12.0, bpm=72.0)
        res = evaluate_quality(packets)

        assert res["accepted"] is True
        assert res["score"] is not None
        assert 0.0 <= res["score"] <= 1.0
        assert len(res["reasons"]) == 0
        assert len(res["selected_regions"]) >= 2
        assert len(res["rejected_regions"]) == 0
        assert res["confidence"] in ("HIGH", "MODERATE")

    def test_short_duration_rejected(self):
        """Test rejection when recording duration is below 10 seconds."""
        packets = make_synthetic_packet_stream(duration_s=5.0)
        res = evaluate_quality(packets)

        assert res["accepted"] is False
        assert res["score"] is None
        assert any("duration" in r.lower() for r in res["reasons"])

    def test_poor_tracking_rejected(self):
        """Test rejection when face tracking drops below 80%."""
        packets = make_synthetic_packet_stream(duration_s=12.0, tracking_prob=0.5)
        res = evaluate_quality(packets)

        assert res["accepted"] is False
        assert res["score"] is None
        assert any("tracking" in r.lower() for r in res["reasons"])

    def test_excessive_motion_rejected(self):
        """Test rejection when mean displacement exceeds threshold."""
        packets = make_synthetic_packet_stream(duration_s=12.0, motion_mean=8.5)
        res = evaluate_quality(packets)

        assert res["accepted"] is False
        assert res["score"] is None
        assert any("motion" in r.lower() for r in res["reasons"])

    def test_severe_underexposure_rejected(self):
        """Test rejection when lighting is below minimum brightness."""
        packets = make_synthetic_packet_stream(duration_s=12.0, brightness_val=15.0)
        res = evaluate_quality(packets)

        assert res["accepted"] is False
        assert res["score"] is None
        assert any("lighting" in r.lower() or "brightness" in r.lower() for r in res["reasons"])

    def test_adaptive_region_selection(self):
        """Test that corrupted regions are rejected while clean regions are retained."""
        corrupted = ["forehead_left", "forehead_right"]
        packets = make_synthetic_packet_stream(
            duration_s=12.0,
            bpm=72.0,
            corrupted_regions=corrupted,
        )
        res = evaluate_quality(packets)

        # 6 regions are clean, 2 corrupted
        for c in corrupted:
            assert c in res["rejected_regions"]
        assert "forehead_center" in res["selected_regions"]
        assert res["accepted"] is True  # 6 clean regions >= min_valid_regions (2)

    def test_algorithm_divergence_rejected(self):
        """Test rejection when Green, CHROM, POS estimates diverge > 8 BPM."""
        packets = make_synthetic_packet_stream(duration_s=12.0, bpm=72.0)
        divergent_hr = {
            "bpm_green": 72.0,
            "bpm_chrom": 85.0,  # 13 BPM divergence
            "bpm_pos": 73.0,
        }
        res = evaluate_quality(packets, hr_estimate=divergent_hr)

        assert res["accepted"] is False
        assert any("divergence" in r.lower() or "algorithm" in r.lower() for r in res["reasons"])
