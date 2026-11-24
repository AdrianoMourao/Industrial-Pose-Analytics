"""Regression tests for geometry, per-indicator coverage and temporal analysis."""

from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

from analytics.angle_calculator import angle_degrees
from analytics.posture_analyzer import METRIC_FIELDS, MetricAccumulator, PostureAnalyzer
from analytics.report import export_analysis_pdf
from analytics.temporal import descriptive_statistics, smooth_series
from core.video_processor import LANDMARK_NAMES


class AngleGeometryTests(unittest.TestCase):
    def test_known_angles(self):
        for expected in (0, 45, 90, 135, 180):
            with self.subTest(expected=expected):
                radians = math.radians(expected)
                point = (math.cos(radians), math.sin(radians))
                self.assertAlmostEqual(angle_degrees(point, (0, 0), (1, 0)), expected)

    def test_degenerate_and_invalid_coordinates_are_rejected(self):
        with self.assertRaises(ValueError):
            angle_degrees((0, 0), (0, 0), (1, 0))
        with self.assertRaises(ValueError):
            angle_degrees((0, 0), (0, 0, 0), (1, 0))
        with self.assertRaises(ValueError):
            angle_degrees((float("nan"), 0), (0, 0), (1, 0))
        with self.assertRaises(ValueError):
            angle_degrees(("invalid", 0), (0, 0), (1, 0))


def landmark(x, y, visibility=1.0, presence=1.0):
    return {"x": x, "y": y, "visibility": visibility, "presence": presence}


def asymmetric_pose():
    return {
        "left_ear": landmark(0.45, 0.20), "right_ear": landmark(0.55, 0.20),
        "left_shoulder": landmark(0.30, 0.40), "right_shoulder": landmark(0.70, 0.40),
        "left_elbow": landmark(0.40, 0.40), "right_elbow": landmark(0.70, 0.30),
        "left_wrist": landmark(0.50, 0.40), "right_wrist": landmark(0.70, 0.40),
        "left_hip": landmark(0.30, 0.60), "right_hip": landmark(0.70, 0.60),
        "left_knee": landmark(0.30, 0.75), "right_knee": landmark(0.70, 0.70),
        "left_ankle": landmark(0.30, 0.90), "right_ankle": landmark(0.80, 0.70),
    }


class IndicatorTests(unittest.TestCase):
    def test_mediapipe_anatomical_landmark_indices(self):
        expected = {
            11: "left_shoulder", 12: "right_shoulder",
            13: "left_elbow", 14: "right_elbow",
            15: "left_wrist", 16: "right_wrist",
            23: "left_hip", 24: "right_hip",
            25: "left_knee", 26: "right_knee",
            27: "left_ankle", 28: "right_ankle",
            7: "left_ear", 8: "right_ear",
        }
        for index, name in expected.items():
            self.assertEqual(LANDMARK_NAMES[index], name)

    def test_indicator_geometry_and_anatomical_sides(self):
        row = PostureAnalyzer().analyze_frame(0, 0, asymmetric_pose(), 100, 100)
        self.assertAlmostEqual(row["trunk_inclination_deg"], 0.0)
        self.assertAlmostEqual(row["left_arm_elevation_deg"], 90.0)
        self.assertAlmostEqual(row["right_arm_elevation_deg"], 180.0)
        self.assertAlmostEqual(row["left_elbow_flexion_deg"], 0.0)
        self.assertAlmostEqual(row["right_elbow_flexion_deg"], 180.0)
        self.assertAlmostEqual(row["head_inclination_deg"], 0.0)
        self.assertAlmostEqual(row["left_knee_flexion_deg"], 0.0)
        self.assertAlmostEqual(row["right_knee_flexion_deg"], 90.0)

    def test_trunk_and_head_inclinations(self):
        points = asymmetric_pose()
        points["left_shoulder"] = landmark(0.40, 0.50)
        points["right_shoulder"] = landmark(0.80, 0.50)
        row = PostureAnalyzer().analyze_frame(0, 0, points, 100, 100)
        self.assertAlmostEqual(row["trunk_inclination_deg"], 45.0)

        points = asymmetric_pose()
        points["left_ear"] = landmark(0.55, 0.20)
        points["right_ear"] = landmark(0.65, 0.20)
        row = PostureAnalyzer().analyze_frame(0, 0, points, 100, 100)
        self.assertAlmostEqual(row["head_inclination_deg"], math.degrees(math.atan(0.5)))

    def test_low_confidence_and_out_of_frame_landmarks_are_excluded(self):
        points = asymmetric_pose()
        points["left_wrist"] = landmark(0.50, 0.40, visibility=0.2)
        points["right_wrist"] = landmark(1.4, 0.40)
        row = PostureAnalyzer(visibility_threshold=0.5).analyze_frame(0, 0, points, 100, 100)
        self.assertIsNone(row["left_elbow_flexion_deg"])
        self.assertIsNone(row["right_elbow_flexion_deg"])
        self.assertIsNotNone(row["left_arm_elevation_deg"])

    def test_coverage_and_minimum_valid_sample_rule(self):
        accumulator = MetricAccumulator()
        accumulator.update({"trunk_inclination_deg": 10, "left_arm_elevation_deg": None})
        accumulator.update({"trunk_inclination_deg": float("nan"), "left_arm_elevation_deg": 20})
        accumulator.update({"trunk_inclination_deg": 30, "left_arm_elevation_deg": None})
        summary = accumulator.summary()
        trunk = summary["trunk_inclination_deg"]
        self.assertEqual(trunk["valid_frames"], 2)
        self.assertEqual(trunk["invalid_frames"], 1)
        self.assertAlmostEqual(trunk["coverage_pct"], 200 / 3)
        self.assertAlmostEqual(trunk["mean"], 20)
        arm = summary["left_arm_elevation_deg"]
        self.assertEqual(arm["valid_frames"], 1)
        self.assertIsNone(arm["mean"])
        self.assertIsNone(arm["min"])
        self.assertIsNone(arm["max"])


class TemporalFilterTests(unittest.TestCase):
    def test_filters_are_independent_and_preserve_gaps(self):
        raw = [0.0, 2.0, None, 8.0, 10.0]
        moving = smooth_series(raw, "moving_average", window_length=3)
        savgol = smooth_series(raw, "savgol", window_length=3, polynomial_order=1)
        self.assertEqual(raw, [0.0, 2.0, None, 8.0, 10.0])
        self.assertIsNone(moving[2])
        self.assertIsNone(savgol[2])
        self.assertNotEqual(moving, savgol)

    def test_savgol_preserves_quadratic_trend_and_peak(self):
        quadratic = [float(value * value) for value in range(-5, 6)]
        filtered = smooth_series(quadratic, "savgol", 5, 2)
        for original, result in zip(quadratic, filtered):
            self.assertAlmostEqual(result, original, places=8)
        peak = [0, 2, 6, 10, 6, 2, 0]
        for method in ("moving_average", "savgol"):
            smoothed_peak = smooth_series(peak, method, 5, 2)
            self.assertAlmostEqual(max(smoothed_peak), 10, delta=0.0)

    def test_statistics_match_valid_samples_and_coverage(self):
        stats = descriptive_statistics([1, None, 3, float("nan")])
        self.assertEqual(stats["valid_frames"], 2)
        self.assertEqual(stats["invalid_frames"], 2)
        self.assertEqual(stats["coverage_pct"], 50)
        self.assertEqual(stats["median"], 2)

    def test_smoothing_rejects_invalid_savgol_windows(self):
        with self.assertRaises(ValueError):
            smooth_series([1, 2, 3], "savgol", 4, 2)


class PdfReportTests(unittest.TestCase):
    def test_exports_pdf_with_temporal_series(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "metrics.csv"
            fields = ("frame", "timestamp_ms", *METRIC_FIELDS)
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for frame in range(8):
                    writer.writerow({
                        "frame": frame,
                        "timestamp_ms": frame * 100,
                        **{field: frame * 2 for field in METRIC_FIELDS},
                    })
            destination = root / "report.pdf"
            export_analysis_pdf(source, destination, "sample.mp4", {"processed_frames": 8})
            self.assertTrue(destination.is_file())
            self.assertTrue(destination.read_bytes().startswith(b"%PDF"))
            self.assertGreater(destination.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
