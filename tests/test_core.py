"""Unit and short integration checks for core calculations and video export."""

from __future__ import annotations

import csv
import tempfile
import threading
import unittest
from pathlib import Path

import cv2
import numpy as np

from analytics.angle_calculator import angle_degrees
from analytics.metrics import detection_rate
from config.constants import MODEL_PATH
from config.settings import DetectionSettings
from core.video_processor import VideoProcessor
from analytics.posture_analyzer import METRIC_FIELDS, PostureAnalyzer


class SettingsTests(unittest.TestCase):
    def test_default_settings_are_valid(self):
        DetectionSettings().validate()

    def test_invalid_confidence_is_rejected(self):
        with self.assertRaises(ValueError):
            DetectionSettings(detection_confidence=1.2).validate()


class AnalyticsTests(unittest.TestCase):
    def test_right_angle(self):
        self.assertAlmostEqual(angle_degrees((1, 0), (0, 0), (0, 1)), 90.0)

    def test_coincident_points_are_rejected(self):
        with self.assertRaises(ValueError):
            angle_degrees((0, 0), (0, 0), (1, 0))

    def test_detection_rate_handles_empty_input(self):
        self.assertEqual(detection_rate(0, 0), 0.0)
        self.assertEqual(detection_rate(3, 4), 75.0)


@unittest.skipUnless(MODEL_PATH.is_file(), "Pose Landmarker model is not present")
class VideoProcessorIntegrationTests(unittest.TestCase):
    def test_short_video_exports_annotated_video_and_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.mp4"
            writer = cv2.VideoWriter(
                str(input_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240)
            )
            self.assertTrue(writer.isOpened(), "MP4 video writer is unavailable")
            for index in range(3):
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(frame, str(index), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (180, 190, 200), 2)
                writer.write(frame)
            writer.release()

            progress_events = []
            processor = VideoProcessor(DetectionSettings(), output_root=root / "export")
            summary = processor.process(input_path, threading.Event(), progress_events.append)

            self.assertEqual(summary.processed_frames, 3)
            self.assertTrue(summary.video_path.is_file())
            self.assertTrue(summary.csv_path.is_file())
            self.assertTrue(summary.metrics_csv_path.is_file())
            self.assertGreater(summary.effective_fps, 0)
            self.assertTrue(progress_events)
            with summary.csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
                header = next(csv.reader(csv_file))
            self.assertEqual(
                header,
                ["frame", "timestamp_ms", "landmark_id", "landmark_name", "x", "y", "z",
                 "visibility", "presence"],
            )
            with summary.metrics_csv_path.open(newline="", encoding="utf-8-sig") as metrics_file:
                metric_rows = csv.DictReader(metrics_file)
                metric_header = metric_rows.fieldnames
                metric_first_row = next(metric_rows)
            self.assertEqual(
                metric_header,
                ["frame", "timestamp_ms", "confidence_threshold", *METRIC_FIELDS],
            )
            self.assertEqual(float(metric_first_row["confidence_threshold"]), 0.5)
            self.assertEqual(summary.metric_summary["trunk_inclination_deg"]["coverage_pct"], 0)


class PostureAnalyzerTests(unittest.TestCase):
    def test_upright_straight_pose_has_near_zero_angles(self):
        point = lambda x, y: {"x": x, "y": y, "visibility": 1.0, "presence": 1.0}
        landmarks = {
            "left_ear": point(0.45, 0.20), "right_ear": point(0.55, 0.20),
            "left_shoulder": point(0.45, 0.40), "right_shoulder": point(0.55, 0.40),
            "left_elbow": point(0.45, 0.45), "right_elbow": point(0.55, 0.45),
            "left_wrist": point(0.45, 0.50), "right_wrist": point(0.55, 0.50),
            "left_hip": point(0.45, 0.60), "right_hip": point(0.55, 0.60),
            "left_knee": point(0.45, 0.70), "right_knee": point(0.55, 0.70),
            "left_ankle": point(0.45, 0.80), "right_ankle": point(0.55, 0.80),
        }
        row = PostureAnalyzer().analyze_frame(0, 0, landmarks, 100, 100)
        for field in METRIC_FIELDS:
            self.assertIsNotNone(row[field], field)
            self.assertAlmostEqual(row[field], 0.0, places=6, msg=field)

    def test_low_visibility_joints_are_excluded(self):
        point = lambda x, y, visibility=1.0: {
            "x": x, "y": y, "visibility": visibility, "presence": 1.0
        }
        landmarks = {
            "left_shoulder": point(0.4, 0.4), "right_shoulder": point(0.6, 0.4),
            "left_hip": point(0.4, 0.6), "right_hip": point(0.6, 0.6),
            "left_elbow": point(0.4, 0.45), "left_wrist": point(0.4, 0.5, 0.1),
        }
        row = PostureAnalyzer().analyze_frame(0, 0, landmarks, 100, 100)
        self.assertIsNone(row["left_elbow_flexion_deg"])
        self.assertEqual(row["trunk_inclination_deg"], 0.0)


if __name__ == "__main__":
    unittest.main()
