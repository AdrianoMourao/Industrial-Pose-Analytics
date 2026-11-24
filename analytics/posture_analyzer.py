"""Frame-by-frame 2D ergonomic measurements from MediaPipe pose landmarks."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from analytics.angle_calculator import angle_degrees

METRIC_FIELDS = (
    "trunk_inclination_deg",
    "left_arm_elevation_deg",
    "right_arm_elevation_deg",
    "left_elbow_flexion_deg",
    "right_elbow_flexion_deg",
    "head_inclination_deg",
    "left_knee_flexion_deg",
    "right_knee_flexion_deg",
)
MIN_VALID_SAMPLES = 2

METRIC_LABELS = {
    "trunk_inclination_deg": "Inclinação do tronco",
    "left_arm_elevation_deg": "Elevação do braço esquerdo",
    "right_arm_elevation_deg": "Elevação do braço direito",
    "left_elbow_flexion_deg": "Flexão do cotovelo esquerdo",
    "right_elbow_flexion_deg": "Flexão do cotovelo direito",
    "head_inclination_deg": "Inclinação cabeça/tronco",
    "left_knee_flexion_deg": "Flexão do joelho esquerdo",
    "right_knee_flexion_deg": "Flexão do joelho direito",
}

METRIC_GROUPS = {
    "Inclinação do tronco": ("trunk_inclination_deg",),
    "Elevação dos braços": ("left_arm_elevation_deg", "right_arm_elevation_deg"),
    "Flexão dos cotovelos": ("left_elbow_flexion_deg", "right_elbow_flexion_deg"),
    "Inclinação da cabeça": ("head_inclination_deg",),
    "Flexão dos joelhos": ("left_knee_flexion_deg", "right_knee_flexion_deg"),
}


def _value(landmark: Any, name: str, default=None):
    if isinstance(landmark, Mapping):
        return landmark.get(name, default)
    return getattr(landmark, name, default)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


class MetricAccumulator:
    """Per-indicator statistics and coverage over valid frame measurements."""

    def __init__(self) -> None:
        self.total_frames = 0
        self.counts = {field: 0 for field in METRIC_FIELDS}
        self.values = {field: [] for field in METRIC_FIELDS}

    def update(self, row: Mapping[str, Any]) -> None:
        self.total_frames += 1
        for field in METRIC_FIELDS:
            value = _number(row.get(field))
            if value is None:
                continue
            self.counts[field] += 1
            self.values[field].append(value)

    def summary(self) -> dict[str, dict[str, float | int | None]]:
        """Return descriptive statistics and valid/invalid frame coverage."""
        result = {}
        for field in METRIC_FIELDS:
            count = self.counts[field]
            values = self.values[field]
            enough = count >= MIN_VALID_SAMPLES
            result[field] = {
                "mean": statistics.fmean(values) if enough else None,
                "median": statistics.median(values) if enough else None,
                "std": statistics.pstdev(values) if enough else None,
                "min": min(values) if enough else None,
                "max": max(values) if enough else None,
                "coverage_pct": count / self.total_frames * 100 if self.total_frames else 0.0,
                "samples": count,
                "valid_frames": count,
                "invalid_frames": self.total_frames - count,
                "total_frames": self.total_frames,
            }
        return result


class PostureAnalyzer:
    """Calculate projected joint angles without assigning ergonomic risk levels.

    Angles are measured in the image plane after converting normalized x/y values
    into pixel coordinates. Their interpretation depends on the camera viewpoint.
    """

    def __init__(self, visibility_threshold: float = 0.5) -> None:
        if not math.isfinite(visibility_threshold) or not 0.0 <= visibility_threshold <= 1.0:
            raise ValueError("visibility_threshold deve estar entre 0 e 1.")
        self.visibility_threshold = visibility_threshold

    def analyze_frame(
        self,
        frame_number: int,
        timestamp_ms: int,
        landmarks: Mapping[str, Any],
        frame_width: int,
        frame_height: int,
    ) -> dict[str, float | int | None]:
        """Calculate five ergonomic indicators for one frame.

        Side-specific indicators are kept separate so asymmetric movement remains
        visible in the time series. Missing or low-confidence joints yield blanks.
        """
        row: dict[str, float | int | None] = {
            "frame": frame_number,
            "timestamp_ms": timestamp_ms,
            **{field: None for field in METRIC_FIELDS},
        }
        points = {
            name: self._point(point, frame_width, frame_height)
            for name, point in landmarks.items()
            if self._valid(point)
        }

        left_shoulder, right_shoulder = points.get("left_shoulder"), points.get("right_shoulder")
        left_hip, right_hip = points.get("left_hip"), points.get("right_hip")
        shoulders = self._midpoint(left_shoulder, right_shoulder)
        hips = self._midpoint(left_hip, right_hip)

        if shoulders and hips:
            row["trunk_inclination_deg"] = self._angle_from_vertical(shoulders, hips)

        for side in ("left", "right"):
            shoulder = points.get(f"{side}_shoulder")
            hip = points.get(f"{side}_hip")
            elbow = points.get(f"{side}_elbow")
            wrist = points.get(f"{side}_wrist")
            knee = points.get(f"{side}_knee")
            ankle = points.get(f"{side}_ankle")

            if shoulder and hip and elbow:
                row[f"{side}_arm_elevation_deg"] = self._safe_angle(hip, shoulder, elbow)
            if shoulder and elbow and wrist:
                inner = self._safe_angle(shoulder, elbow, wrist)
                row[f"{side}_elbow_flexion_deg"] = 180.0 - inner if inner is not None else None
            if hip and knee and ankle:
                inner = self._safe_angle(hip, knee, ankle)
                row[f"{side}_knee_flexion_deg"] = 180.0 - inner if inner is not None else None

        ears = self._midpoint(points.get("left_ear"), points.get("right_ear"))
        if ears and shoulders and hips:
            trunk_up = (2 * shoulders[0] - hips[0], 2 * shoulders[1] - hips[1])
            row["head_inclination_deg"] = self._safe_angle(ears, shoulders, trunk_up)
        return row

    def rebuild_from_landmarks_csv(
        self,
        landmarks_csv: Path,
        metrics_csv: Path,
        frame_width: int,
        frame_height: int,
        total_frames: int | None = None,
        fps: float = 30.0,
    ) -> dict[str, dict[str, float | int | None]]:
        """Create an ergonomic time series from an already-exported landmark CSV."""
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = ("frame", "timestamp_ms", "confidence_threshold", *METRIC_FIELDS)
        accumulator = MetricAccumulator()
        with landmarks_csv.open(newline="", encoding="utf-8-sig") as source, metrics_csv.open(
            "w", newline="", encoding="utf-8-sig"
        ) as destination:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            current_frame = None
            current_timestamp = 0
            current_landmarks: dict[str, dict[str, Any]] = {}

            def emit_empty(frame_number: int) -> None:
                row = self.analyze_frame(
                    frame_number, round(frame_number * 1000 / fps), {}, frame_width, frame_height
                )
                row["confidence_threshold"] = self.visibility_threshold
                accumulator.update(row)
                writer.writerow(row)

            def flush_frame() -> None:
                if current_frame is None:
                    return
                result = self.analyze_frame(
                    current_frame, current_timestamp, current_landmarks, frame_width, frame_height
                )
                result["confidence_threshold"] = self.visibility_threshold
                accumulator.update(result)
                writer.writerow(result)

            for row in reader:
                frame = int(row["frame"])
                timestamp = int(float(row.get("timestamp_ms") or 0))
                if current_frame is None:
                    for missing_frame in range(frame):
                        emit_empty(missing_frame)
                if current_frame is not None and frame != current_frame:
                    flush_frame()
                    for missing_frame in range(current_frame + 1, frame):
                        emit_empty(missing_frame)
                    current_landmarks = {}
                current_frame = frame
                current_timestamp = timestamp
                name = row.get("landmark_name", "")
                if name:
                    current_landmarks[name] = row
            flush_frame()
            if total_frames is not None:
                first_missing = current_frame + 1 if current_frame is not None else 0
                for missing_frame in range(first_missing, total_frames):
                    emit_empty(missing_frame)
        return accumulator.summary()

    @staticmethod
    def summarize_metrics_csv(metrics_csv: Path) -> dict[str, dict[str, float | int | None]]:
        """Read a stored time series and calculate its compact summary."""
        accumulator = MetricAccumulator()
        with metrics_csv.open(newline="", encoding="utf-8-sig") as source:
            for row in csv.DictReader(source):
                accumulator.update(row)
        return accumulator.summary()

    def _valid(self, landmark: Any) -> bool:
        if landmark is None:
            return False
        has_confidence = False
        for field in ("visibility", "presence"):
            if isinstance(landmark, Mapping):
                present = field in landmark
            else:
                present = hasattr(landmark, field)
            if not present:
                continue
            raw_confidence = _value(landmark, field)
            if raw_confidence is None or raw_confidence == "":
                continue
            has_confidence = True
            confidence = _number(raw_confidence)
            if confidence is None or not 0.0 <= confidence <= 1.0:
                return False
            if confidence < self.visibility_threshold:
                return False
        return has_confidence

    @staticmethod
    def _point(landmark: Any, width: int, height: int) -> tuple[float, float] | None:
        if width <= 0 or height <= 0:
            return None
        x = _number(_value(landmark, "x"))
        y = _number(_value(landmark, "y"))
        if x is None or y is None or not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            return None
        return x * width, y * height

    @staticmethod
    def _midpoint(a, b):
        if a is None or b is None:
            return None
        return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)

    @classmethod
    def _angle_from_vertical(cls, shoulders, hips) -> float | None:
        up = (hips[0], hips[1] - max(abs(shoulders[1] - hips[1]), 1.0))
        return cls._safe_angle(shoulders, hips, up)

    @staticmethod
    def _safe_angle(a: Sequence[float], vertex: Sequence[float], c: Sequence[float]):
        try:
            return angle_degrees(a, vertex, c)
        except ValueError:
            return None
