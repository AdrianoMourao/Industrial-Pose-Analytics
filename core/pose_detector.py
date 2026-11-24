"""MediaPipe Tasks Pose Landmarker wrapper and skeleton drawing."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config.settings import DetectionSettings

logger = logging.getLogger(__name__)


class PoseDetector:
    """Run VIDEO-mode pose detection and annotate BGR video frames."""

    def __init__(self, model_path: Path, settings: DetectionSettings) -> None:
        settings.validate()
        if not model_path.is_file():
            raise FileNotFoundError(f"Modelo MediaPipe não encontrado: {model_path}")

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=settings.number_of_poses,
            min_pose_detection_confidence=settings.detection_confidence,
            min_pose_presence_confidence=settings.presence_confidence,
            min_tracking_confidence=settings.tracking_confidence,
        )
        self._settings = settings
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._connections = vision.PoseLandmarksConnections.POSE_LANDMARKS

    def detect(self, frame_bgr, timestamp_ms: int):
        """Return the first detected pose and a copy of the frame with its skeleton."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        annotated = frame_bgr.copy()
        landmarks = result.pose_landmarks[0] if result.pose_landmarks else None

        if landmarks:
            height, width = annotated.shape[:2]
            valid = [self._is_valid(point) for point in landmarks]
            for connection in self._connections:
                start, end = connection.start, connection.end
                if valid[start] and valid[end]:
                    a = (int(landmarks[start].x * width), int(landmarks[start].y * height))
                    b = (int(landmarks[end].x * width), int(landmarks[end].y * height))
                    cv2.line(annotated, a, b, (214, 213, 91), 2, cv2.LINE_AA)

            for index, point in enumerate(landmarks):
                if valid[index]:
                    center = (int(point.x * width), int(point.y * height))
                    cv2.circle(annotated, center, 4, (103, 216, 239), -1, cv2.LINE_AA)
                    if index in {11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28}:
                        cv2.putText(
                            annotated,
                            vision.PoseLandmark(index).name.replace("_", " ").title(),
                            (center[0] + 6, center[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.32,
                            (241, 245, 247),
                            1,
                            cv2.LINE_AA,
                        )
        return landmarks, annotated

    def _is_valid(self, landmark) -> bool:
        visibility = getattr(landmark, "visibility", None)
        presence = getattr(landmark, "presence", None)
        threshold = self._settings.visibility_threshold
        if visibility is not None and visibility < threshold:
            return False
        if presence is not None and presence < threshold:
            return False
        return True

    def close(self) -> None:
        """Release the MediaPipe graph resources."""
        self._landmarker.close()

    def __enter__(self) -> "PoseDetector":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
