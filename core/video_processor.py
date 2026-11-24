"""Offline video processing with safe cancellation and CSV export."""

from __future__ import annotations

import csv
import logging
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2

from config.constants import MODEL_PATH
from config.settings import DetectionSettings
from core.pose_detector import PoseDetector
from analytics.posture_analyzer import METRIC_FIELDS, MetricAccumulator, PostureAnalyzer

logger = logging.getLogger(__name__)

LANDMARK_NAMES = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky",
    "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
)


@dataclass(slots=True)
class ProcessingSummary:
    """Performance and output summary for a processed input video."""

    processed_frames: int
    detected_frames: int
    total_frames: int
    elapsed_seconds: float
    effective_fps: float
    detection_rate: float
    video_path: Path
    csv_path: Path
    metrics_csv_path: Path
    metric_summary: dict


ProgressCallback = Callable[[dict], None]


class VideoProcessor:
    """Process an input video one frame at a time."""

    def __init__(
        self,
        settings: DetectionSettings,
        model_path: Path = MODEL_PATH,
        output_root: Path | None = None,
    ) -> None:
        self.settings = settings
        self.model_path = model_path
        self.output_root = output_root

    def process(
        self,
        video_path: Path,
        cancel_event: threading.Event,
        progress: ProgressCallback,
    ) -> ProcessingSummary:
        """Process a video and report progress from the calling worker thread."""
        video_path = video_path.resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

        output_dir = self.output_root or video_path.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_video = output_dir / f"{video_path.stem}_pose_{run_id}.mp4"
        output_csv = output_dir / f"{video_path.stem}_landmarks_{run_id}.csv"
        metrics_csv = output_dir / f"{video_path.stem}_ergonomic_metrics_{run_id}.csv"

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError("O OpenCV não conseguiu abrir o vídeo selecionado.")

        writer = None
        processed = detected = 0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if math.isfinite(fps) and fps > 0 else 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            capture.release()
            raise ValueError("Não foi possível identificar a resolução do vídeo.")

        start_time = time.perf_counter()
        last_timestamp = -1
        analyzer = PostureAnalyzer(self.settings.visibility_threshold)
        metric_accumulator = MetricAccumulator()
        try:
            writer = cv2.VideoWriter(
                str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            if not writer.isOpened():
                raise RuntimeError("Não foi possível criar o vídeo MP4 de saída.")

            with (
                PoseDetector(self.model_path, self.settings) as detector,
                output_csv.open("w", newline="", encoding="utf-8-sig") as csv_file,
                metrics_csv.open("w", newline="", encoding="utf-8-sig") as metrics_file,
            ):
                csv_writer = csv.DictWriter(
                    csv_file,
                    fieldnames=(
                        "frame", "timestamp_ms", "landmark_id", "landmark_name", "x", "y", "z",
                        "visibility", "presence",
                    ),
                )
                csv_writer.writeheader()
                metrics_writer = csv.DictWriter(
                    metrics_file,
                    fieldnames=("frame", "timestamp_ms", "confidence_threshold", *METRIC_FIELDS),
                )
                metrics_writer.writeheader()

                preview_interval = max(1, round(fps / 10))
                while not cancel_event.is_set():
                    ok, frame = capture.read()
                    if not ok:
                        break
                    timestamp = max(round(processed * 1000 / fps), last_timestamp + 1)
                    last_timestamp = timestamp
                    landmarks, annotated = detector.detect(frame, timestamp)
                    named_landmarks = {}
                    if landmarks:
                        detected += 1
                        for index, point in enumerate(landmarks):
                            named_landmarks[LANDMARK_NAMES[index]] = point
                            visibility = getattr(point, "visibility", None)
                            presence = getattr(point, "presence", None)
                            if visibility is not None and visibility < self.settings.visibility_threshold:
                                continue
                            if presence is not None and presence < self.settings.visibility_threshold:
                                continue
                            csv_writer.writerow(
                                {
                                    "frame": processed,
                                    "timestamp_ms": timestamp,
                                    "landmark_id": index,
                                    "landmark_name": LANDMARK_NAMES[index],
                                    "x": point.x,
                                    "y": point.y,
                                    "z": point.z,
                                    "visibility": visibility,
                                    "presence": presence,
                                }
                            )
                    metric_row = analyzer.analyze_frame(
                        processed, timestamp, named_landmarks, width, height
                    )
                    metric_row["confidence_threshold"] = self.settings.visibility_threshold
                    metrics_writer.writerow(metric_row)
                    metric_accumulator.update(metric_row)
                    writer.write(annotated)
                    processed += 1

                    # Send preview frames often enough to look like a live feed,
                    # without flooding Tk's UI queue on high-FPS source videos.
                    if processed == 1 or processed % preview_interval == 0:
                        elapsed = time.perf_counter() - start_time
                        progress(
                            {
                                "type": "progress",
                                "processed": processed,
                                "total": total_frames,
                                "detected": detected,
                                "fps": processed / elapsed if elapsed else 0.0,
                                "elapsed": elapsed,
                                "preview_frame": annotated.copy(),
                            }
                        )

            elapsed = time.perf_counter() - start_time
            if cancel_event.is_set():
                raise InterruptedError("Processamento cancelado pelo usuário.")
            if processed == 0:
                raise ValueError("Nenhum frame válido foi lido do vídeo.")
            rate = detected / processed * 100 if processed else 0.0
            summary = ProcessingSummary(
                processed_frames=processed,
                detected_frames=detected,
                total_frames=total_frames or processed,
                elapsed_seconds=elapsed,
                effective_fps=processed / elapsed if elapsed else 0.0,
                detection_rate=rate,
                video_path=output_video,
                csv_path=output_csv,
                metrics_csv_path=metrics_csv,
                metric_summary=metric_accumulator.summary(),
            )
            logger.info(
                "Vídeo processado: %s frames, %.1f%% detecção, %.2f FPS efetivos",
                processed, rate, summary.effective_fps,
            )
            return summary
        except InterruptedError:
            capture.release()
            if writer is not None:
                writer.release()
                writer = None
            output_video.unlink(missing_ok=True)
            output_csv.unlink(missing_ok=True)
            metrics_csv.unlink(missing_ok=True)
            logger.info("Processamento cancelado pelo usuário: %s", video_path)
            raise
        except Exception:
            logger.exception("Falha ao processar vídeo %s", video_path)
            capture.release()
            if writer is not None:
                writer.release()
                writer = None
            output_video.unlink(missing_ok=True)
            output_csv.unlink(missing_ok=True)
            metrics_csv.unlink(missing_ok=True)
            raise
        finally:
            capture.release()
            if writer is not None:
                writer.release()
