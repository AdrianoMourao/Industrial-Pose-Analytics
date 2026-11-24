"""Small helpers for processing summary metrics."""


def detection_rate(detected_frames: int, processed_frames: int) -> float:
    """Calculate the percentage of frames containing a valid pose."""
    if processed_frames <= 0:
        return 0.0
    return detected_frames / processed_frames * 100.0
