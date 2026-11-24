"""Runtime detection settings."""

from dataclasses import dataclass


@dataclass(slots=True)
class DetectionSettings:
    """Tunable MediaPipe Pose Landmarker options."""

    detection_confidence: float = 0.5
    presence_confidence: float = 0.5
    tracking_confidence: float = 0.5
    visibility_threshold: float = 0.5
    number_of_poses: int = 1

    def validate(self) -> None:
        """Raise ValueError if any configured confidence is outside [0, 1]."""
        for name in (
            "detection_confidence",
            "presence_confidence",
            "tracking_confidence",
            "visibility_threshold",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} deve estar entre 0 e 1.")
        if self.number_of_poses < 1:
            raise ValueError("number_of_poses deve ser pelo menos 1.")
