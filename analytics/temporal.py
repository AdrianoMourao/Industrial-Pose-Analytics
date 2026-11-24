"""Independent temporal filters and descriptive statistics for metric series."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable

import numpy as np


FILTER_METHODS = ("none", "moving_average", "savgol")


def _finite(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def smooth_series(
    values: Iterable[float | None],
    method: str = "none",
    window_length: int = 5,
    polynomial_order: int = 2,
) -> list[float | None]:
    """Smooth finite segments while retaining observed local extrema.

    Savitzky-Golay uses local least-squares polynomials and NumPy only. Invalid
    windows are rejected rather than silently changing the selected filter.
    Local peaks and troughs are restored at their original timestamps so a
    smoother cannot erase a short movement or change its peak angle.
    """
    if method not in FILTER_METHODS:
        raise ValueError(f"Filtro desconhecido: {method}")
    if window_length < 1:
        raise ValueError("A janela deve ser maior que zero.")
    if method == "savgol" and (window_length < 3 or window_length % 2 == 0):
        raise ValueError("A janela Savitzky-Golay deve ser ímpar e maior ou igual a 3.")
    if method == "savgol" and not 0 <= polynomial_order < window_length:
        raise ValueError("A ordem polinomial deve ser menor que a janela.")

    source = [_finite(value) for value in values]
    if method == "none":
        return source
    result: list[float | None] = [None] * len(source)
    index = 0
    while index < len(source):
        if source[index] is None:
            index += 1
            continue
        start = index
        while index < len(source) and source[index] is not None:
            index += 1
        end = index
        segment = [float(value) for value in source[start:end]]
        if method == "moving_average":
            for local_index, value in enumerate(segment):
                left = local_index - (window_length - 1) // 2
                right = left + window_length
                if left < 0:
                    left, right = 0, min(len(segment), window_length)
                elif right > len(segment):
                    right = len(segment)
                    left = max(0, right - window_length)
                result[start + local_index] = statistics.fmean(segment[left:right])
            _preserve_local_extrema(segment, result, start)
            continue

        effective_window = min(window_length, len(segment))
        if effective_window % 2 == 0:
            effective_window -= 1
        if effective_window < 3 or polynomial_order >= effective_window:
            result[start:end] = segment
            continue
        half_window = effective_window // 2
        for local_index in range(len(segment)):
            left = min(max(local_index - half_window, 0), len(segment) - effective_window)
            right = left + effective_window
            x = np.arange(left, right, dtype=float) - local_index
            design = np.vander(x, N=polynomial_order + 1, increasing=True)
            coefficients, *_ = np.linalg.lstsq(design, np.asarray(segment[left:right]), rcond=None)
            result[start + local_index] = float(coefficients[0])
        _preserve_local_extrema(segment, result, start)
    return result


def _preserve_local_extrema(segment: list[float], result: list[float | None], offset: int) -> None:
    """Keep local peaks and troughs at their original values after smoothing."""
    if len(segment) < 2:
        return
    if segment[0] >= segment[1]:
        result[offset] = segment[0]
    if segment[-1] >= segment[-2]:
        result[offset + len(segment) - 1] = segment[-1]
    for index in range(1, len(segment) - 1):
        previous, current, following = segment[index - 1:index + 2]
        is_peak = current >= previous and current >= following and (current > previous or current > following)
        is_trough = current <= previous and current <= following and (current < previous or current < following)
        if is_peak or is_trough:
            result[offset + index] = current


def descriptive_statistics(values: Iterable[float | None], total_frames: int | None = None) -> dict:
    """Summarize valid finite values and coverage against the total frame count."""
    parsed = [_finite(value) for value in values]
    valid = [value for value in parsed if value is not None]
    total = len(parsed) if total_frames is None else max(0, int(total_frames))
    enough = len(valid) >= 2
    return {
        "mean": statistics.fmean(valid) if enough else None,
        "median": statistics.median(valid) if enough else None,
        "std": statistics.pstdev(valid) if enough else None,
        "min": min(valid) if enough else None,
        "max": max(valid) if enough else None,
        "valid_frames": len(valid),
        "invalid_frames": max(total - len(valid), 0),
        "total_frames": total,
        "coverage_pct": len(valid) / total * 100 if total else 0.0,
    }
