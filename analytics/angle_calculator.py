"""Geometric angle helpers reserved for ergonomic indicators."""

from __future__ import annotations

import math
from collections.abc import Sequence


def angle_degrees(a: Sequence[float], vertex: Sequence[float], c: Sequence[float]) -> float:
    """Return the 2D/3D angle ABC in degrees, or raise for degenerate vectors."""
    try:
        dimensions = len(a)
        matching_dimensions = len(vertex) == dimensions and len(c) == dimensions
    except TypeError as error:
        raise ValueError("Os pontos precisam ser sequências 2D ou 3D.") from error
    if dimensions not in (2, 3) or not matching_dimensions:
        raise ValueError("Os pontos precisam ter a mesma dimensão (2D ou 3D).")
    try:
        coordinates = [[float(value) for value in point] for point in (a, vertex, c)]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("As coordenadas devem ser numéricas e finitas.") from error
    if not all(math.isfinite(value) for point in coordinates for value in point):
        raise ValueError("As coordenadas devem ser numéricas e finitas.")
    a_values, vertex_values, c_values = coordinates
    ba = [x - v for x, v in zip(a_values, vertex_values)]
    bc = [x - v for x, v in zip(c_values, vertex_values)]
    norm_ba = math.sqrt(sum(value * value for value in ba))
    norm_bc = math.sqrt(sum(value * value for value in bc))
    if norm_ba == 0 or norm_bc == 0:
        raise ValueError("Não é possível calcular o ângulo com pontos coincidentes.")
    cosine = sum(x * y for x, y in zip(ba, bc)) / (norm_ba * norm_bc)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
