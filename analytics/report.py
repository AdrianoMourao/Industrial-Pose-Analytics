"""PDF export for a video run's kinematic indicators and coverage."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Mapping

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from analytics.posture_analyzer import METRIC_FIELDS, METRIC_LABELS
from analytics.temporal import descriptive_statistics, smooth_series


def export_analysis_pdf(
    metrics_csv: Path,
    destination: Path,
    video_name: str,
    processing_info: Mapping[str, object] | None = None,
    filter_method: str = "savgol",
    window_length: int = 5,
    polynomial_order: int = 2,
) -> Path:
    """Write a two-page PDF with metadata, statistics and temporal plots."""
    times: list[float] = []
    series = {field: [] for field in METRIC_FIELDS}
    with metrics_csv.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            try:
                times.append(float(row.get("timestamp_ms") or 0) / 1000)
            except (TypeError, ValueError):
                times.append(float(len(times)))
            for field in METRIC_FIELDS:
                raw = row.get(field, "")
                try:
                    series[field].append(float(raw) if raw not in (None, "") else None)
                except (TypeError, ValueError):
                    series[field].append(None)
    if not times:
        raise ValueError("O CSV de indicadores não possui frames para o relatório.")

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    info = dict(processing_info or {})
    filter_name = {
        "none": "Sem suavização",
        "moving_average": "Média móvel",
        "savgol": "Savitzky-Golay",
    }.get(filter_method, filter_method)
    with PdfPages(destination) as pdf:
        for page_start in (0, 4):
            figure = Figure(figsize=(11.69, 8.27), layout="constrained")
            grid = figure.add_gridspec(
                5, 2, height_ratios=(1.2, 1.5, 0.35, 1.5, 0.35)
            )
            title = "Relatório de análise cinemática"
            figure.suptitle(title, fontsize=17, fontweight="bold", x=0.06, ha="left")
            if page_start == 0:
                metadata = [
                    f"Vídeo: {video_name}",
                    f"Gerado em: {datetime.now():%d/%m/%Y %H:%M}",
                    f"Frames: {info.get('processed_frames', len(times))}/"
                    f"{info.get('total_frames', len(times))}    ·    "
                    f"FPS efetivo: {info.get('effective_fps', 'não informado')}    ·    "
                    f"Tempo: {info.get('elapsed_seconds', 'não informado')} s",
                    f"FPS do vídeo: {info.get('video_fps', 'não informado')}    ·    "
                    f"Duração: {info.get('video_duration', 'não informada')} s",
                    f"Frames com pose: {info.get('detected_frames', 'não informado')}    ·    "
                    f"Taxa de detecção: {info.get('detection_rate', 'não informada')}",
                    f"Confiança mínima de landmarks: {info.get('confidence_threshold', 'não informada')}",
                    f"Filtro no gráfico: {filter_name} (janela {window_length}, ordem {polynomial_order})",
                    f"CSV de indicadores: {metrics_csv.name}    ·    Aplicação: {info.get('application_version', 'não informada')}",
                ]
                figure.text(0.06, 0.91, "\n".join(metadata), fontsize=8.5, va="top")
            else:
                figure.text(0.06, 0.91, f"Vídeo: {video_name}    ·    Continuação dos indicadores",
                            fontsize=9, va="top")

            for local_index, field in enumerate(METRIC_FIELDS[page_start:page_start + 4]):
                chart_row = 1 if local_index < 2 else 3
                column_index = local_index % 2
                axis = figure.add_subplot(grid[chart_row, column_index])
                values = series[field]
                stats = descriptive_statistics(values)
                filtered = smooth_series(values, filter_method, window_length, polynomial_order)
                stride = max(1, (len(times) + 4999) // 5000)
                axis.plot(times[::stride], values[::stride], color="#687782", alpha=0.55,
                          linewidth=0.8, label="Original")
                if filter_method != "none":
                    axis.plot(times[::stride], filtered[::stride], color="#148eaa", linewidth=1.3,
                              label="Suavizado")
                axis.set_title(METRIC_LABELS[field], loc="left", fontsize=10)
                axis.set_xlabel("Tempo (s)", fontsize=8)
                axis.set_ylabel("Ângulo (°)", fontsize=8)
                axis.grid(True, alpha=0.25)
                axis.tick_params(labelsize=7)
                stat_text = (
                    f"média {stats['mean']:.1f}°  ·  mediana {stats['median']:.1f}°  ·  "
                    f"DP {stats['std']:.1f}°\nmin {stats['min']:.1f}°  ·  max {stats['max']:.1f}°  ·  "
                    f"cobertura {stats['coverage_pct']:.1f}% ({stats['valid_frames']}/{stats['total_frames']})"
                    if stats["mean"] is not None else
                    f"Amostras válidas insuficientes  ·  cobertura {stats['coverage_pct']:.1f}% "
                    f"({stats['valid_frames']}/{stats['total_frames']})"
                )
                stats_axis = figure.add_subplot(grid[chart_row + 1, column_index])
                stats_axis.axis("off")
                stats_axis.text(0, 0.9, stat_text, fontsize=7.5, va="top", wrap=True)
                axis.legend(fontsize=7, loc="upper right")

            figure.text(
                0.06, 0.015,
                "Limitações: ângulos projetados em 2D dependem da posição da câmera e não representam "
                "uma avaliação clínica ou classificação automática de segurança. A suavização é apenas "
                "uma leitura auxiliar; os valores originais permanecem no CSV.",
                fontsize=7.5, wrap=True,
            )
            pdf.savefig(figure)
            figure.clear()
    return destination
