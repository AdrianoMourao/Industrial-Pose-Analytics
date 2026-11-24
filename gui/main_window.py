"""Main CustomTkinter interface for Industrial Pose Analytics."""

from __future__ import annotations

import logging
import math
import queue
import threading
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import csv
from pathlib import Path

import customtkinter as ctk
import cv2
from PIL import Image

from config.constants import APP_VERSION, COLORS, MODEL_PATH, OUTPUT_DIR, ROOT_DIR
from config.settings import DetectionSettings
from core.video_processor import ProcessingSummary, VideoProcessor
from analytics.posture_analyzer import (
    METRIC_FIELDS, METRIC_GROUPS, METRIC_LABELS, PostureAnalyzer,
)
from analytics.temporal import descriptive_statistics, smooth_series

logger = logging.getLogger(__name__)


class MainWindow(ctk.CTk):
    """Responsive multi-page dashboard with non-blocking video processing."""

    NAV_ITEMS = ("Dashboard", "Video Analysis", "Analytics", "Settings")

    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.title("Industrial Pose Analytics")
        self.geometry("1450x940")
        self.minsize(1120, 760)
        self.configure(fg_color=COLORS["bg"])

        self.video_path: Path | None = None
        self.output_dir = OUTPUT_DIR
        self.settings = DetectionSettings()
        self.worker: threading.Thread | None = None
        self.cancel_event = threading.Event()
        self.events: queue.Queue[dict] = queue.Queue()
        self.active_page = "Dashboard"
        self.session_runs: list[ProcessingSummary] = []
        self.last_video_name = "Nenhum vídeo processado"
        self.latest_metrics_path: Path | None = None
        self.latest_metric_summary: dict = {}
        self.latest_source_video = ""
        self.latest_processing_info: dict = {}
        self.last_video_path: Path | None = None
        self.preview_image = None
        self._video_info: dict = {}
        self._restore_latest_metrics()

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._build_topbar()
        self.content_host = ctk.CTkFrame(self, fg_color=COLORS["bg"], corner_radius=0)
        self.content_host.grid(row=1, column=0, sticky="nsew", padx=34, pady=(16, 48))
        self.content_host.grid_columnconfigure(0, weight=1)
        self.content_host.grid_rowconfigure(0, weight=1)
        self.page_host = self.content_host
        self.page_frames: dict[str, ctk.CTkFrame] = {}
        self.built_pages: set[str] = set()
        self.dirty_pages: set[str] = set()
        self._build_footer()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.show_page("Dashboard")
        # Build the remaining pages after the first paint, so the first visit
        # does not trigger a visible burst of widget creation.
        self._warm_page_queue = [page for page in self.NAV_ITEMS if page != "Dashboard"]
        self.after(700, self._warm_next_page)
        self.after(120, self._poll_events)

    def _build_topbar(self) -> None:
        bar = ctk.CTkFrame(
            self, fg_color=COLORS["surface"], corner_radius=0, height=72,
            border_width=0,
        )
        bar.grid(row=0, column=0, sticky="ew")
        bar.grid_propagate(False)
        bar.grid_columnconfigure(2, weight=1)

        brand = ctk.CTkFrame(bar, fg_color="transparent")
        brand.grid(row=0, column=0, padx=(28, 22), sticky="w")
        ctk.CTkLabel(
            brand, text="✳", text_color=COLORS["accent"],
            font=ctk.CTkFont(size=30, weight="bold"),
        ).pack(side="left", padx=(0, 10))
        ctk.CTkLabel(
            brand, text="Industrial Pose", text_color=COLORS["text"],
            font=ctk.CTkFont(size=19, weight="bold"),
        ).pack(side="left")
        ctk.CTkFrame(bar, width=1, height=30, fg_color=COLORS["border"]).grid(
            row=0, column=1, padx=(0, 14)
        )

        nav = ctk.CTkFrame(bar, fg_color="transparent")
        nav.grid(row=0, column=2, sticky="w")
        self.nav_buttons: dict[str, ctk.CTkButton] = {}
        nav_icons = {"Dashboard": "▦", "Video Analysis": "▶", "Analytics": "▤", "Settings": "⚙"}
        for item in self.NAV_ITEMS:
            button = ctk.CTkButton(
                nav,
                text=f"{nav_icons[item]}   {item}",
                command=lambda page=item: self.show_page(page),
                width=142 if item != "Video Analysis" else 168,
                height=40,
                corner_radius=9,
                fg_color=COLORS["surface_alt"] if item == "Dashboard" else "transparent",
                hover_color=COLORS["surface_alt"],
                text_color=COLORS["text"] if item == "Dashboard" else COLORS["muted"],
                font=ctk.CTkFont(size=13, weight="bold" if item == "Dashboard" else "normal"),
            )
            button.pack(side="left", padx=3)
            self.nav_buttons[item] = button

        right = ctk.CTkFrame(bar, fg_color="transparent")
        right.grid(row=0, column=3, padx=(14, 28), sticky="e")
        ctk.CTkLabel(
            right, text="◈", text_color=COLORS["accent"],
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(
            right, text="Local Workspace", text_color=COLORS["text"],
            font=ctk.CTkFont(size=12),
        ).pack(side="left", padx=(0, 14))
        ctk.CTkFrame(right, width=1, height=28, fg_color=COLORS["border"]).pack(
            side="left", padx=(0, 12)
        )
        ctk.CTkLabel(
            right, text="IPA", width=36, height=36, corner_radius=10,
            fg_color="#29363b", text_color=COLORS["accent"],
            font=ctk.CTkFont(size=11, weight="bold"),
        ).pack(side="left")

    def _build_footer(self) -> None:
        footer = ctk.CTkFrame(self, fg_color=COLORS["surface"], corner_radius=0, height=34)
        footer.place(relx=0, rely=1, relwidth=1, anchor="sw")
        footer.grid_propagate(False)
        ctk.CTkLabel(
            footer, text="Industrial Pose Analytics  ·  Processamento local",
            text_color=COLORS["muted"], font=ctk.CTkFont(size=11),
        ).pack(side="left", padx=28)
        self.footer_status = ctk.CTkLabel(
            footer, text=f"Versão {APP_VERSION}   •   Pronto",
            text_color=COLORS["muted"], font=ctk.CTkFont(size=11),
        )
        self.footer_status.pack(side="right", padx=28)

    def show_page(self, page: str) -> None:
        """Show a cached page, building it only on first visit or when stale."""
        self.active_page = page
        for name, frame in self.page_frames.items():
            if name != page:
                frame.grid_remove()
        for item, button in self.nav_buttons.items():
            active = item == page
            button.configure(
                fg_color=COLORS["surface_alt"] if active else "transparent",
                text_color=COLORS["text"] if active else COLORS["muted"],
                font=ctk.CTkFont(size=13, weight="bold" if active else "normal"),
            )
        frame = self.page_frames.get(page)
        if frame is None:
            frame = ctk.CTkFrame(self.content_host, fg_color="transparent", corner_radius=0)
            frame.grid_columnconfigure(0, weight=1)
            self.page_frames[page] = frame
        self.page_host = frame
        self._ensure_page_built(page, frame)
        frame.grid(row=0, column=0, sticky="nsew")

    def _ensure_page_built(self, page: str, frame: ctk.CTkFrame) -> None:
        if page in self.built_pages and page not in self.dirty_pages:
            return
        frame.grid_remove()
        for row in range(4):
            frame.grid_rowconfigure(row, weight=0, minsize=0)
        for child in frame.winfo_children():
            child.destroy()
        builders = {
            "Dashboard": self._build_dashboard,
            "Video Analysis": self._build_video_page,
            "Analytics": self._build_analytics_page,
            "Settings": self._build_settings_page,
        }
        previous_host = self.page_host
        self.page_host = frame
        try:
            builders[page]()
        finally:
            self.page_host = previous_host
        self.built_pages.add(page)
        self.dirty_pages.discard(page)

    def _warm_next_page(self) -> None:
        if not self._warm_page_queue:
            return
        if self.worker and self.worker.is_alive():
            self.after(500, self._warm_next_page)
            return
        page = self._warm_page_queue.pop(0)
        frame = self.page_frames.get(page)
        if frame is None:
            frame = ctk.CTkFrame(self.content_host, fg_color="transparent", corner_radius=0)
            frame.grid_columnconfigure(0, weight=1)
            self.page_frames[page] = frame
        self._ensure_page_built(page, frame)
        if self._warm_page_queue:
            self.after(250, self._warm_next_page)

    def _page_header(self, title: str, subtitle: str, action: bool = False) -> None:
        header = ctk.CTkFrame(self.page_host, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", pady=(10, 16))
        header.grid_columnconfigure(0, weight=1)
        text = ctk.CTkFrame(header, fg_color="transparent")
        text.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            text, text=title, text_color=COLORS["text"],
            font=ctk.CTkFont(size=27, weight="bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            text, text=subtitle, text_color=COLORS["muted"],
            font=ctk.CTkFont(size=13),
        ).pack(anchor="w", pady=(4, 0))
        if action:
            self._button(
                header, "＋  Selecionar vídeo", self._select_video,
                primary=True, width=178,
            ).grid(row=0, column=1, sticky="e")

    def _build_dashboard(self) -> None:
        self._page_header(
            "Industrial Pose Analytics",
            "Visibilidade ergonômica para operações mais seguras e eficientes.",
            action=True,
        )
        badges = ctk.CTkFrame(self.page_host, fg_color="transparent")
        badges.grid(row=1, column=0, sticky="ew", pady=(0, 17))
        for label in ("COMPUTER VISION", "POSE LANDMARKER", "PROCESSAMENTO OFFLINE", "MODELO FULL"):
            ctk.CTkLabel(
                badges, text=label, height=28, fg_color=COLORS["surface_alt"],
                text_color=COLORS["muted"], corner_radius=5,
                font=ctk.CTkFont(size=10, weight="bold"),
            ).pack(side="left", padx=(0, 7))

        body = ctk.CTkScrollableFrame(
            self.page_host, fg_color="transparent", corner_radius=0,
            scrollbar_button_color=COLORS["border"],
        )
        body.grid(row=2, column=0, sticky="nsew")
        self.page_host.grid_rowconfigure(2, weight=1)
        body.grid_columnconfigure((0, 1, 2, 3), weight=1, uniform="metrics")
        run_count = len(self.session_runs)
        frames = sum(run.processed_frames for run in self.session_runs)
        mean_detection = (
            sum(run.detection_rate for run in self.session_runs) / run_count if run_count else 0
        )
        mean_fps = (
            sum(run.effective_fps for run in self.session_runs) / run_count if run_count else 0
        )
        metrics = [
            ("Vídeos processados", str(run_count), "▤", "nesta sessão"),
            ("Frames analisados", f"{frames:,}".replace(",", "."), "⌕", "processamento offline"),
            ("Taxa de detecção", f"{mean_detection:.1f}%", "◉", "média da sessão"),
            ("FPS efetivo", f"{mean_fps:.1f}" if run_count else "—", "ϟ", "velocidade média"),
        ]
        for column, (title, value, icon, foot) in enumerate(metrics):
            card = self._card(body)
            card.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 8, 8), pady=(0, 16))
            content = self._card_content(card)
            content.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(
                content, text=icon, width=36, height=36, corner_radius=9,
                fg_color="#26353a", text_color=COLORS["accent"],
                font=ctk.CTkFont(size=18, weight="bold"),
            ).grid(row=0, column=0, sticky="w")
            ctk.CTkLabel(
                content, text="● pronto", text_color=COLORS["green"],
                font=ctk.CTkFont(size=10, weight="bold"),
            ).grid(row=0, column=1, sticky="e")
            ctk.CTkLabel(
                content, text=title, text_color=COLORS["muted"],
                font=ctk.CTkFont(size=12),
            ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(16, 2))
            ctk.CTkLabel(
                content, text=value, text_color=COLORS["text"],
                font=ctk.CTkFont(size=27, weight="bold"),
            ).grid(row=2, column=0, columnspan=2, sticky="w")
            ctk.CTkLabel(
                content, text=foot, text_color=COLORS["muted"],
                font=ctk.CTkFont(size=10),
            ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(1, 0))

        charts = ctk.CTkFrame(body, fg_color="transparent")
        charts.grid(row=1, column=0, columnspan=4, sticky="ew")
        charts.grid_columnconfigure((0, 1), weight=1, uniform="chart")
        detection_card = self._card(charts)
        detection_card.grid(row=0, column=0, sticky="nsew", padx=(0, 9), pady=(0, 16))
        rate_card = self._card(charts)
        rate_card.grid(row=0, column=1, sticky="nsew", padx=(9, 0), pady=(0, 16))
        detection_content = self._card_content(detection_card)
        rate_content = self._card_content(rate_card)
        self._chart_title(detection_content, "Detecção por frame", "Frames com e sem pose no último vídeo")
        self._chart_title(rate_content, "Taxa de detecção", "Evolução das análises desta sessão")
        self._draw_detection_chart(detection_content)
        self._draw_rate_chart(rate_content)

        insight = self._card(body)
        insight.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(0, 14))
        insight_content = self._card_content(insight)
        insight_content.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            insight_content, text="Status do ambiente", text_color=COLORS["text"],
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, sticky="w")
        model_ready = MODEL_PATH.is_file()
        ctk.CTkLabel(
            insight_content,
            text=("●  Pose Landmarker Full disponível" if model_ready else "●  Modelo ausente"),
            text_color=COLORS["green"] if model_ready else COLORS["red"],
            font=ctk.CTkFont(size=12, weight="bold"),
        ).grid(row=1, column=0, sticky="w", pady=(12, 4))
        latest = self.last_video_name
        ctk.CTkLabel(
            insight_content, text=f"Último vídeo  ·  {latest}", text_color=COLORS["muted"],
            font=ctk.CTkFont(size=12),
        ).grid(row=2, column=0, sticky="w")
        self._button(
            insight_content, "Abrir análise de vídeo  →", lambda: self.show_page("Video Analysis"),
            primary=True, width=205,
        ).grid(row=0, column=1, rowspan=3, sticky="e", padx=(12, 0))

    def _chart_title(self, parent, title: str, subtitle: str) -> None:
        ctk.CTkLabel(
            parent, text=title, text_color=COLORS["text"],
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            parent, text=subtitle, text_color=COLORS["muted"],
            font=ctk.CTkFont(size=11),
        ).pack(anchor="w", pady=(3, 8))

    def _chart_axes(self, figure):
        figure.patch.set_facecolor(COLORS["surface"])
        axis = figure.add_subplot(111)
        axis.set_facecolor(COLORS["surface"])
        axis.tick_params(colors=COLORS["muted"], labelsize=9)
        for edge in axis.spines.values():
            edge.set_color(COLORS["border"])
        axis.grid(axis="y", color=COLORS["chart_grid"], linestyle="--", linewidth=0.7)
        axis.set_axisbelow(True)
        return axis

    def _embed_chart(self, parent, figure) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        canvas = FigureCanvasTkAgg(figure, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x", expand=True, pady=(4, 0))

    def _draw_detection_chart(self, parent) -> None:
        from matplotlib.figure import Figure

        figure = Figure(figsize=(5.5, 2.4), dpi=100, layout="constrained")
        axis = self._chart_axes(figure)
        if self.session_runs:
            run = self.session_runs[-1]
            names = ["Pose detectada", "Sem detecção"]
            values = [run.detected_frames, max(run.processed_frames - run.detected_frames, 0)]
        else:
            names, values = ["Pose detectada", "Sem detecção"], [0, 0]
        bars = axis.bar(names, values, color=[COLORS["accent"], "#596168"], width=0.48)
        axis.set_ylim(bottom=0, top=max(max(values, default=0) * 1.18, 1))
        axis.set_ylabel("Frames", color=COLORS["muted"], fontsize=9)
        axis.bar_label(bars, padding=3, color=COLORS["text"], fontsize=9)
        self._embed_chart(parent, figure)

    def _draw_rate_chart(self, parent) -> None:
        from matplotlib.figure import Figure

        figure = Figure(figsize=(5.5, 2.4), dpi=100, layout="constrained")
        axis = self._chart_axes(figure)
        values = [run.detection_rate for run in self.session_runs]
        if values:
            axis.plot(range(1, len(values) + 1), values, color=COLORS["accent"], linewidth=2.3)
            axis.scatter(range(1, len(values) + 1), values, color=COLORS["accent"], s=24, zorder=3)
            axis.set_xticks(range(1, len(values) + 1))
        else:
            axis.text(0.5, 0.52, "As análises concluídas aparecerão aqui", ha="center", va="center",
                      transform=axis.transAxes, color=COLORS["muted"], fontsize=9)
            axis.set_xlim(0, 2)
        axis.set_ylim(0, 100)
        axis.set_xlabel("Análise da sessão", color=COLORS["muted"], fontsize=9)
        axis.set_ylabel("Detecção (%)", color=COLORS["muted"], fontsize=9)
        self._embed_chart(parent, figure)

    def _build_video_page(self) -> None:
        self._page_header(
            "Video Analysis",
            "Selecione uma atividade, acompanhe o processamento e exporte os resultados.",
        )
        shell = ctk.CTkScrollableFrame(self.page_host, fg_color="transparent", corner_radius=0)
        shell.grid(row=1, column=0, sticky="nsew")
        self.page_host.grid_rowconfigure(1, weight=1)
        shell.grid_columnconfigure((0, 1), weight=1, uniform="video-columns")
        shell.grid_rowconfigure(0, weight=1)

        input_card = self._card(shell)
        input_card.grid(row=0, column=0, sticky="nsew", padx=(0, 9), pady=(0, 16))
        input_content = self._card_content(input_card)
        processing_card = self._card(shell)
        processing_card.grid(row=0, column=1, sticky="nsew", padx=(9, 0), pady=(0, 16))
        processing_content = self._card_content(processing_card)
        self._section_heading(input_content, "Video Input", "Arquivo, propriedades e prévia")
        self._button(input_content, "＋  Selecionar arquivo de vídeo", self._select_video, primary=True).pack(
            fill="x", pady=(16, 12)
        )
        self.preview_label = ctk.CTkLabel(
            input_content, text="Selecione um vídeo para visualizar a prévia",
            fg_color="#0d0f11", text_color=COLORS["muted"], corner_radius=10,
            height=310, font=ctk.CTkFont(size=13),
        )
        self.preview_label.pack(fill="both", expand=True, pady=(0, 12))
        if self.video_path is not None and self.preview_image is not None:
            self.preview_label.configure(image=self.preview_image, text="")
        self.file_name_label = ctk.CTkLabel(
            input_content, text="Nenhum vídeo selecionado", anchor="w",
            text_color=COLORS["text"], font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.file_name_label.pack(fill="x", pady=(0, 10))
        self.file_info_label = ctk.CTkLabel(
            input_content, text="Formato: MP4, AVI ou MOV  ·  aguardando arquivo",
            anchor="w", text_color=COLORS["muted"], font=ctk.CTkFont(size=11),
        )
        self.file_info_label.pack(fill="x", pady=(0, 16))
        path_row = ctk.CTkFrame(input_content, fg_color="transparent")
        path_row.pack(fill="x")
        path_row.grid_columnconfigure(0, weight=1)
        self.output_label = ctk.CTkLabel(
            path_row, text=f"Saída: {self.output_dir}", anchor="w",
            text_color=COLORS["muted"], font=ctk.CTkFont(size=10),
        )
        self.output_label.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self._button(path_row, "Pasta…", self._select_output).grid(row=0, column=1, sticky="e")

        self._section_heading(processing_content, "Processing", "Execução frame a frame com MediaPipe Tasks")
        status = ctk.CTkFrame(processing_content, fg_color=COLORS["surface_alt"], corner_radius=9)
        status.pack(fill="x", pady=(16, 18))
        self.processing_status = ctk.CTkLabel(
            status, text="●  Pronto para processar", anchor="w",
            text_color=COLORS["green"], font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.processing_status.pack(fill="x", padx=12, pady=11)
        self.progress_bar = ctk.CTkProgressBar(
            processing_content, height=10, corner_radius=6,
            fg_color=COLORS["surface_alt"], progress_color=COLORS["accent"],
        )
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", pady=(0, 7))
        self.percent_label = ctk.CTkLabel(
            processing_content, text="0%", anchor="e", text_color=COLORS["muted"],
            font=ctk.CTkFont(size=11, weight="bold"),
        )
        self.percent_label.pack(fill="x", pady=(0, 14))

        self.progress_text = ctk.CTkLabel(
            processing_content, text="Frames processados     0 / 0\nFrames com pose          0\nFPS efetivo                   —\nTempo decorrido           00:00",
            justify="left", anchor="w", text_color=COLORS["muted"],
            font=ctk.CTkFont(size=12, family="Consolas"),
        )
        self.progress_text.pack(fill="x", pady=(0, 18))
        button_row = ctk.CTkFrame(processing_content, fg_color="transparent")
        button_row.pack(fill="x", pady=(0, 20))
        button_row.grid_columnconfigure((0, 1), weight=1)
        self.start_button = self._button(
            button_row, "▶  Start Processing", self._start_processing, primary=True
        )
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.cancel_button = self._button(
            button_row, "Cancel Processing", self._cancel_processing
        )
        self.cancel_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))
        self.cancel_button.configure(state="disabled")

        self._section_heading(processing_content, "Results", "Saídas e estatísticas da análise")
        self.results_text = ctk.CTkLabel(
            processing_content,
            text="Vídeo processado: aguardando\nCSV de landmarks: aguardando\nTaxa de detecção: —\nTempo total: —",
            justify="left", anchor="w", text_color=COLORS["muted"],
            font=ctk.CTkFont(size=11),
        )
        self.results_text.pack(fill="x", pady=(13, 13))
        result_buttons = ctk.CTkFrame(processing_content, fg_color="transparent")
        result_buttons.pack(fill="x")
        self._button(result_buttons, "Abrir pasta de saída", self._open_output).pack(side="left")
        self.open_video_button = self._button(
            result_buttons, "▶  Visualizar resultado", self._open_processed_video, primary=True
        )
        self.open_video_button.pack(side="right")
        self.open_video_button.configure(state="disabled")
        self._update_file_labels()

    def _build_analytics_page(self) -> None:
        self._page_header("Analytics", "Indicadores ergonômicos e análise temporal")
        body = ctk.CTkScrollableFrame(self.page_host, fg_color="transparent", corner_radius=0)
        body.grid(row=1, column=0, sticky="nsew")
        self.page_host.grid_rowconfigure(1, weight=1)
        body.grid_columnconfigure(0, weight=1)

        if not self.latest_metrics_path or not self.latest_metrics_path.is_file():
            empty = self._card(body)
            empty.grid(row=0, column=0, sticky="ew", pady=(0, 14))
            empty_content = self._card_content(empty)
            self._section_heading(
                empty_content,
                "Aguardando análise de vídeo",
                "Os indicadores aparecem aqui quando um vídeo tiver sido processado.",
            )
            self._button(
                empty_content, "Abrir Video Analysis  →",
                lambda: self.show_page("Video Analysis"), primary=True, width=205,
            ).pack(anchor="w", pady=(18, 0))
            return

        source = self.latest_source_video or self.latest_metrics_path.name
        source_card = self._card(body)
        source_card.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        source_content = self._card_content(source_card)
        self._section_heading(
            source_content,
            "Indicadores cinemáticos",
            f"Vídeo: {source}  ·  {self.latest_metrics_path.name}",
        )
        self._button(
            source_content, "Exportar relatório PDF", self._export_analysis_report,
            primary=True, width=190,
        ).pack(anchor="e", pady=(12, 0))

        cards = ctk.CTkFrame(body, fg_color="transparent")
        cards.grid(row=1, column=0, sticky="ew", pady=(0, 14))
        cards.grid_columnconfigure((0, 1, 2), weight=1, uniform="ergonomic-metrics")
        for index, (group, fields) in enumerate(METRIC_GROUPS.items()):
            card = self._card(cards)
            card.grid(row=index // 3, column=index % 3, sticky="nsew",
                      padx=(0 if index % 3 == 0 else 8, 8), pady=(0, 12))
            content = self._card_content(card)
            self._build_metric_summary(content, group, fields)

        chart = self._card(body)
        chart.grid(row=2, column=0, sticky="ew", pady=(0, 14))
        chart_content = self._card_content(chart)
        self._section_heading(chart_content, "Evolução temporal", "Selecione um indicador para explorar os ângulos ao longo do vídeo.")
        selector_row = ctk.CTkFrame(chart_content, fg_color="transparent")
        selector_row.pack(fill="x", pady=(13, 8))
        ctk.CTkLabel(
            selector_row, text="Indicador", text_color=COLORS["muted"],
            font=ctk.CTkFont(size=11),
        ).pack(side="left", padx=(0, 10))
        self.metric_choice = ctk.CTkOptionMenu(
            selector_row, values=list(METRIC_LABELS.values()),
            command=self._on_metric_selected, width=260, height=36,
        )
        self.metric_choice.set(METRIC_LABELS["trunk_inclination_deg"])
        self.metric_choice.pack(side="left")
        ctk.CTkLabel(
            selector_row, text="Filtro", text_color=COLORS["muted"],
            font=ctk.CTkFont(size=11),
        ).pack(side="left", padx=(18, 8))
        self.filter_choice = ctk.CTkOptionMenu(
            selector_row,
            values=["Original", "Média móvel", "Savitzky-Golay"],
            command=self._on_filter_selected, width=180, height=36,
        )
        self.filter_choice.set("Savitzky-Golay")
        self.filter_choice.pack(side="left")
        ctk.CTkLabel(
            selector_row, text="Janela (ímpar p/ SG)", text_color=COLORS["muted"],
            font=ctk.CTkFont(size=11),
        ).pack(side="left", padx=(16, 6))
        self.filter_window_entry = ctk.CTkEntry(selector_row, width=58, height=36)
        self.filter_window_entry.insert(0, "5")
        self.filter_window_entry.pack(side="left")
        ctk.CTkLabel(
            selector_row, text="Ordem SG", text_color=COLORS["muted"],
            font=ctk.CTkFont(size=11),
        ).pack(side="left", padx=(10, 6))
        self.filter_order_entry = ctk.CTkEntry(selector_row, width=48, height=36)
        self.filter_order_entry.insert(0, "2")
        self.filter_order_entry.pack(side="left")
        self._button(selector_row, "Aplicar", self._apply_temporal_filter, width=88).pack(
            side="left", padx=(8, 0)
        )
        self.metric_chart_host = ctk.CTkFrame(chart_content, fg_color="transparent")
        self.metric_chart_host.pack(fill="x", expand=True, pady=(2, 3))
        self._draw_metric_series("trunk_inclination_deg")

        note = self._card(body)
        note.grid(row=3, column=0, sticky="ew", pady=(0, 14))
        note_content = self._card_content(note)
        ctk.CTkLabel(
            note_content,
            text=("Ângulos em graus calculados a partir da projeção 2D dos landmarks. "
                  "A orientação da câmera influencia a leitura; os valores são medidas geométricas "
                  "e não classificações de risco ergonômico. Esquerdo/direito indicam o lado anatômico "
                  "identificado pelo MediaPipe, podendo aparecer invertido em vídeo frontal ou espelhado."),
            text_color=COLORS["muted"], wraplength=1100, justify="left",
            font=ctk.CTkFont(size=11),
        ).pack(anchor="w")

    def _build_metric_summary(self, parent, group: str, fields: tuple[str, ...]) -> None:
        ctk.CTkLabel(
            parent, text=group, text_color=COLORS["text"],
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", pady=(0, 10))
        for field in fields:
            stats = self.latest_metric_summary.get(field, {})
            mean_value = stats.get("mean")
            side = "E" if field.startswith("left_") else "D" if field.startswith("right_") else ""
            prefix = f"{side}  " if side else ""
            if mean_value is None:
                summary_text = f"{prefix}Estatísticas indisponíveis (<{2} amostras válidas)"
            else:
                summary_text = (
                    f"{prefix}Média {mean_value:.1f}°  ·  Mediana {stats['median']:.1f}°"
                )
            ctk.CTkLabel(
                parent, text=summary_text, anchor="w",
                text_color=COLORS["accent"] if mean_value is not None else COLORS["muted"],
                font=ctk.CTkFont(size=11, weight="bold"),
            ).pack(fill="x", pady=(0, 3))
            if mean_value is not None:
                ctk.CTkLabel(
                    parent,
                    text=(f"DP {stats['std']:.1f}°  ·  min {stats['min']:.1f}°  ·  "
                          f"máx {stats['max']:.1f}°"),
                    anchor="w", text_color=COLORS["muted"], font=ctk.CTkFont(size=10),
                ).pack(fill="x", pady=(0, 3))
            coverage = stats.get("coverage_pct", 0.0)
            ctk.CTkLabel(
                parent,
                text=(f"Cobertura {coverage:.1f}%  ·  válidos {stats.get('valid_frames', stats.get('samples', 0))}"
                      f" / inválidos {stats.get('invalid_frames', stats.get('total_frames', 0) - stats.get('samples', 0))}"),
                anchor="w", text_color=COLORS["muted"],
                font=ctk.CTkFont(size=9),
            ).pack(fill="x", pady=(0, 7))

    def _on_metric_selected(self, label: str) -> None:
        field = next((name for name, text in METRIC_LABELS.items() if text == label), None)
        if field:
            self._draw_metric_series(field)

    def _on_filter_selected(self, _label: str) -> None:
        field = getattr(self, "_current_metric_field", "trunk_inclination_deg")
        self._draw_metric_series(field)

    def _apply_temporal_filter(self) -> None:
        field = getattr(self, "_current_metric_field", "trunk_inclination_deg")
        try:
            self._filter_settings()
        except ValueError as error:
            messagebox.showerror("Parâmetros do filtro", str(error))
            return
        self._draw_metric_series(field)

    def _filter_settings(self) -> tuple[str, int, int]:
        label = self.filter_choice.get() if hasattr(self, "filter_choice") else "Savitzky-Golay"
        method = {
            "Original": "none",
            "Média móvel": "moving_average",
            "Savitzky-Golay": "savgol",
        }.get(label, "savgol")
        try:
            window = int(self.filter_window_entry.get())
        except (AttributeError, ValueError):
            window = 5
        try:
            order = int(self.filter_order_entry.get())
        except (AttributeError, ValueError):
            order = 2
        if method == "savgol" and (window < 3 or window % 2 == 0):
            raise ValueError("Savitzky-Golay exige uma janela ímpar de pelo menos 3.")
        if method == "savgol" and not 0 <= order < window:
            raise ValueError("A ordem Savitzky-Golay deve ser menor que a janela.")
        if window < 1:
            raise ValueError("A janela do filtro deve ser maior que zero.")
        return method, window, order

    def _draw_metric_series(self, field: str) -> None:
        if not getattr(self, "metric_chart_host", None):
            return
        for child in self.metric_chart_host.winfo_children():
            child.destroy()
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        figure = Figure(figsize=(9.6, 3.1), dpi=100, layout="constrained")
        axis = self._chart_axes(figure)
        times, values = self._read_metric_series(field)
        self._current_metric_field = field
        try:
            method, window, order = self._filter_settings()
        except ValueError as error:
            method, window, order = "none", 5, 2
            if hasattr(self, "filter_window_entry"):
                self.filter_window_entry.configure(border_color=COLORS["red"])
        else:
            if hasattr(self, "filter_window_entry"):
                self.filter_window_entry.configure(border_color=COLORS["border"])
        filtered = smooth_series(values, method, window, order)
        if values and any(value is not None and math.isfinite(value) for value in values):
            stride = max(1, (len(times) + 1799) // 1800)
            axis.plot(times[::stride], values[::stride], color=COLORS["muted"],
                      linewidth=1.0, alpha=0.75, label="Original")
            if method != "none":
                axis.plot(times[::stride], filtered[::stride], color=COLORS["accent"],
                          linewidth=1.8, label="Suavizado")
            summary = self.latest_metric_summary.get(field, {})
            mean_value = summary.get("mean")
            if mean_value is not None:
                axis.axhline(mean_value, color="#f2b866", linewidth=1.1, linestyle="--",
                             alpha=0.8, label=f"Média {mean_value:.1f}°")
            axis.legend(frameon=False, labelcolor=COLORS["muted"], fontsize=8, loc="upper right")
            axis.set_xlim(left=0)
            axis.set_ylim(bottom=0)
        else:
            axis.text(0.5, 0.5, "Não há landmarks suficientes para este indicador",
                      transform=axis.transAxes, ha="center", va="center",
                      color=COLORS["muted"], fontsize=10)
        axis.set_title(METRIC_LABELS[field], color=COLORS["text"], loc="left", fontsize=11)
        axis.set_xlabel("Tempo do vídeo (s)", color=COLORS["muted"], fontsize=9)
        axis.set_ylabel("Ângulo (°)", color=COLORS["muted"], fontsize=9)
        original_stats = descriptive_statistics(values)
        if original_stats["mean"] is None:
            stats_line = (f"Original: amostras válidas insuficientes · cobertura "
                          f"{original_stats['coverage_pct']:.1f}% "
                          f"({original_stats['valid_frames']}/{original_stats['total_frames']})")
        else:
            stats_line = (
                f"Original — média {original_stats['mean']:.1f}° · mediana {original_stats['median']:.1f}° · "
                f"DP {original_stats['std']:.1f}° · min {original_stats['min']:.1f}° · "
                f"máx {original_stats['max']:.1f}° · cobertura {original_stats['coverage_pct']:.1f}% "
                f"({original_stats['valid_frames']} válidos / {original_stats['invalid_frames']} inválidos)"
            )
        if method != "none":
            filtered_stats = descriptive_statistics(filtered)
            if filtered_stats["mean"] is not None:
                stats_line += (
                    f"\nSuavizado — média {filtered_stats['mean']:.1f}° · mediana {filtered_stats['median']:.1f}° · "
                    f"DP {filtered_stats['std']:.1f}° · min {filtered_stats['min']:.1f}° · "
                    f"máx {filtered_stats['max']:.1f}°"
                )
        ctk.CTkLabel(
            self.metric_chart_host, text=stats_line, anchor="w", justify="left",
            text_color=COLORS["muted"], font=ctk.CTkFont(size=10),
        ).pack(fill="x", pady=(0, 7))
        canvas = FigureCanvasTkAgg(figure, master=self.metric_chart_host)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x", expand=True)
        self._metric_figure = figure

    def _read_metric_series(self, field: str):
        if not self.latest_metrics_path or not self.latest_metrics_path.is_file():
            return [], []
        times, values = [], []
        with self.latest_metrics_path.open(newline="", encoding="utf-8-sig") as source:
            for index, row in enumerate(csv.DictReader(source)):
                value = row.get(field, "")
                timestamp = row.get("timestamp_ms", "")
                try:
                    time_seconds = float(timestamp or 0) / 1000.0
                except ValueError:
                    time_seconds = index / 30.0
                times.append(time_seconds)
                try:
                    numeric_value = float(value) if value else float("nan")
                    values.append(numeric_value if math.isfinite(numeric_value) else float("nan"))
                except (TypeError, ValueError):
                    values.append(float("nan"))
        return times, values

    def _export_analysis_report(self) -> None:
        if not self.latest_metrics_path or not self.latest_metrics_path.is_file():
            messagebox.showinfo("Relatório indisponível", "Não há um CSV de indicadores para exportar.")
            return
        try:
            method, window, order = self._filter_settings()
        except ValueError as error:
            messagebox.showerror("Parâmetros do filtro", str(error))
            return
        source_stem = self.latest_source_video or self.latest_metrics_path.stem
        initial_name = f"{source_stem}_relatorio.pdf"
        destination = filedialog.asksaveasfilename(
            title="Exportar relatório da análise",
            initialdir=str(self.latest_metrics_path.parent),
            initialfile=initial_name,
            defaultextension=".pdf",
            filetypes=[("Documento PDF", "*.pdf")],
        )
        if not destination:
            return
        try:
            from analytics.report import export_analysis_pdf

            info = dict(self.latest_processing_info)
            info.setdefault("processed_frames", self.latest_metric_summary.get(
                "trunk_inclination_deg", {}).get("total_frames", "não informado"))
            info.setdefault("effective_fps", "não informado")
            info.setdefault("elapsed_seconds", "não informado")
            info.setdefault("detected_frames", "não informado")
            info.setdefault("detection_rate", "não informada")
            export_analysis_pdf(
                self.latest_metrics_path, Path(destination),
                self.last_video_name if self.last_video_name != "Nenhum vídeo processado"
                else (self.latest_source_video or self.latest_metrics_path.name),
                info, method, window, order,
            )
        except (OSError, ValueError) as error:
            messagebox.showerror("Falha ao exportar PDF", str(error))
            return
        messagebox.showinfo("Relatório exportado", f"PDF salvo em:\n{destination}")

    def _restore_latest_metrics(self) -> None:
        """Load an existing metrics file or analyze the last exported landmarks."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        metric_files = sorted(
            OUTPUT_DIR.glob("*_ergonomic_metrics_*.csv"),
            key=lambda path: path.stat().st_mtime,
        )
        if metric_files:
            path = metric_files[-1]
            self.latest_metrics_path = path
            self.latest_metric_summary = PostureAnalyzer.summarize_metrics_csv(path)
            total_frames = self.latest_metric_summary.get("trunk_inclination_deg", {}).get(
                "total_frames", 0)
            self.latest_processing_info = {
                "processed_frames": total_frames,
                "total_frames": total_frames,
                "application_version": APP_VERSION,
            }
            with path.open(newline="", encoding="utf-8-sig") as source:
                first_row = next(csv.DictReader(source), {})
            if first_row.get("confidence_threshold"):
                self.latest_processing_info["confidence_threshold"] = first_row[
                    "confidence_threshold"
                ]
            self.latest_source_video = path.name.split("_ergonomic_metrics_", 1)[0]
            run_id = path.stem.rsplit("_ergonomic_metrics_", 1)[-1]
            processed_video = OUTPUT_DIR / f"{self.latest_source_video}_pose_{run_id}.mp4"
            if processed_video.is_file():
                self.last_video_path = processed_video
                capture = cv2.VideoCapture(str(processed_video))
                if capture.isOpened():
                    video_fps = capture.get(cv2.CAP_PROP_FPS)
                    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    if video_fps > 0:
                        self.latest_processing_info["video_fps"] = f"{video_fps:.2f}"
                    if video_frames > 0 and video_fps > 0:
                        self.latest_processing_info["video_duration"] = f"{video_frames / video_fps:.2f}"
                capture.release()
            self.last_video_name = self.latest_source_video
            return

        landmark_files = sorted(
            OUTPUT_DIR.glob("*_landmarks_*.csv"), key=lambda path: path.stat().st_mtime
        )
        if not landmark_files:
            return
        landmarks_csv = landmark_files[-1]
        prefix, run_id = landmarks_csv.stem.split("_landmarks_", 1)
        processed_video = OUTPUT_DIR / f"{prefix}_pose_{run_id}.mp4"
        width, height = 1, 1
        total_frames, fps = None, 30.0
        if processed_video.is_file():
            capture = cv2.VideoCapture(str(processed_video))
            if capture.isOpened():
                width = max(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 1)
                height = max(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 1)
                total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            capture.release()
            self.last_video_path = processed_video
        metrics_csv = OUTPUT_DIR / f"{prefix}_ergonomic_metrics_{run_id}.csv"
        analyzer = PostureAnalyzer(self.settings.visibility_threshold)
        self.latest_metric_summary = analyzer.rebuild_from_landmarks_csv(
            landmarks_csv, metrics_csv, width, height, total_frames, fps
        )
        self.latest_metrics_path = metrics_csv
        self.latest_source_video = prefix
        self.last_video_name = prefix

    def _build_settings_page(self) -> None:
        self._page_header("Settings", "Parâmetros do modelo e preferências visuais")
        card = self._card(self.page_host)
        card.grid(row=1, column=0, sticky="nw", pady=8)
        content = self._card_content(card)
        ctk.CTkLabel(
            content, text="Parâmetros de detecção", text_color=COLORS["text"],
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(anchor="w", pady=(4, 5))
        ctk.CTkLabel(
            content, text="Valores de confiança entre 0 e 1. Aplicados na próxima análise.",
            text_color=COLORS["muted"], font=ctk.CTkFont(size=11),
        ).pack(anchor="w", pady=(0, 15))

        self.setting_entries = {}
        setting_fields = (
            ("detection_confidence", "Detection confidence"),
            ("presence_confidence", "Presence confidence"),
            ("tracking_confidence", "Tracking confidence"),
            ("visibility_threshold", "Minimum landmark confidence"),
        )
        for key, label in setting_fields:
            row = ctk.CTkFrame(content, fg_color="transparent")
            row.pack(fill="x", pady=6)
            ctk.CTkLabel(
                row, text=label, width=235, anchor="w", text_color=COLORS["text"],
                font=ctk.CTkFont(size=12),
            ).pack(side="left")
            entry = ctk.CTkEntry(row, width=120, height=36, corner_radius=8)
            entry.insert(0, str(getattr(self.settings, key)))
            entry.pack(side="right")
            self.setting_entries[key] = entry
        ctk.CTkLabel(
            content, text="Tema", text_color=COLORS["text"], font=ctk.CTkFont(size=12),
        ).pack(anchor="w", pady=(16, 5))
        self.theme_option = ctk.CTkOptionMenu(
            card, values=["Dark", "Light", "System"], command=ctk.set_appearance_mode,
            width=160, height=36, corner_radius=8,
        )
        self.theme_option.set("Dark")
        self.theme_option.pack(anchor="w", pady=(0, 16))
        self._button(content, "Salvar parâmetros", self._save_settings, primary=True).pack(anchor="w")

    def _save_settings(self) -> None:
        try:
            values = {
                key: float(entry.get()) for key, entry in self.setting_entries.items()
            }
            updated = DetectionSettings(**values)
            updated.validate()
        except (ValueError, TypeError) as error:
            messagebox.showerror("Parâmetros inválidos", str(error))
            return
        self.settings = updated
        messagebox.showinfo("Configurações salvas", "Os novos parâmetros serão usados na próxima análise.")

    def _select_video(self) -> None:
        selected = filedialog.askopenfilename(
            title="Selecionar vídeo", initialdir=str(ROOT_DIR),
            filetypes=[("Vídeos", "*.mp4 *.avi *.mov"), ("Todos os arquivos", "*.*")],
        )
        if not selected:
            return
        path = Path(selected)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            messagebox.showerror("Vídeo inválido", "O OpenCV não conseguiu abrir esse arquivo.")
            return
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        ok, frame = capture.read()
        capture.release()
        self.video_path = path
        self.output_dir = OUTPUT_DIR
        self._video_info = {"width": width, "height": height, "fps": fps, "frames": frame_count,
                            "duration": duration}
        if ok:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            pil_image.thumbnail((800, 450))
            self.preview_image = ctk.CTkImage(
                light_image=pil_image, dark_image=pil_image,
                size=pil_image.size,
            )
        if self.active_page != "Video Analysis":
            self.show_page("Video Analysis")
        self._update_file_labels()
        if ok:
            self.preview_label.configure(image=self.preview_image, text="")
        self.footer_status.configure(text=f"Versão {APP_VERSION}   •   Vídeo selecionado")

    def _update_file_labels(self) -> None:
        if hasattr(self, "file_name_label"):
            self.file_name_label.configure(text=self.video_path.name if self.video_path else "Nenhum vídeo selecionado")
            info = self._video_info
            if info:
                seconds = int(info["duration"])
                duration_text = f"{seconds // 60:02d}:{seconds % 60:02d}"
                self.file_info_label.configure(
                    text=(f"{info['width']} × {info['height']}  ·  {info['fps']:.2f} FPS  ·  "
                          f"{duration_text}  ·  {info['frames']:,} frames".replace(",", "."))
                )
            self.output_label.configure(text=f"Saída: {self.output_dir}")

    def _select_output(self) -> None:
        selected = filedialog.askdirectory(title="Selecionar pasta de saída", mustexist=True)
        if selected:
            self.output_dir = Path(selected)
            self.output_label.configure(text=f"Saída: {self.output_dir}")

    def _start_processing(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        if self.video_path is None:
            messagebox.showinfo("Selecione um vídeo", "Escolha um vídeo antes de iniciar o processamento.")
            return
        if not MODEL_PATH.is_file():
            messagebox.showerror(
                "Modelo não encontrado", f"O arquivo do Pose Landmarker Full não está disponível:\n{MODEL_PATH}"
            )
            return
        try:
            self.settings.validate()
        except ValueError as error:
            messagebox.showerror("Configuração inválida", str(error))
            return

        self.cancel_event = threading.Event()
        self.progress_bar.set(0)
        self.percent_label.configure(text="0%")
        self.processing_status.configure(text="●  Iniciando modelo e vídeo…", text_color=COLORS["accent"])
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        for button in self.nav_buttons.values():
            button.configure(state="disabled")
        self.open_video_button.configure(state="disabled")
        self.results_text.configure(
            text="Vídeo processado: em execução\nCSV de landmarks: em execução\nTaxa de detecção: —\nTempo total: —"
        )
        processor = VideoProcessor(self.settings, output_root=self.output_dir)
        video_path = self.video_path
        self.worker = threading.Thread(
            target=self._worker_process, args=(processor, video_path),
            name="video-processing", daemon=True,
        )
        self.worker.start()

    def _worker_process(self, processor: VideoProcessor, video_path: Path) -> None:
        try:
            summary = processor.process(video_path, self.cancel_event, self.events.put)
            self.events.put({"type": "complete", "summary": summary})
        except InterruptedError:
            self.events.put({"type": "cancelled"})
        except Exception as error:
            logger.exception("Erro durante processamento da interface")
            self.events.put({"type": "error", "message": str(error)})

    def _cancel_processing(self) -> None:
        self.cancel_event.set()
        self.cancel_button.configure(state="disabled")
        self.processing_status.configure(text="●  Cancelando com segurança…", text_color="#f3bc70")

    def _poll_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                kind = event["type"]
                if kind == "progress":
                    self._show_progress(event)
                elif kind == "complete":
                    self._show_complete(event["summary"])
                elif kind == "cancelled":
                    self._finish_worker("●  Processamento cancelado", COLORS["muted"])
                elif kind == "error":
                    self._finish_worker("●  Falha no processamento", COLORS["red"])
                    messagebox.showerror("Falha no processamento", event["message"])
        except queue.Empty:
            pass
        self.after(120, self._poll_events)

    def _show_progress(self, event: dict) -> None:
        processed, total = event["processed"], event["total"]
        ratio = min(processed / total, 1.0) if total else 0.0
        self.progress_bar.set(ratio)
        self.percent_label.configure(text=f"{ratio * 100:.1f}%")
        elapsed = int(event["elapsed"])
        self.progress_text.configure(
            text=(f"Frames processados     {processed:,} / {total:,}\n"
                  f"Frames com pose          {event['detected']:,}\n"
                  f"FPS efetivo                   {event['fps']:.2f}\n"
                  f"Tempo decorrido           {elapsed // 60:02d}:{elapsed % 60:02d}").replace(",", ".")
        )
        self.processing_status.configure(
            text=f"●  Processando frame {processed:,} de {total:,}".replace(",", "."),
            text_color=COLORS["accent"],
        )
        frame = event.get("preview_frame")
        if frame is not None and hasattr(self, "preview_label"):
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            pil_image.thumbnail((800, 450))
            self.preview_image = ctk.CTkImage(
                light_image=pil_image,
                dark_image=pil_image,
                size=pil_image.size,
            )
            self.preview_label.configure(image=self.preview_image, text="")

    def _show_complete(self, summary: ProcessingSummary) -> None:
        self.session_runs.append(summary)
        self.last_video_name = self.video_path.name if self.video_path else "Vídeo analisado"
        self.progress_bar.set(1)
        self.percent_label.configure(text="100%")
        elapsed = int(summary.elapsed_seconds)
        self.progress_text.configure(
            text=(f"Frames processados     {summary.processed_frames:,} / {summary.total_frames:,}\n"
                  f"Frames com pose          {summary.detected_frames:,}\n"
                  f"FPS efetivo                   {summary.effective_fps:.2f}\n"
                  f"Tempo decorrido           {elapsed // 60:02d}:{elapsed % 60:02d}").replace(",", ".")
        )
        self.results_text.configure(
            text=(f"Vídeo processado: {summary.video_path.name}\n"
                  f"CSV de landmarks: {summary.csv_path.name}\n"
                  f"CSV ergonômico: {summary.metrics_csv_path.name}\n"
                  f"Taxa de detecção: {summary.detection_rate:.1f}%\n"
                  f"Tempo total: {summary.elapsed_seconds:.1f} s")
        )
        self.last_video_path = summary.video_path
        self.latest_metrics_path = summary.metrics_csv_path
        self.latest_metric_summary = summary.metric_summary
        self.latest_source_video = self.video_path.stem if self.video_path else ""
        self.latest_processing_info = {
            "processed_frames": summary.processed_frames,
            "total_frames": summary.total_frames,
            "detected_frames": summary.detected_frames,
            "effective_fps": f"{summary.effective_fps:.2f}",
            "elapsed_seconds": f"{summary.elapsed_seconds:.2f}",
            "detection_rate": f"{summary.detection_rate:.1f}%",
            "confidence_threshold": f"{self.settings.visibility_threshold:.2f}",
            "application_version": APP_VERSION,
            "video_fps": f"{self._video_info.get('fps', 0):.2f}",
            "video_duration": f"{self._video_info.get('duration', 0):.2f}",
        }
        self.dirty_pages.update({"Dashboard", "Analytics"})
        self._finish_worker("●  Processamento concluído", COLORS["green"])
        self.open_video_button.configure(state="normal")

    def _finish_worker(self, status: str, color: str) -> None:
        self.processing_status.configure(text=status, text_color=color)
        self.start_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        for button in self.nav_buttons.values():
            button.configure(state="normal")
        self.footer_status.configure(text=f"Versão {APP_VERSION}   •   {status.replace('●  ', '')}")

    def _on_close(self) -> None:
        """Cancel active work before releasing the application window."""
        if self.worker and self.worker.is_alive():
            self.cancel_event.set()
            self.after(100, self._wait_for_worker_close)
            return
        self.destroy()

    def _wait_for_worker_close(self) -> None:
        if self.worker and self.worker.is_alive():
            self.after(100, self._wait_for_worker_close)
        else:
            self.destroy()

    def _open_output(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            import os

            os.startfile(self.output_dir)  # type: ignore[attr-defined]
        except OSError as error:
            messagebox.showerror("Não foi possível abrir", str(error))

    def _open_processed_video(self) -> None:
        path = getattr(self, "last_video_path", None)
        if path and path.is_file():
            import os

            os.startfile(path)  # type: ignore[attr-defined]

    @staticmethod
    def _card(parent) -> ctk.CTkFrame:
        card = ctk.CTkFrame(
            parent, fg_color=COLORS["surface"], corner_radius=12,
            border_width=1, border_color=COLORS["border"],
        )
        content = ctk.CTkFrame(card, fg_color="transparent", corner_radius=0)
        content.pack(fill="both", expand=True, padx=17, pady=14)
        card.content_frame = content
        return card

    @staticmethod
    def _card_content(card: ctk.CTkFrame) -> ctk.CTkFrame:
        """Return the inset content area shared by all rounded panels."""
        return card.content_frame

    @staticmethod
    def _button(parent, text: str, command, primary: bool = False, width: int = 150):
        return ctk.CTkButton(
            parent, text=text, command=command, width=width, height=38, corner_radius=8,
            fg_color=COLORS["accent"] if primary else COLORS["surface_alt"],
            hover_color=COLORS["accent_hover"] if primary else "#2b3034",
            text_color="#101517" if primary else COLORS["text"],
            border_width=0 if primary else 1,
            border_color=COLORS["border"],
            font=ctk.CTkFont(size=12, weight="bold" if primary else "normal"),
        )

    @staticmethod
    def _section_heading(parent, title: str, subtitle: str) -> None:
        ctk.CTkLabel(
            parent, text=title, text_color=COLORS["text"],
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            parent, text=subtitle, text_color=COLORS["muted"],
            font=ctk.CTkFont(size=11),
        ).pack(anchor="w", pady=(3, 0))
