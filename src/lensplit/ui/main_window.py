"""Local-only desktop workflow for shot detection, grouping, and export."""

from __future__ import annotations

import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import QDate, QEvent, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import (
    QAction,
    QBrush,
    QCloseEvent,
    QColor,
    QFont,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QGraphicsVideoItem
from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QProgressBar,
)

from lensplit.analysis.asr import (
    AsrCancelledError,
    AsrConfig,
    AsrWorker,
    LocalTranscriber,
    TranscriptResult,
)
from lensplit.analysis.fast import FastDetectionConfig
from lensplit.analysis.person import (
    PersonCancelledError,
    PersonDetectConfig,
    PersonDetectWorker,
    YoloPersonDetector,
)
from lensplit.analysis.transnet import (
    LocalModelStatus,
    TransNetConfig,
    TransNetDetector,
)
from lensplit.analysis.worker import AnalysisWorker
from lensplit.core.casting import AvailabilityIssue, CastingData
from lensplit.core.model import ShotProject
from lensplit.core.numbering import (
    MergeMode,
    NumberingSequence,
    NumberingStyle,
    format_groups,
    transition_statistics,
)
from lensplit.core.session import DEFAULT_NUMBERING, PROJECT_SUFFIX, ProjectSession
from lensplit.export.exporter import (
    ExportCancelledError,
    ExportOptions,
    ExportReport,
    FFmpegExporter,
)
from lensplit.export.worker import ExportWorker
from lensplit.ui.actor_schedule import ActorSchedulePage
from lensplit.ui.shooting_table import ShootingTablePage, style_dark_combo
from lensplit.ui.timeline import TimelineItem, TimelineStrip
from lensplit.video.decoder import (
    DecodeCancelledError,
    VideoInfo,
    extract_thumbnail,
    probe_video,
)


class MarkedSlider(QSlider):
    """水平进度条，在轨道上叠加检测切点与手动标记。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(Qt.Horizontal, parent)
        self._cuts: list[float] = []
        self._marks: list[float] = []
        self.setMinimumHeight(26)

    def set_positions(self, cuts: list[float], marks: list[float]) -> None:
        self._cuts = sorted(cuts)
        self._marks = sorted(marks)
        self.update()

    def _x_for(self, ratio: float) -> float:
        return 4.0 + max(0.0, min(1.0, ratio)) * (self.width() - 8.0)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        for ratio in self._cuts:
            painter.setPen(QPen(QColor(90, 160, 230, 190), 1))
            x = self._x_for(ratio)
            painter.drawLine(int(x), 5, int(x), self.height() - 5)
        for ratio in self._marks:
            painter.setPen(QPen(QColor(255, 186, 66), 2))
            x = self._x_for(ratio)
            painter.drawLine(int(x), 3, int(x), self.height() - 3)
        painter.end()


def _parse_qdate(value: str) -> QDate:
    parsed = QDate.fromString(value or "", "yyyy-MM-dd")
    return parsed if parsed.isValid() else QDate(2000, 1, 1)


def _time_text(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


class ExportDialog(QDialog):
    """Collect export settings without embedding network-capable services."""

    def __init__(self, parent: QWidget, default_directory: str) -> None:
        super().__init__(parent)
        self.setWindowTitle("导出镜头")
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        self.output_edit = QLineEdit(default_directory)
        browse = QPushButton("选择目录…")
        browse.clicked.connect(self._browse)
        output_layout.addWidget(self.output_edit, 1)
        output_layout.addWidget(browse)
        form.addRow("输出目录", output_row)

        self.scope_combo = QComboBox()
        self.scope_combo.addItem("合并后的逻辑镜头", "logical")
        self.scope_combo.addItem("全部原始镜头", "source")
        form.addRow("导出范围", self.scope_combo)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("精确切点（重编码）", "exact")
        self.mode_combo.addItem("快速复制（关键帧对齐）", "copy")
        form.addRow("导出方式", self.mode_combo)

        self.encoder_combo = QComboBox()
        self.encoder_combo.addItem("自动：NVENC → QSV → AMF → CPU", "auto")
        self.encoder_combo.addItem("仅 CPU（libx264）", "cpu")
        form.addRow("视频编码器", self.encoder_combo)

        self.boundary_hint = QLabel("精确模式按 PTS 重编码，并自动测试本机可用的硬件编码器。")
        self.boundary_hint.setWordWrap(True)
        self.boundary_hint.setObjectName("mutedLabel")
        form.addRow("", self.boundary_hint)
        self.mode_combo.currentIndexChanged.connect(self._update_mode)
        self._update_mode()

        note = QLabel("同时导出 UTF-8 CSV 和 JSON 清单；视频、模型及元数据均只在本机处理。")
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("开始导出")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        for combo in self.findChildren(QComboBox):
            style_dark_combo(combo)

    def _browse(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择导出目录", self.output_edit.text())
        if directory:
            self.output_edit.setText(directory)

    def _update_mode(self) -> None:
        copying = self.mode_combo.currentData() == "copy"
        self.encoder_combo.setEnabled(not copying)
        self.boundary_hint.setText(
            "流复制不重编码，切点会偏移到附近关键帧；需要准确边界时请使用精确模式。"
            if copying
            else "精确模式按 PTS 重编码，并自动测试本机可用的硬件编码器。"
        )

    def accept(self) -> None:
        if not self.output_edit.text().strip():
            QMessageBox.warning(self, "缺少输出目录", "请选择镜头导出目录。")
            return
        super().accept()

    def export_options(
        self,
        style: NumberingStyle,
        merge_mode: MergeMode,
        analysis_metadata: dict,
        transcripts: dict | None = None,
    ) -> ExportOptions:
        return ExportOptions(
            output_dir=self.output_edit.text().strip(),
            scope=self.scope_combo.currentData(),
            mode=self.mode_combo.currentData(),
            encoder=self.encoder_combo.currentData(),
            numbering_style=style,
            merge_mode=merge_mode,
            analysis_metadata=analysis_metadata,
            transcripts=transcripts or {},
        )


class ScheduleConflictDialog(QDialog):
    """Ask how to resolve shoot days that fall outside an actor's window."""

    def __init__(
        self,
        parent: QWidget,
        issues: list[AvailabilityIssue],
        title: str = "档期与拍摄安排不一致",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(620)
        self.choice = "ignore"

        layout = QVBoxLayout(self)
        hint = QLabel(
            "页面 3「演员与档期」是主控档期。下面这些镜头日期不在演员档期内，"
            "请选择处理方式："
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        listing = QPlainTextEdit()
        listing.setReadOnly(True)
        listing.setPlainText("\n".join(f"· {issue.message}" for issue in issues))
        listing.setFixedHeight(min(220, 30 + 22 * len(issues)))
        layout.addWidget(listing)

        buttons = QHBoxLayout()
        extend = QPushButton("延长档期")
        extend.setToolTip("把演员档期的端点扩到能覆盖这些拍摄日")
        extend.clicked.connect(lambda: self._choose("extend"))
        buttons.addWidget(extend)
        clamp = QPushButton("拍摄日改为档期内")
        clamp.setToolTip("把这些镜头的拍摄日改到演员档期内最近的一天")
        clamp.clicked.connect(lambda: self._choose("clamp"))
        buttons.addWidget(clamp)
        ignore = QPushButton("暂不处理")
        ignore.setDefault(True)
        ignore.clicked.connect(lambda: self._choose("ignore"))
        buttons.addWidget(ignore)
        layout.addLayout(buttons)

    def _choose(self, choice: str) -> None:
        self.choice = choice
        self.accept()


class MainWindow(QMainWindow):
    """Single-video local editing window with thread-safe background events."""

    def __init__(self) -> None:
        super().__init__()
        self.video_path: str | None = None
        self.video_info: VideoInfo | None = None
        self.project: ShotProject | None = None
        self.casting = CastingData()
        self._history: list[dict] = []
        self._analysis_metadata: dict = {}
        self._model_status = LocalModelStatus(configured=False, available=False)
        self._analysis_worker: AnalysisWorker | None = None
        self._person_worker: PersonDetectWorker | None = None
        self._person_model_path = self._default_person_model_path()
        self._person_note = ""
        self._actor_panel_loading = False
        self._slot_state: tuple | None = None
        self._slot_syncing = False
        self._session_path: str | None = None
        self._dirty = False
        self._export_worker: ExportWorker | None = None
        self._busy_kind: str | None = None
        self._task_serial = 0
        self._task_started = 0.0
        self._events: queue.Queue[tuple] = queue.Queue()
        self._preview_end: float | None = None
        self._refreshing = False
        self._last_output_directory: str | None = None
        self._thumbnail_generation = 0
        self._thumbnail_cache: dict[str, QPixmap] = {}
        self._thumbnail_pending: set[tuple[int, str]] = set()
        self._thumbnail_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="lensplit-thumbnail"
        )

        self.setWindowTitle("Lensplit · 本地镜头切分与合并")
        self.resize(1480, 920)
        self.setMinimumSize(1100, 720)
        self._build_ui()
        self._build_menu()
        for combo in self.findChildren(QComboBox):
            style_dark_combo(combo)
        self._set_dirty(False)
        self._apply_dark_titlebar()
        self._update_model_controls()
        self._refresh_project()
        self._update_person_controls()
        self._auto_load_default_model()
        self._event_timer = QTimer(self)
        self._event_timer.setInterval(40)
        self._event_timer.timeout.connect(self._drain_events)
        self._event_timer.start()
        self._preview_label: str | None = None
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(25)
        self._preview_timer.timeout.connect(self._check_preview_end)

    def _apply_dark_titlebar(self) -> None:
        """Match the native title bar to the dark palette (Windows 10/11)."""
        if os.name != "nt":
            return
        try:
            import ctypes

            enabled = ctypes.c_int(1)
            hwnd = ctypes.windll.user32.GetParent(int(self.winId()))
            for attribute in (20, 19):
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, attribute, ctypes.byref(enabled), ctypes.sizeof(enabled)
                )
        except Exception:
            pass

    def _build_ui(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QDialog { background: #151922; color: #e8edf5; }
            QWidget { color: #e8edf5; font-size: 13px; }
            QGroupBox { border: 1px solid #303847; border-radius: 7px; margin-top: 12px;
                        padding-top: 10px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #aebcd0; }
            QPushButton { background: #2a3343; border: 1px solid #3b475a; border-radius: 5px;
                          padding: 7px 12px; min-height: 20px; }
            QPushButton:hover { background: #354258; border-color: #62779a; }
            QPushButton:pressed { background: #202938; }
            QPushButton:disabled { color: #687384; background: #202630; border-color: #2b3340; }
            QPushButton#primaryButton { background: #3979dc; border-color: #5591ed; font-weight: 600; }
            QPushButton#primaryButton:hover { background: #4a88e7; }
            QPushButton#primaryButton:disabled { background: #26313f; border-color: #33404f; color: #6f7a89; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit { background: #202733; color: #edf2fa;
                border: 1px solid #394557; border-radius: 4px; padding: 5px; min-height: 20px; }
            QPlainTextEdit, QTextEdit { background: #11161e; color: #edf2fa;
                border: 1px solid #394557; border-radius: 4px; padding: 5px;
                selection-background-color: #3979dc; }
            QPlainTextEdit:focus, QTextEdit:focus { border-color: #5591ed; }
            QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
            QDateEdit:disabled { color: #6f7a89; }
            QLineEdit { placeholder-text-color: #74839a; }
            QPlainTextEdit { placeholder-text-color: #74839a; }
            QSlider::groove:horizontal { height: 5px; background: #333e4e; border-radius: 2px; }
            QSlider::handle:horizontal { width: 14px; margin: -5px 0; background: #67a2f4; border-radius: 7px; }
            QProgressBar { background: #202733; border: 1px solid #354154; border-radius: 4px; text-align: center; }
            QProgressBar::chunk { background: #3979dc; }
            QLabel#mutedLabel { color: #95a3b7; font-weight: 400; }
            QLabel#statsLabel { color: #a9c8f2; font-size: 14px; font-weight: 600; }
            QVideoWidget { background: #080b10; border: 1px solid #303847; }
            QStatusBar { background: #11161e; color: #b9c5d5; }
            QToolTip { background: #263142; color: white; border: 1px solid #506079; }
            QScrollBar:vertical { background: #151922; width: 10px; margin: 0; border: none; }
            QScrollBar::handle:vertical { background: #3b475a; border-radius: 5px; min-height: 24px; }
            QScrollBar::handle:vertical:hover { background: #50627c; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
            QScrollBar:horizontal { background: #151922; height: 10px; margin: 0; border: none; }
            QScrollBar::handle:horizontal { background: #3b475a; border-radius: 5px; min-width: 24px; }
            QScrollBar::handle:horizontal:hover { background: #50627c; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }
            QMenuBar { background: #151922; color: #dbe5f2; border-bottom: 1px solid #232c3a; }
            QMenuBar::item { background: transparent; padding: 4px 10px; }
            QMenuBar::item:selected { background: #2a3443; border-radius: 4px; }
            QMenu { background: #1b2230; color: #e8edf5; border: 1px solid #3b475a; padding: 4px; }
            QMenu::item { padding: 6px 26px 6px 18px; border-radius: 4px; }
            QMenu::item:selected { background: #3979dc; }
            QMenu::separator { height: 1px; background: #39455a; margin: 4px 8px; }
            QComboBox QAbstractItemView { background: #1b2230; color: #edf2fa;
                border: 1px solid #394557; selection-background-color: #3979dc;
                selection-color: #ffffff; outline: 0; }
            QComboBox QAbstractItemView::item { min-height: 22px; padding: 3px 6px;
                background: #1b2230; color: #edf2fa; }
            QComboBox QAbstractItemView::item:selected { background: #3979dc; color: #ffffff; }
            QScrollArea { background: transparent; border: none; }
            QScrollArea > QWidget#qt_scrollarea_viewport { background: transparent; }
            QScrollArea > QWidget > QWidget { background: transparent; }
            QSplitter::handle { background: #1b2130; }
            QSplitter::handle:hover { background: #2a3446; }
            QTabWidget::pane { border: 1px solid #2b3545; border-radius: 6px; top: -1px; }
            QTabBar::tab { background: #1b2230; color: #aab7c9; padding: 7px 24px; margin-right: 4px;
                border: 1px solid #2b3545; border-bottom: none;
                border-top-left-radius: 6px; border-top-right-radius: 6px; }
            QTabBar::tab:hover { color: #ffffff; background: #232c3a; }
            QTabBar::tab:selected { background: #223a5c; color: #ffffff; border-color: #3f5f8c; }
            QTableWidget { background: #11161e; alternate-background-color: #141a24;
                gridline-color: #232c3a; border: 1px solid #2b3545; border-radius: 6px; }
            QTableWidget::item { padding: 4px; }
            QTableWidget::item:selected { background: #223a5c; color: #ffffff; }
            QHeaderView::section { background: #1b2230; color: #aebcd0; padding: 6px;
                border: none; border-right: 1px solid #232c3a; border-bottom: 1px solid #2b3545;
                font-weight: 600; }
            QCalendarWidget QWidget { background: #1b2230; color: #e8edf5; }
            QCalendarWidget QAbstractItemView:enabled { background: #11161e; color: #e8edf5;
                selection-background-color: #3979dc; selection-color: #ffffff; }
            QCalendarWidget QToolButton { background: #2a3343; color: #e8edf5; }
            QPlainTextEdit#cellEdit { background: transparent; border: none; padding: 2px; }
            QPlainTextEdit#cellEdit:focus { background: #161d28; border: 1px solid #3979dc; }
            QPushButton#roleButton { text-align: left; padding: 5px 9px; }
            QPushButton#shotThumbButton { padding: 2px; }
            QLabel#conflictLabel { background: #3a2130; border: 1px solid #6d3a4c;
                border-radius: 5px; padding: 7px 10px; color: #ffd4d4; }
            QLabel#conflictTag { background: #5a2130; border: 1px solid #8a3a4e;
                border-radius: 4px; padding: 1px 7px; color: #ffd9d9; font-weight: 600; }
            """
        )
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 8, 0, 0)
        outer.setSpacing(0)
        self.setCentralWidget(central)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        outer.addWidget(self.tabs, 1)

        editor_page = QWidget()
        root = QVBoxLayout(editor_page)
        root.setContentsMargins(12, 10, 12, 8)
        root.setSpacing(9)
        self.tabs.addTab(editor_page, "剪辑")

        source_group = QGroupBox("1 · 导入与本地检测")
        source = QGridLayout(source_group)
        source.setHorizontalSpacing(8)
        source.setVerticalSpacing(8)
        self.open_button = QPushButton("导入视频…")
        self.open_button.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        self.open_button.clicked.connect(self.import_video)
        self.file_label = QLabel("尚未导入视频")
        self.file_label.setObjectName("mutedLabel")
        self.file_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.media_label = QLabel("视频仅在本机解码，不上传文件或帧数据")
        self.media_label.setObjectName("mutedLabel")
        self.media_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        source.addWidget(self.open_button, 0, 0)
        source.addWidget(self.file_label, 0, 1, 1, 4)
        source.addWidget(self.media_label, 0, 5, 1, 3)

        source.addWidget(QLabel("检测引擎"), 1, 0)
        self.engine_combo = QComboBox()
        self.engine_combo.addItem("极速帧差 / 直方图", "fast")
        self.engine_combo.addItem("本地 TransNetV2 ONNX", "transnet")
        self.engine_combo.currentIndexChanged.connect(self._engine_changed)
        self.engine_combo.setMinimumWidth(170)
        source.addWidget(self.engine_combo, 1, 1, 1, 2)
        source.addWidget(QLabel("阈值"), 1, 3)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(0.60)
        source.addWidget(self.threshold_spin, 1, 4)
        source.addWidget(QLabel("最短镜头（秒）"), 1, 5)
        self.min_shot_spin = QDoubleSpinBox()
        self.min_shot_spin.setRange(0.0, 60.0)
        self.min_shot_spin.setDecimals(2)
        self.min_shot_spin.setSingleStep(0.1)
        self.min_shot_spin.setValue(0.50)
        source.addWidget(self.min_shot_spin, 1, 6)
        self.analyze_button = QPushButton("开始分析")
        self.analyze_button.setObjectName("primaryButton")
        self.analyze_button.clicked.connect(self.start_analysis)
        self.analyze_button.setMinimumWidth(104)
        source.addWidget(self.analyze_button, 1, 7)

        self.model_button = QPushButton("选择本地 ONNX…")
        self.model_button.clicked.connect(self.select_model)
        source.addWidget(self.model_button, 2, 0)
        self.model_path_label = QLabel("未配置模型")
        self.model_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.model_path_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        source.addWidget(self.model_path_label, 2, 1, 1, 3)
        self.clear_model_button = QPushButton("清除模型")
        self.clear_model_button.clicked.connect(self.clear_model)
        source.addWidget(self.clear_model_button, 2, 4)

        self.model_status_label = QLabel()
        self.model_status_label.setObjectName("mutedLabel")
        self.model_status_label.setWordWrap(False)
        self.model_status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        source.addWidget(self.model_status_label, 2, 5, 1, 3)
        source.setColumnStretch(1, 1)
        source.setColumnStretch(4, 1)
        root.addWidget(source_group)

        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.setChildrenCollapsible(False)
        root.addWidget(main_splitter, 1)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        main_splitter.addWidget(splitter)
        left = QWidget()
        left.setMinimumWidth(430)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        preview_group = QGroupBox("2 · 连续画面预览")
        preview_layout = QVBoxLayout(preview_group)
        self.video_scene = QGraphicsScene(self)
        self.video_scene.setBackgroundBrush(QColor("#080b10"))
        self.video_view = QGraphicsView(self.video_scene)
        self.video_view.setMinimumSize(400, 225)
        self.video_view.setFrameShape(QFrame.NoFrame)
        self.video_view.setAlignment(Qt.AlignCenter)
        self.video_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.video_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.video_view.viewport().installEventFilter(self)
        self.video_item = QGraphicsVideoItem()
        self.video_item.setAspectRatioMode(Qt.KeepAspectRatio)
        self.video_scene.addItem(self.video_item)
        self.video_item.nativeSizeChanged.connect(self._video_size_changed)
        self.shot_badge_backdrop = QGraphicsRectItem()
        self.shot_badge_backdrop.setBrush(QBrush(QColor(0, 0, 0, 165)))
        self.shot_badge_backdrop.setPen(Qt.NoPen)
        self.shot_badge_backdrop.setZValue(9)
        self.shot_badge_backdrop.setVisible(False)
        self.video_scene.addItem(self.shot_badge_backdrop)
        self.shot_badge = QGraphicsTextItem()
        badge_font = QFont()
        badge_font.setPointSize(18)
        badge_font.setBold(True)
        self.shot_badge.setFont(badge_font)
        self.shot_badge.setDefaultTextColor(QColor("#ffffff"))
        self.shot_badge.setZValue(10)
        self.shot_badge.setVisible(False)
        self.video_scene.addItem(self.shot_badge)
        preview_layout.addWidget(self.video_view, 1)
        self._group_labels: list[str] = []
        self._group_ranges: list[tuple[float, float]] = []
        self._manual_marks: list[float] = []
        self._transcripts: dict[str, str] = {}
        self._transcript_results: dict[str, TranscriptResult] = {}
        self._asr_worker: AsrWorker | None = None
        self._asr_transcriber: LocalTranscriber | None = None
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.player.setVideoOutput(self.video_item)
        self.audio_output.setVolume(0.8)
        self.player.positionChanged.connect(self._position_changed)
        self.player.durationChanged.connect(self._duration_changed)
        self.player.playbackStateChanged.connect(self._playback_state_changed)
        self.player.errorOccurred.connect(self._player_error)

        playback_row = QHBoxLayout()
        self.play_button = QPushButton("播放")
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.toggle_playback)
        playback_row.addWidget(self.play_button)
        self.position_slider = MarkedSlider()
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.player.setPosition)
        self.position_slider.sliderReleased.connect(self._slider_released)
        playback_row.addWidget(self.position_slider, 1)
        self.position_label = QLabel("00:00.000 / 00:00.000")
        playback_row.addWidget(self.position_label)

        badge_row = QHBoxLayout()
        badge_row.addWidget(QLabel("镜号位置"))
        self.badge_position_combo = QComboBox()
        for text, key in (
            ("左上角", "top-left"),
            ("顶部居中", "top-center"),
            ("右上角", "top-right"),
            ("中间偏左", "middle-left"),
            ("画面正中", "middle-center"),
            ("中间偏右", "middle-right"),
            ("左下角", "bottom-left"),
            ("底部居中", "bottom-center"),
            ("右下角", "bottom-right"),
        ):
            self.badge_position_combo.addItem(text, key)
        self.badge_position_combo.currentIndexChanged.connect(self._place_badge)
        badge_row.addWidget(self.badge_position_combo)
        badge_row.addWidget(QLabel("字号"))
        self.badge_size_spin = QSpinBox()
        self.badge_size_spin.setRange(8, 72)
        self.badge_size_spin.setValue(18)
        self.badge_size_spin.setSuffix(" pt")
        self.badge_size_spin.valueChanged.connect(self._badge_font_changed)
        badge_row.addWidget(self.badge_size_spin)
        self.badge_hint = QLabel("播放或单击镜头时，镜号实时叠加在画面上（仅预览，不写入导出视频）")
        self.badge_hint.setObjectName("mutedLabel")
        badge_row.addWidget(self.badge_hint, 1)
        preview_layout.addLayout(playback_row)
        preview_layout.addLayout(badge_row)
        self.stats_label = QLabel("原始转场 0 ｜ 有效转场 0 ｜ 合并镜内部转场 0")
        self.stats_label.setObjectName("statsLabel")
        preview_layout.addWidget(self.stats_label)
        left_layout.addWidget(preview_group, 1)

        transition_group = QGroupBox("3 · 转场切点微调")
        transition_layout = QGridLayout(transition_group)
        self.transition_combo = QComboBox()
        self.transition_combo.currentIndexChanged.connect(self._transition_changed)
        transition_layout.addWidget(self.transition_combo, 0, 0, 1, 3)
        transition_layout.addWidget(QLabel("新切点（秒）"), 1, 0)
        self.cut_spin = QDoubleSpinBox()
        self.cut_spin.setDecimals(6)
        self.cut_spin.setSingleStep(0.001)
        self.cut_spin.setRange(0.000001, 86_400.0)
        transition_layout.addWidget(self.cut_spin, 1, 1)
        self.apply_cut_button = QPushButton("应用切点")
        self.apply_cut_button.clicked.connect(self.apply_boundary)
        transition_layout.addWidget(self.apply_cut_button, 1, 2)
        transition_hint = QLabel("调整共同边界会同步修改前后两个原始镜头；内部转场不会因合并而删除。")
        transition_hint.setObjectName("mutedLabel")
        transition_hint.setWordWrap(True)
        transition_layout.addWidget(transition_hint, 2, 0, 1, 3)
        manual_row = QHBoxLayout()
        self.mark_button = QPushButton("在当前帧标记切点")
        self.mark_button.clicked.connect(self.add_manual_mark)
        manual_row.addWidget(self.mark_button)
        self.unmark_button = QPushButton("撤销标记")
        self.unmark_button.clicked.connect(self.remove_last_mark)
        manual_row.addWidget(self.unmark_button)
        self.apply_marks_button = QPushButton("应用手动切分")
        self.apply_marks_button.clicked.connect(self.apply_manual_marks)
        manual_row.addWidget(self.apply_marks_button)
        transition_layout.addLayout(manual_row, 3, 0, 1, 3)
        self.mark_label = QLabel("手动切点：0 个（黄线＝待应用标记，蓝线＝已有切点）")
        self.mark_label.setObjectName("mutedLabel")
        self.mark_label.setWordWrap(True)
        transition_layout.addWidget(self.mark_label, 4, 0, 1, 3)
        left_layout.addWidget(transition_group)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.viewport().setAutoFillBackground(False)
        left_scroll.viewport().setStyleSheet("background: transparent;")
        left_scroll.setWidget(left)
        splitter.addWidget(left_scroll)

        right = QWidget()
        right.setMinimumWidth(430)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        shots_group = QGroupBox("5 · 镜头编号")
        shots_layout = QVBoxLayout(shots_group)

        numbering = QGridLayout()
        numbering.addWidget(QLabel("起始值"), 0, 0)
        self.start_spin = QSpinBox()
        self.start_spin.setRange(1, 999_999)
        self.start_spin.setValue(1)
        numbering.addWidget(self.start_spin, 0, 1)
        numbering.addWidget(QLabel("补零宽度"), 0, 2)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(0, 12)
        self.width_spin.setValue(3)
        numbering.addWidget(self.width_spin, 0, 3)
        numbering.addWidget(QLabel("序列"), 1, 0)
        self.sequence_combo = QComboBox()
        self.sequence_combo.addItem("十进制：001", NumberingSequence.DECIMAL.value)
        self.sequence_combo.addItem("小写字母：00a", NumberingSequence.LOWER.value)
        self.sequence_combo.addItem("大写字母：00A", NumberingSequence.UPPER.value)
        numbering.addWidget(self.sequence_combo, 1, 1)
        numbering.addWidget(QLabel("合并编号模式"), 1, 2)
        self.merge_mode_combo = QComboBox()
        self.merge_mode_combo.addItem("组合：001+002，003", MergeMode.COMBINED.value)
        self.merge_mode_combo.addItem("递进：001，002", MergeMode.SEQUENTIAL.value)
        self.merge_mode_combo.addItem("首号：001，003", MergeMode.FIRST.value)
        numbering.addWidget(self.merge_mode_combo, 1, 3)
        numbering.addWidget(QLabel("前缀"), 2, 0)
        self.prefix_edit = QLineEdit()
        self.prefix_edit.setPlaceholderText("例如 S")
        numbering.addWidget(self.prefix_edit, 2, 1)
        numbering.addWidget(QLabel("后缀"), 2, 2)
        self.suffix_edit = QLineEdit()
        self.suffix_edit.setPlaceholderText("例如 #")
        numbering.addWidget(self.suffix_edit, 2, 3)
        for widget, maximum in (
            (self.start_spin, 90),
            (self.width_spin, 90),
            (self.sequence_combo, 140),
            (self.merge_mode_combo, 170),
            (self.prefix_edit, 90),
            (self.suffix_edit, 90),
        ):
            widget.setMaximumWidth(maximum)
        numbering.setHorizontalSpacing(8)
        numbering.setColumnStretch(1, 1)
        numbering.setColumnStretch(3, 1)
        shots_layout.addLayout(numbering)
        self.number_example_label = QLabel("示例：001、002、003")
        self.number_example_label.setObjectName("mutedLabel")
        shots_layout.addWidget(self.number_example_label)

        actor_group = QGroupBox("4 · 人物自动标记与演员分配")
        actor_layout = QGridLayout(actor_group)
        actor_layout.setHorizontalSpacing(8)
        actor_layout.setVerticalSpacing(7)

        self.person_mark_button = QPushButton("自动标记全部镜头")
        self.person_mark_button.clicked.connect(lambda: self.start_person_detection(False))
        actor_layout.addWidget(self.person_mark_button, 0, 0, 1, 2)
        self.person_mark_selected_button = QPushButton("仅标记所选镜头")
        self.person_mark_selected_button.clicked.connect(
            lambda: self.start_person_detection(True)
        )
        actor_layout.addWidget(self.person_mark_selected_button, 0, 2, 1, 2)
        actor_layout.addWidget(QLabel("置信度"), 0, 4)
        self.person_confidence_spin = QDoubleSpinBox()
        self.person_confidence_spin.setRange(0.10, 0.90)
        self.person_confidence_spin.setSingleStep(0.05)
        self.person_confidence_spin.setDecimals(2)
        self.person_confidence_spin.setValue(0.20)
        actor_layout.addWidget(self.person_confidence_spin, 0, 5)
        actor_layout.addWidget(QLabel("采样帧"), 0, 6)
        self.person_samples_spin = QSpinBox()
        self.person_samples_spin.setRange(1, 9)
        self.person_samples_spin.setValue(5)
        self.person_samples_spin.setToolTip(
            "每个逻辑镜头抽取的检测帧数，取人数最多的一帧（更容易抓全画面里的人）"
        )
        actor_layout.addWidget(self.person_samples_spin, 0, 7)

        actor_layout.addWidget(QLabel("当前镜头"), 1, 0)
        self.actor_shot_label = QLabel("—")
        self.actor_shot_label.setObjectName("statsLabel")
        actor_layout.addWidget(self.actor_shot_label, 1, 1)
        actor_layout.addWidget(QLabel("人数"), 1, 2)
        self.actor_count_spin = QSpinBox()
        self.actor_count_spin.setRange(0, 99)
        self.actor_count_spin.setSpecialValueText("未标注")
        self.actor_count_spin.valueChanged.connect(self._actor_count_changed)
        actor_layout.addWidget(self.actor_count_spin, 1, 3)
        actor_layout.addWidget(QLabel("拍摄日期"), 1, 4)
        self.actor_date_edit = QDateEdit()
        self.actor_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.actor_date_edit.setCalendarPopup(True)
        self.actor_date_edit.setSpecialValueText("未定")
        self.actor_date_edit.setMinimumDate(QDate(2000, 1, 1))
        self.actor_date_edit.setMaximumDate(QDate(2100, 12, 31))
        self.actor_date_edit.dateChanged.connect(self._actor_date_changed)
        actor_layout.addWidget(self.actor_date_edit, 1, 5, 1, 3)

        self.slot_header_label = QLabel("角色与演员 · 人数决定角色槽，可改名并从演员库分配")
        self.slot_header_label.setObjectName("mutedLabel")
        actor_layout.addWidget(self.slot_header_label, 2, 0, 1, 8)

        self.slot_area = QWidget()
        self.slot_layout = QVBoxLayout(self.slot_area)
        self.slot_layout.setContentsMargins(0, 0, 0, 0)
        self.slot_layout.setSpacing(4)
        self.slot_scroll = QScrollArea()
        self.slot_scroll.setWidgetResizable(True)
        self.slot_scroll.setFrameShape(QFrame.NoFrame)
        self.slot_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.slot_scroll.setWidget(self.slot_area)
        self.slot_scroll.setFixedHeight(96)
        actor_layout.addWidget(self.slot_scroll, 3, 0, 1, 8)

        self.actor_schedule_label = QLabel("未选择镜头")
        self.actor_schedule_label.setObjectName("mutedLabel")
        self.actor_schedule_label.setWordWrap(True)
        actor_layout.addWidget(self.actor_schedule_label, 4, 0, 1, 8)
        self.person_status_label = QLabel("人物标记：等待导入视频")
        self.person_status_label.setObjectName("mutedLabel")
        self.person_status_label.setWordWrap(True)
        actor_layout.addWidget(self.person_status_label, 5, 0, 1, 8)
        actor_layout.setColumnStretch(1, 0)
        actor_layout.setColumnStretch(3, 0)
        actor_layout.setColumnStretch(5, 1)
        actor_layout.setColumnStretch(7, 0)
        for widget, maximum in (
            (self.person_confidence_spin, 80),
            (self.person_samples_spin, 62),
            (self.actor_count_spin, 86),
            (self.actor_date_edit, 128),
        ):
            widget.setMaximumWidth(maximum)
        right_layout.addWidget(actor_group)
        right_layout.addWidget(shots_group)

        asr_group = QGroupBox("6 · 台词识别（本地模型）")
        asr_layout = QVBoxLayout(asr_group)
        asr_row = QHBoxLayout()
        self.asr_button = QPushButton("识别全部台词")
        self.asr_button.clicked.connect(self.start_transcription)
        asr_row.addWidget(self.asr_button)
        self.asr_cancel_button = QPushButton("取消识别")
        self.asr_cancel_button.clicked.connect(self.cancel_transcription)
        self.asr_cancel_button.setEnabled(False)
        asr_row.addWidget(self.asr_cancel_button)
        self.asr_status_label = QLabel()
        self.asr_status_label.setObjectName("mutedLabel")
        asr_row.addWidget(self.asr_status_label, 1)
        asr_layout.addLayout(asr_row)
        self.transcript_edit = QPlainTextEdit()
        self.transcript_edit.setPlaceholderText(
            "选中一个镜头后可在此查看和修改台词；空片段会标注「无台词」。"
        )
        self.transcript_edit.setFixedHeight(78)
        asr_layout.addWidget(self.transcript_edit)
        transcript_row = QHBoxLayout()
        self.transcript_hint = QLabel("识别结果仅供参考，可手动改正后保存。")
        self.transcript_hint.setObjectName("mutedLabel")
        transcript_row.addWidget(self.transcript_hint, 1)
        self.save_transcript_button = QPushButton("保存台词修改")
        self.save_transcript_button.clicked.connect(self.save_transcript)
        transcript_row.addWidget(self.save_transcript_button)
        asr_layout.addLayout(transcript_row)
        right_layout.addWidget(asr_group)

        export_group = QGroupBox("7 · 本地导出")
        export_layout = QHBoxLayout(export_group)
        export_hint = QLabel("可导出原始镜头或合并后的完整连续区间，并附 CSV / JSON 清单。")
        export_hint.setWordWrap(True)
        export_layout.addWidget(export_hint, 1)
        self.export_button = QPushButton("导出镜头…")
        self.export_button.setObjectName("primaryButton")
        self.export_button.clicked.connect(self.start_export)
        export_layout.addWidget(self.export_button)
        right_layout.addWidget(export_group)
        right_layout.addStretch(1)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.NoFrame)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_scroll.viewport().setAutoFillBackground(False)
        right_scroll.viewport().setStyleSheet("background: transparent;")
        right_scroll.setWidget(right)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([850, 580])

        timeline_group = QGroupBox("镜头序列 · 时间线（Ctrl / Shift 多选，单击预览）")
        timeline_group.setMinimumHeight(264)
        timeline_layout = QVBoxLayout(timeline_group)
        timeline_layout.setSpacing(6)
        action_row = QHBoxLayout()
        self.merge_button = QPushButton("合并所选")
        self.merge_button.setToolTip("把相邻的所选镜头合成为一个连续逻辑镜头")
        self.merge_button.clicked.connect(self.merge_selected)
        action_row.addWidget(self.merge_button)
        self.split_button = QPushButton("拆解为原始镜头")
        self.split_button.setToolTip("把合并过的镜头拆回独立原始镜头")
        self.split_button.clicked.connect(self.split_selected)
        action_row.addWidget(self.split_button)
        self.play_shot_button = QPushButton("播放所选")
        self.play_shot_button.clicked.connect(self.play_selected_group)
        action_row.addWidget(self.play_shot_button)
        self.undo_button = QPushButton("撤销")
        self.undo_button.clicked.connect(self.undo)
        action_row.addWidget(self.undo_button)
        self.selection_label = QLabel("未选择镜头 · 单击卡片预览，Ctrl / Shift 可多选")
        self.selection_label.setObjectName("mutedLabel")
        action_row.addWidget(self.selection_label, 1)
        timeline_layout.addLayout(action_row)
        self.timeline = TimelineStrip()
        self.timeline.activated.connect(self._preview_group)
        self.timeline.selection_changed.connect(self._timeline_selection_changed)
        timeline_layout.addWidget(self.timeline)
        main_splitter.addWidget(timeline_group)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 0)
        main_splitter.setSizes([600, 264])

        self.shooting_page = ShootingTablePage()
        self.shooting_page.preview_requested.connect(self._preview_from_shooting)
        self.shooting_page.transcript_changed.connect(self._shooting_transcript_changed)
        self.shooting_page.data_changed.connect(self._shooting_data_changed)
        self.shooting_page.schedule_changed.connect(self._schedule_edited)
        self.tabs.addTab(self.shooting_page, "拍摄表")

        self.actor_page = ActorSchedulePage()
        self.actor_page.data_changed.connect(self._actor_schedule_changed)
        self.actor_page.window_changed.connect(self._actor_window_changed)
        self.tabs.addTab(self.actor_page, "演员与档期")
        self.tabs.currentChanged.connect(self._tab_changed)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setFixedWidth(220)
        self.progress_bar.hide()
        self.cancel_task_button = QPushButton("取消任务")
        self.cancel_task_button.clicked.connect(self.cancel_current_task)
        self.cancel_task_button.hide()
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.statusBar().addPermanentWidget(self.cancel_task_button)
        self.statusBar().showMessage("就绪 · 所有处理均在本地完成")

        self._numbering_controls = [
            self.start_spin,
            self.width_spin,
            self.sequence_combo,
            self.merge_mode_combo,
            self.prefix_edit,
            self.suffix_edit,
        ]
        for control in self._numbering_controls[:4]:
            if isinstance(control, (QSpinBox, QDoubleSpinBox)):
                control.valueChanged.connect(self._numbering_changed)
            else:
                control.currentIndexChanged.connect(self._numbering_changed)
        self.prefix_edit.textChanged.connect(self._numbering_changed)
        self.suffix_edit.textChanged.connect(self._numbering_changed)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("文件(&F)")
        open_project_action = QAction("打开工程…", self)
        open_project_action.setShortcut(QKeySequence.Open)
        open_project_action.triggered.connect(self.open_project)
        file_menu.addAction(open_project_action)
        save_project_action = QAction("保存工程", self)
        save_project_action.setShortcut(QKeySequence.Save)
        save_project_action.triggered.connect(self.save_project)
        file_menu.addAction(save_project_action)
        save_as_action = QAction("工程另存为…", self)
        save_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_as_action.triggered.connect(lambda: self.save_project(save_as=True))
        file_menu.addAction(save_as_action)
        file_menu.addSeparator()
        open_action = QAction("导入视频…", self)
        open_action.setShortcut(QKeySequence("Ctrl+I"))
        open_action.triggered.connect(self.import_video)
        file_menu.addAction(open_action)
        export_action = QAction("导出镜头…", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self.start_export)
        file_menu.addAction(export_action)
        file_menu.addSeparator()
        quit_action = QAction("退出", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        edit_menu = self.menuBar().addMenu("编辑(&E)")
        undo_action = QAction("撤销", self)
        undo_action.setShortcut(QKeySequence.Undo)
        undo_action.triggered.connect(self.undo)
        edit_menu.addAction(undo_action)
        merge_action = QAction("合并相邻镜头", self)
        merge_action.setShortcut(QKeySequence("Ctrl+M"))
        merge_action.triggered.connect(self.merge_selected)
        edit_menu.addAction(merge_action)
        split_action = QAction("拆分为原始镜头", self)
        split_action.triggered.connect(self.split_selected)
        edit_menu.addAction(split_action)

    def import_video(self) -> None:
        if self._busy_kind:
            if self._busy_kind == "export":
                QMessageBox.information(
                    self,
                    "正在导出",
                    "导出任务尚未完成，请先在窗口底部点击「取消导出」，再导入新视频。",
                )
                return
            task = {"analysis": "分析", "asr": "台词识别"}.get(self._busy_kind, "当前任务")
            answer = QMessageBox.question(
                self,
                "任务进行中",
                f"{task}尚未完成，是否取消并导入新视频？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
            self._cancel_and_reset_busy()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入本地视频",
            str(Path(self.video_path).parent) if self.video_path else "",
            "视频文件 (*.mp4 *.mov *.mkv *.avi *.m4v *.ts *.webm);;所有文件 (*)",
        )
        if not path:
            return
        try:
            info = probe_video(path)
        except Exception as error:
            QMessageBox.critical(self, "无法读取视频", str(error))
            return
        self.load_video(path, info)

    def load_video(self, path: str, info: VideoInfo | None = None) -> None:
        """Load a local file; also provides a deterministic smoke-test entry point."""
        self.player.stop()
        self.player.setSource(QUrl())
        self.video_path = str(Path(path).expanduser().resolve())
        self.video_info = info or probe_video(self.video_path)
        self.project = None
        self.casting = CastingData()
        self._history.clear()
        self._analysis_metadata = {}
        self._thumbnail_generation += 1
        self._thumbnail_cache.clear()
        self._thumbnail_pending.clear()
        self.shooting_page.clear_thumbnails()
        self.player.setSource(QUrl.fromLocalFile(self.video_path))
        self._update_video_labels()
        self._preview_end = None
        self._preview_label = None
        self._session_path = None
        self._set_dirty(False)
        self._refresh_project()
        self.statusBar().showMessage("视频已导入，请选择本地检测引擎")

    def _update_video_labels(self) -> None:
        if not self.video_path:
            self.file_label.setText("尚未导入视频")
            self.file_label.setToolTip("")
            self.media_label.setText("视频仅在本机解码，不上传文件或帧数据")
            return
        self.file_label.setText(self.video_path)
        self.file_label.setToolTip(self.video_path)
        if self.video_info is None:
            self.media_label.setText("视频信息未知（文件可能已被移动）")
            return
        duration = self.video_info.duration or 0.0
        fps = (
            f"{self.video_info.average_fps:.3f} fps"
            if self.video_info.average_fps
            else "帧率未知"
        )
        self.media_label.setText(
            f"{self.video_info.width}×{self.video_info.height} · {fps} · {_time_text(duration)}"
        )

    # -- project files ---------------------------------------------------
    def _numbering_state(self) -> dict:
        return {
            "start": self.start_spin.value(),
            "width": self.width_spin.value(),
            "prefix": self.prefix_edit.text(),
            "suffix": self.suffix_edit.text(),
            "sequence": self.sequence_combo.currentData(),
            "merge_mode": self.merge_mode_combo.currentData(),
        }

    def session_snapshot(self) -> ProjectSession:
        """Collect every piece of working state worth persisting."""

        return ProjectSession(
            video_path=self.video_path or "",
            project=self.project,
            casting=self.casting,
            transcripts=dict(self._transcripts),
            analysis=dict(self._analysis_metadata),
            numbering=self._numbering_state(),
        )

    def _default_session_path(self) -> str:
        if self.video_path:
            return str(Path(self.video_path).with_suffix(PROJECT_SUFFIX))
        return str(Path.home() / f"Lensplit 工程{PROJECT_SUFFIX}")

    def save_project(self, save_as: bool = False) -> bool:
        """Save the current session, asking for a location the first time."""

        if self.project is None:
            QMessageBox.information(
                self, "还没有可保存的内容", "请先导入视频并完成镜头分析。"
            )
            return False
        path = self._session_path
        if save_as or not path:
            path, _ = QFileDialog.getSaveFileName(
                self,
                "保存 Lensplit 工程",
                path or self._default_session_path(),
                f"Lensplit 工程 (*{PROJECT_SUFFIX});;所有文件 (*)",
            )
            if not path:
                return False
        try:
            saved = self.session_snapshot().save(path)
        except OSError as error:
            QMessageBox.warning(self, "无法保存工程", f"写入失败：{error}")
            return False
        self._session_path = str(saved)
        self._set_dirty(False)
        self.statusBar().showMessage(f"工程已保存：{saved}", 12000)
        return True

    def save_session_to(self, path: str) -> bool:
        """Save to an explicit path without asking (used by tests and scripts)."""

        try:
            saved = self.session_snapshot().save(path)
        except OSError as error:
            QMessageBox.warning(self, "无法保存工程", f"写入失败：{error}")
            return False
        self._session_path = str(saved)
        self._set_dirty(False)
        return True

    def open_project(self) -> bool:
        """Pick a session file and restore it."""

        if self._busy_kind:
            QMessageBox.information(self, "任务进行中", "请先等待当前任务结束或取消。")
            return False
        if not self._confirm_discard():
            return False
        start = str(Path(self._session_path).parent) if self._session_path else ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开 Lensplit 工程",
            start,
            f"Lensplit 工程 (*{PROJECT_SUFFIX});;所有文件 (*)",
        )
        if not path:
            return False
        return self.open_session_from(path)

    def open_session_from(self, path: str) -> bool:
        try:
            session = ProjectSession.load(path)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "无法打开工程", str(error))
            return False
        self._apply_session(session, path)
        return True

    def _apply_session(self, session: ProjectSession, path: str | None = None) -> None:
        self.player.stop()
        self.player.setSource(QUrl())
        self._thumbnail_generation += 1
        self._thumbnail_cache.clear()
        self._thumbnail_pending.clear()
        self.shooting_page.clear_thumbnails()
        self._history.clear()
        self._slot_state = None

        self.video_path = session.video_path or None
        self.video_info = None
        if self.video_path and Path(self.video_path).is_file():
            try:
                self.video_info = probe_video(self.video_path)
            except Exception:
                self.video_info = None
            self.player.setSource(QUrl.fromLocalFile(self.video_path))
        self._update_video_labels()

        self.project = session.project
        self.casting = session.casting
        self._transcripts = dict(session.transcripts)
        self._analysis_metadata = dict(session.analysis)
        self._apply_numbering(session.numbering)
        self._preview_end = None
        self._preview_label = None
        self._refresh_project(select_first=True)
        self._session_path = path
        self._set_dirty(False)
        if self.video_path and not Path(self.video_path).is_file():
            QMessageBox.warning(
                self,
                "视频文件不见了",
                f"找不到视频：{self.video_path}\n\n"
                "分镜、演员、台词数据已载入，可以继续查看和编辑；"
                "把视频放回原位置后即可播放与导出。",
            )
        self.statusBar().showMessage(f"工程已打开：{path or '未命名'}", 12000)

    def _apply_numbering(self, state: dict) -> None:
        values = {**DEFAULT_NUMBERING, **dict(state or {})}
        self._refreshing = True
        try:
            self.start_spin.setValue(int(values.get("start", 1)))
            self.width_spin.setValue(int(values.get("width", 3)))
            self.prefix_edit.setText(str(values.get("prefix", "")))
            self.suffix_edit.setText(str(values.get("suffix", "")))
            sequence = self.sequence_combo.findData(values.get("sequence"))
            self.sequence_combo.setCurrentIndex(sequence if sequence >= 0 else 0)
            merge = self.merge_mode_combo.findData(values.get("merge_mode"))
            self.merge_mode_combo.setCurrentIndex(merge if merge >= 0 else 0)
        finally:
            self._refreshing = False

    def _set_dirty(self, dirty: bool) -> None:
        self._dirty = dirty
        name = Path(self._session_path).stem if self._session_path else "未命名工程"
        marker = " *" if dirty else ""
        self.setWindowTitle(f"Lensplit · {name}{marker} · 本地镜头切分与合并")

    def _mark_dirty(self) -> None:
        if not self._dirty:
            self._set_dirty(True)

    def _confirm_discard(self) -> bool:
        if not self._dirty or self.project is None:
            return True
        answer = QMessageBox.question(
            self,
            "工程尚未保存",
            "当前工程有未保存的修改，是否先保存？",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if answer == QMessageBox.Cancel:
            return False
        if answer == QMessageBox.Save:
            return self.save_project()
        return True

    def _engine_changed(self) -> None:
        self.threshold_spin.setValue(
            0.60 if self.engine_combo.currentData() == "fast" else 0.40
        )

    @staticmethod
    def _default_model_path() -> str | None:
        override = os.environ.get("LENSPLIT_MODEL")
        candidates = [Path(override)] if override else []
        candidates.append(Path.cwd() / "models" / "transnetv2.onnx")
        candidates.append(
            Path(__file__).resolve().parents[3] / "models" / "transnetv2.onnx"
        )
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        return None

    def _auto_load_default_model(self) -> None:
        path = self._default_model_path()
        if not path:
            return
        if not self._apply_model(path, announce=False):
            return
        self.engine_combo.setCurrentIndex(1)
        self.statusBar().showMessage(
            f"已自动加载本地模型：{Path(path).name} · 阈值 {self.threshold_spin.value():.2f}"
        )

    def _apply_model(self, path: str, announce: bool) -> bool:
        detector = TransNetDetector(model_path=path)
        self._model_status = detector.status
        self._analysis_metadata["model_path"] = path
        self._update_model_controls()
        self._update_action_states()
        if not self._model_status.available and announce:
            QMessageBox.warning(
                self,
                "本地模型不可用",
                self._model_status.reason or "无法配置该模型。AI 模式保持禁用。",
            )
        return self._model_status.available

    def select_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择本地 TransNetV2 ONNX 模型", "", "ONNX 模型 (*.onnx);;所有文件 (*)"
        )
        if not path:
            return
        self._apply_model(path, announce=True)

    def clear_model(self) -> None:
        self._model_status = LocalModelStatus(configured=False, available=False)
        self._analysis_metadata.pop("model_path", None)
        self._update_model_controls()
        self._update_action_states()

    def _update_model_controls(self) -> None:
        status = self._model_status
        item = self.engine_combo.model().item(1)
        if item is not None:
            item.setEnabled(status.available)
        if not status.available and self.engine_combo.currentData() == "transnet":
            self.engine_combo.setCurrentIndex(0)
        self.model_path_label.setText(status.path or "未配置模型")
        self.model_path_label.setToolTip(status.path or "")
        if not status.configured:
            text = "AI 模式已禁用；请选择本机 ONNX 文件。极速模式可直接使用。"
        elif not status.available:
            text = f"AI 模式已禁用：{status.reason or '模型不可用'}"
        else:
            providers = " → ".join(status.providers)
            text = f"模型就绪 · {status.active_provider or providers}"
        self.model_status_label.setText(text)
        detail = status.reason or (
            f"路径：{status.path}\nSHA256：{status.sha256 or '未知'}\n"
            f"Provider 顺序：{' → '.join(status.providers)}"
        )
        self.model_status_label.setToolTip(detail)

    def start_analysis(self) -> None:
        if self._busy_kind or not self.video_path:
            return
        engine = self.engine_combo.currentData()
        duration = (self.video_info.duration or 0.0) if self.video_info else 0.0
        if engine != "fast" and duration > 600:
            answer = QMessageBox.question(
                self,
                "AI 分析预计较慢",
                f"当前视频约 {duration / 60:.0f} 分钟，AI 逐帧分析会持续较长时间。\n"
                "是否改用「极速」模式（通常几十秒出结果）？",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes,
            )
            if answer == QMessageBox.Cancel:
                return
            if answer == QMessageBox.Yes:
                self.engine_combo.setCurrentIndex(0)
                engine = "fast"
        try:
            if engine == "fast":
                config = FastDetectionConfig(
                    threshold=self.threshold_spin.value(),
                    min_shot_seconds=self.min_shot_spin.value(),
                )
                model_path = None
            else:
                if not self._model_status.available:
                    raise ValueError("本地 TransNetV2 模型未配置或不可用，AI 模式已禁用。")
                model_path = self._model_status.path
                config = TransNetConfig(
                    threshold=self.threshold_spin.value(),
                    min_shot_seconds=self.min_shot_spin.value(),
                    model_path=model_path,
                )
        except ValueError as error:
            QMessageBox.warning(self, "检测参数无效", str(error))
            return

        metadata = {
            "engine": engine,
            "parameters": asdict(config),
            "model": asdict(self._model_status) if engine == "transnet" else None,
        }
        self._task_serial += 1
        token = self._task_serial
        self._task_started = time.monotonic()
        self._analysis_metadata = metadata
        worker = AnalysisWorker(
            engine,
            self.video_path,
            config=config,
            model_path=model_path,
            on_progress=lambda value: self._events.put(("analysis_progress", token, value)),
            on_result=lambda result: self._events.put(("analysis_result", token, result)),
            on_error=lambda error: self._events.put(("analysis_error", token, error)),
            on_warning=lambda text: self._events.put(("analysis_warning", token, text)),
            on_model_status=lambda status: self._events.put(("model_status", token, status)),
        )
        self._analysis_worker = worker
        self._set_busy("analysis")
        self._progress(0.0, "正在本地分析视频…")
        try:
            worker.start()
        except Exception:
            self._analysis_worker = None
            self._set_busy(None)
            raise

    def set_project(self, project: ShotProject, analysis_metadata: dict | None = None) -> None:
        """Install a validated project and rebuild all visible numbering."""
        self.player.pause()
        self.project = project
        self.casting = CastingData()
        self._history.clear()
        if analysis_metadata is not None:
            self._analysis_metadata = analysis_metadata
        self._thumbnail_generation += 1
        self._thumbnail_cache.clear()
        self._thumbnail_pending.clear()
        self.shooting_page.clear_thumbnails()
        self._refresh_project(select_first=True)
        self._preview_end = None
        self._session_path = None
        self._mark_dirty()

    def _drain_events(self) -> None:
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                return
            kind = event[0]
            if kind == "thumbnail":
                self._thumbnail_ready(*event[1:])
                continue
            token = event[1]
            if token != self._task_serial:
                continue
            if kind == "analysis_progress" and self._analysis_worker is not None:
                self._progress(event[2], "正在本地分析视频…")
            elif kind == "analysis_warning" and self._analysis_worker is not None:
                self.statusBar().showMessage(f"检测提示：{event[2]}", 10000)
            elif kind == "model_status" and self._analysis_worker is not None:
                self._model_status = event[2]
                self._analysis_metadata["model"] = asdict(event[2])
                self._update_model_controls()
            elif kind == "analysis_result" and self._analysis_worker is not None:
                self._analysis_worker = None
                self._analysis_metadata["elapsed_seconds"] = time.monotonic() - self._task_started
                self._set_busy(None)
                self.set_project(event[2])
                stats = transition_statistics(event[2])
                self.statusBar().showMessage(
                    f"分析完成：{len(event[2].source_shots)} 个原始镜头，{stats['total']} 个转场，"
                    f"耗时 {self._analysis_metadata['elapsed_seconds']:.2f} 秒"
                )
            elif kind == "analysis_error" and self._analysis_worker is not None:
                worker = self._analysis_worker
                self._analysis_worker = None
                self._set_busy(None)
                if worker.cancelled or isinstance(event[2], DecodeCancelledError):
                    self.statusBar().showMessage("分析已取消，未使用部分检测结果")
                else:
                    self.statusBar().showMessage("分析失败")
                    QMessageBox.critical(self, "本地分析失败", str(event[2]))
            elif kind == "export_progress" and self._export_worker is not None:
                self._progress(event[2], "正在本地导出镜头…")
            elif kind == "export_warning" and self._export_worker is not None:
                self.statusBar().showMessage(f"导出提示：{event[2]}", 10000)
            elif kind == "export_result" and self._export_worker is not None:
                self._export_worker = None
                self._set_busy(None)
                self._export_finished(event[2])
            elif kind == "export_error" and self._export_worker is not None:
                worker = self._export_worker
                self._export_worker = None
                self._set_busy(None)
                if worker.cancelled or isinstance(event[2], ExportCancelledError):
                    self.statusBar().showMessage("导出已取消，未完成的 .part 文件已清理")
                else:
                    self.statusBar().showMessage("导出失败")
                    QMessageBox.critical(self, "本地导出失败", str(event[2]))
            elif kind == "asr_progress" and self._asr_worker is not None:
                index, total = event[2], max(1, event[3])
                self._progress(index / total, "正在本地识别台词…")
                self.asr_status_label.setText(f"识别中 {index}/{total}")
            elif kind == "asr_clip" and self._asr_worker is not None:
                self._apply_transcript(event[2], event[3])
            elif kind == "asr_result" and self._asr_worker is not None:
                self._asr_worker = None
                self._set_busy(None)
                self.asr_cancel_button.setEnabled(False)
                clips = len(event[2])
                spoken = sum(1 for item in event[2].values() if item.has_speech)
                self.asr_status_label.setText(f"完成 {clips} 个镜头，{spoken} 个有台词")
                self._refresh_project()
                self.statusBar().showMessage(
                    f"台词识别完成：{spoken}/{clips} 个镜头有台词，其余标记为无台词"
                )
            elif kind == "asr_error" and self._asr_worker is not None:
                self._asr_worker = None
                self._set_busy(None)
                self.asr_cancel_button.setEnabled(False)
                if isinstance(event[2], AsrCancelledError):
                    self.asr_status_label.setText("已取消，保留已完成部分")
                    self._refresh_project()
                    self.statusBar().showMessage("台词识别已取消，已完成部分保留")
                else:
                    self.asr_status_label.setText("识别失败")
                    self.statusBar().showMessage("台词识别失败")
                    QMessageBox.critical(self, "台词识别失败", str(event[2]))
            elif kind == "person_progress" and self._person_worker is not None:
                self._progress(event[2], "正在本地识别画面人物…")
                self.person_status_label.setText(f"人物标记中 · {event[2]:.0%}")
            elif kind == "person_shot" and self._person_worker is not None:
                group_id, count = event[2], event[3]
                self.casting.set_character_count(group_id, count)
                self.shooting_page.apply_character_count(group_id, count)
                if self.timeline.current_id() == group_id:
                    self._actor_panel_loading = True
                    self.actor_count_spin.setValue(count)
                    self._actor_panel_loading = False
            elif kind == "person_result" and self._person_worker is not None:
                self._person_worker = None
                self._set_busy(None)
                counts = event[2]
                self._person_note = f"人物标记完成 · {len(counts)} 镜，人数已写入拍摄表"
                self._mark_dirty()
                self.person_status_label.setText(self._person_note)
                self._refresh_project(select_group_id=self._current_shot())
                self.statusBar().showMessage(
                    f"人物标记完成：{len(counts)} 个镜头的人数已写入拍摄表"
                )
            elif kind == "person_error" and self._person_worker is not None:
                worker = self._person_worker
                self._person_worker = None
                self._set_busy(None)
                if worker.cancelled or isinstance(event[2], PersonCancelledError):
                    self._person_note = "人物标记已取消，已完成的镜头保留"
                    self.person_status_label.setText(self._person_note)
                    self.statusBar().showMessage("人物标记已取消")
                else:
                    self._person_note = ""
                    self.person_status_label.setText("人物标记失败")
                    self.statusBar().showMessage("人物标记失败")
                    QMessageBox.critical(self, "人物标记失败", str(event[2]))

    def _set_busy(self, kind: str | None) -> None:
        self._busy_kind = kind
        if kind is None:
            self.progress_bar.hide()
            self.cancel_task_button.hide()
            self.analyze_button.setText("开始分析")
        else:
            self.progress_bar.show()
            self.cancel_task_button.show()
            self.cancel_task_button.setEnabled(True)
            self.cancel_task_button.setText(
                {
                    "analysis": "取消分析",
                    "export": "取消导出",
                    "asr": "取消识别",
                    "person": "取消标记",
                }.get(kind, "取消任务")
            )
            if kind == "analysis":
                self.analyze_button.setText("分析中…")
        self._update_action_states()

    def _progress(self, value: float, text: str) -> None:
        self.progress_bar.setValue(round(max(0.0, min(1.0, value)) * 1000))
        self.statusBar().showMessage(f"{text} {value:.1%}")
        if self._busy_kind == "analysis":
            self.analyze_button.setText(f"分析中 {value:.0%}…")

    def _cancel_and_reset_busy(self) -> None:
        workers = (self._analysis_worker, self._asr_worker, self._person_worker)
        for worker in workers:
            if worker is not None:
                worker.cancel()
        self._analysis_worker = None
        self._asr_worker = None
        self._person_worker = None
        self._task_serial += 1
        self._set_busy(None)
        self.statusBar().showMessage("已取消当前任务")

    def cancel_current_task(self) -> None:
        worker = (
            self._analysis_worker
            or self._export_worker
            or self._asr_worker
            or self._person_worker
        )
        if worker is None:
            return
        worker.cancel()
        self.cancel_task_button.setEnabled(False)
        self.statusBar().showMessage("正在取消并清理临时数据…")

    def numbering_style(self) -> NumberingStyle:
        return NumberingStyle(
            start=self.start_spin.value(),
            width=self.width_spin.value(),
            prefix=self.prefix_edit.text(),
            suffix=self.suffix_edit.text(),
            sequence=self.sequence_combo.currentData(),
        )

    def merge_mode(self) -> MergeMode:
        return MergeMode(self.merge_mode_combo.currentData())

    def _numbering_changed(self) -> None:
        if not self._refreshing:
            self._refresh_project()

    def _refresh_project(self, select_first: bool = False, select_group_id: str | None = None) -> None:
        if self._refreshing:
            return
        self._refreshing = True
        try:
            if self.project is None:
                self.timeline.set_items([])
                self.stats_label.setText("原始转场 0 ｜ 有效转场 0 ｜ 合并镜内部转场 0")
                self.number_example_label.setText("示例：001、002、003")
                self._group_labels = []
                self._group_ranges = []
                self._update_badge()
                self._refresh_transitions()
                self._update_action_states()
                self._refresh_shooting_table()
                self._refresh_actor_schedule()
                self._refresh_actor_panel()
                self._update_person_controls()
                return
            labels = format_groups(self.project.groups, self.numbering_style(), self.merge_mode())
            self._group_labels = list(labels)
            self._group_ranges = [
                self.project.group_interval(group.uuid) for group in self.project.groups
            ]
            source_shots = {shot.uuid: shot for shot in self.project.source_shots}
            items = [
                TimelineItem(
                    uuid=group.uuid,
                    label=label,
                    start=start,
                    end=end,
                    source_count=len(group.source_ids),
                    transcript=self._transcripts.get(group.uuid, "").strip(),
                    recognized=group.uuid in self._transcripts,
                    source_text="、".join(
                        f"{source_shots[source_id].ordinal:03d}" for source_id in group.source_ids
                    ),
                )
                for group, label, (start, end) in zip(
                    self.project.groups, labels, self._group_ranges
                )
            ]
            self.timeline.set_items(items)
            known = [item.uuid for item in items]
            if select_group_id and select_group_id in known:
                self.timeline.set_selection([select_group_id], select_group_id)
            elif select_first and known:
                self.timeline.set_selection([known[0]], known[0])
            else:
                current = self.timeline.current_id()
                if current not in known:
                    current = self.timeline.selected_ids()[0] if self.timeline.selected_ids() else None
                self.timeline.set_selection(self.timeline.selected_ids(), current)
            stats = transition_statistics(self.project)
            self.stats_label.setText(
                f"原始转场 {stats['total']} ｜ 有效转场 {stats['effective']} ｜ "
                f"合并镜内部转场 {stats['internal']}"
            )
            self.number_example_label.setText("当前镜号：" + "、".join(labels[:5]) + (" …" if len(labels) > 5 else ""))
            self._refresh_transitions()
        finally:
            self._refreshing = False
        if self.project is not None and self.video_path:
            for group in self.project.groups:
                self._request_thumbnail(group.uuid)
        self._update_action_states()
        self._update_badge()
        self._sync_slider_marks()
        self._refresh_shooting_table()
        self._refresh_actor_schedule()
        self._refresh_actor_panel()
        self._update_person_controls()
        self._update_slider_scope()

    def _labels_map(self) -> dict[str, str]:
        if self.project is None:
            return {}
        return {
            group.uuid: label
            for group, label in zip(self.project.groups, self._group_labels)
        }

    def _refresh_shooting_table(self) -> None:
        self.shooting_page.set_context(
            self.project, self.casting, self._labels_map(), self._transcripts
        )
        if self.project is not None:
            for group_id, pixmap in self._thumbnail_cache.items():
                self.shooting_page.set_thumbnail(group_id, pixmap)
        if self.video_path:
            stem = Path(self.video_path).stem
            self.shooting_page.set_export_directory(
                str(Path(self.video_path).parent / f"{stem}_拍摄表.csv")
            )
        self._update_shooting_tab()

    def _update_shooting_tab(self) -> None:
        issues = self.casting.conflicts(self._labels_map())
        errors = sum(1 for issue in issues if issue.is_error)
        self.tabs.setTabText(1, f"拍摄表 · {errors} 处撞档" if errors else "拍摄表")
        self._update_actor_tab()

    def _update_actor_tab(self) -> None:
        outside = sum(
            len(self.casting.check_actor_schedule(actor.uuid))
            for actor in self.casting.actors
        )
        self.tabs.setTabText(
            2, f"演员与档期 · {outside} 镜越界" if outside else "演员与档期"
        )

    def _refresh_actor_schedule(self) -> None:
        self.actor_page.set_context(self.project, self.casting, self._labels_map())
        self._update_actor_tab()

    def _tab_changed(self, index: int) -> None:
        if index == 1:
            self.shooting_page.activate()
        elif index == 2:
            self.actor_page.activate()

    def _preview_from_shooting(self, group_uuid: str) -> None:
        if self.project is None or not self.video_path:
            self.statusBar().showMessage("请先在「剪辑」页导入视频，再播放分镜")
            return
        self.tabs.setCurrentIndex(0)
        self.timeline.set_selection([group_uuid], group_uuid)
        self._preview_group(group_uuid)

    def _shooting_transcript_changed(self, group_uuid: str, text: str) -> None:
        if self.project is None:
            return
        self._transcripts[group_uuid] = text.strip()
        self._mark_dirty()
        self._update_item_transcript(group_uuid)
        if self.timeline.current_id() == group_uuid:
            self.transcript_edit.setPlainText(text)

    def _shooting_data_changed(self) -> None:
        self._mark_dirty()
        self._update_shooting_tab()
        self._refresh_actor_panel()
        issues = self.casting.conflicts(self._labels_map())
        self.statusBar().showMessage(
            f"拍摄表已更新 · 当前 {len(issues)} 条档期提示"
            if issues
            else "拍摄表已更新"
        )

    def _actor_schedule_changed(self) -> None:
        self._mark_dirty()
        if self.project is not None:
            self.shooting_page.rebuild()
        self._update_shooting_tab()
        self._update_actor_tab()
        self._refresh_actor_panel()

    # -- page 1: current shot panel --------------------------------------
    @staticmethod
    def _default_person_model_path() -> str:
        override = os.environ.get("LENSPLIT_PERSON_MODEL")
        candidates = [Path(override)] if override else []
        candidates.append(Path.cwd() / "models" / "yolov8n.onnx")
        candidates.append(Path(__file__).resolve().parents[3] / "models" / "yolov8n.onnx")
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        return str(candidates[-1])

    def _person_ready(self) -> bool:
        return Path(self._person_model_path).is_file()

    def _update_person_controls(self) -> None:
        status = YoloPersonDetector(
            PersonDetectConfig(model_path=self._person_model_path)
        ).status
        busy = bool(self._busy_kind)
        usable = status.available and self.project is not None and bool(self.video_path)
        self.person_mark_button.setEnabled(usable and not busy)
        self.person_mark_selected_button.setEnabled(usable and not busy)
        tooltip = status.message
        if status.available:
            tooltip = "用本地 YOLOv8n 统计每个镜头的人物数量并写入拍摄表\n模型：" + status.path
        self.person_mark_button.setToolTip(tooltip)
        self.person_mark_selected_button.setToolTip(tooltip)
        if self._busy_kind == "person":
            return
        if self._person_note:
            self.person_status_label.setText(self._person_note)
        elif self.project is None or not self.video_path:
            self.person_status_label.setText("人物标记：等待导入视频")
        elif not status.available:
            self.person_status_label.setText(f"人物标记：不可用（{status.message}）")
        else:
            self.person_status_label.setText(
                f"人物标记：{status.message} · {Path(status.path).name}"
            )

    def _current_shot(self) -> str | None:
        if self.project is None:
            return None
        return self.timeline.current_id()

    def _shot_label(self, group_uuid: str) -> str:
        index = next(
            (
                index
                for index, group in enumerate(self.project.groups)
                if group.uuid == group_uuid
            ),
            None,
        ) if self.project is not None else None
        if index is not None and 0 <= index < len(self._group_labels):
            return self._group_labels[index]
        return group_uuid[:8]

    def _actor_conflict(self, group_uuid: str, actor) -> bool:
        """True when the shot day is known and falls outside the actor's window."""

        if actor is None:
            return False
        shot = self.casting.shot(group_uuid)
        day = shot.shoot_date if shot is not None else ""
        if not day or (not actor.available_from and not actor.available_to):
            return False
        return not actor.is_available(day)

    def _schedule_text(self, group_uuid: str) -> str:
        """One-line status for the current shot: shoot day plus casting state."""

        roles = self.casting.roles_for_shot(group_uuid)
        shot = self.casting.shot(group_uuid)
        day = shot.shoot_date if shot is not None else ""
        if not roles:
            if shot is not None and shot.character_count:
                return f"已标注 {shot.character_count} 人 · 角色槽正在生成"
            return "尚未标注人数：可先用「人物自动标记」，也可以手动填人数。"
        parts = [
            f"拍摄日 {day}" if day else "拍摄日未定（填日期后用演员档期校验冲突）"
        ]
        pending = sum(1 for role in roles if not role.actor_uuid)
        if pending:
            parts.append(f"{pending} 个角色待分配")
        clashes = [
            actor.name
            for actor in self.casting.actors_for_shot(group_uuid)
            if self._actor_conflict(group_uuid, actor)
        ]
        if clashes:
            parts.append("档期冲突：" + "、".join(clashes))
        elif day and not pending:
            parts.append("档期无冲突")
        return " · ".join(parts)

    def _refresh_actor_panel(self) -> None:
        current = self._current_shot()
        shot = self.casting.shot(current) if current else None
        self._actor_panel_loading = True
        try:
            if current is None:
                self.actor_shot_label.setText("—")
                self.actor_count_spin.setValue(0)
                self.actor_date_edit.setDate(QDate(2000, 1, 1))
                self.actor_schedule_label.setText(
                    "在下方镜头序列中选择一个镜头，即可标注人数、拍摄日期与角色。"
                )
            else:
                self.actor_shot_label.setText(self._shot_label(current))
                self.actor_count_spin.setValue(shot.character_count if shot else 0)
                self.actor_date_edit.setDate(
                    _parse_qdate(shot.shoot_date if shot else "")
                )
                self.actor_schedule_label.setText(self._schedule_text(current))
        finally:
            self._actor_panel_loading = False
        self._render_slots(current or "")
        has_shot = current is not None and not self._busy_kind
        self.actor_count_spin.setEnabled(has_shot)
        self.actor_date_edit.setEnabled(has_shot)
        self.slot_scroll.setEnabled(has_shot)

    def _actor_count_changed(self, value: int) -> None:
        if self._actor_panel_loading or self._busy_kind:
            return
        current = self._current_shot()
        if current is None:
            return
        self.casting.set_character_count(current, value)
        self.shooting_page.apply_character_count(current, value)
        self._mark_dirty()
        self._update_shooting_tab()
        self._rebuild_slots(current)
        self.actor_schedule_label.setText(self._schedule_text(current))

    def _actor_date_changed(self, qdate: QDate) -> None:
        if self._actor_panel_loading or self._busy_kind:
            return
        current = self._current_shot()
        if current is None:
            return
        value = "" if qdate == QDate(2000, 1, 1) else qdate.toString("yyyy-MM-dd")
        self.casting.set_shoot_date(current, value)
        self.shooting_page.apply_shoot_date(current, value)
        if value:
            self.casting.fill_missing_windows(current)
        self._mark_dirty()
        self._slot_state = None
        self.actor_schedule_label.setText(self._schedule_text(current))
        self._rebuild_slots(current)
        self._update_shooting_tab()
        self._refresh_actor_schedule()

    def _actor_brief(self, actor) -> str:
        """Page 1 shows an actor's traits, never their availability window."""

        if actor is None:
            return "尚未分配 · 从演员库选择"
        return actor.notes or "未填写特点备注"

    def _slot_fingerprint(self, group_uuid: str) -> tuple:
        if not group_uuid:
            return ()
        shot = self.casting.shot(group_uuid)
        day = shot.shoot_date if shot is not None else ""
        parts = []
        for role in self.casting.shot_slots(group_uuid):
            actor = self.casting.actor(role.actor_uuid)
            parts.append(
                (
                    role.uuid,
                    role.name,
                    role.actor_uuid,
                    actor.notes if actor is not None else "",
                    self._actor_conflict(group_uuid, actor),
                )
            )
        return (group_uuid, day, tuple(parts))

    def _render_slots(self, group_uuid: str) -> None:
        """Rebuild the role slot rows only when their data changed."""

        fingerprint = self._slot_fingerprint(group_uuid)
        if fingerprint == self._slot_state:
            return
        self._slot_state = fingerprint
        self._rebuild_slots(group_uuid)

    def _rebuild_slots(self, group_uuid: str) -> None:
        self._slot_syncing = True
        try:
            while self.slot_layout.count():
                item = self.slot_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            slots = self.casting.shot_slots(group_uuid) if group_uuid else []
            if not slots:
                hint = QLabel(
                    "选中镜头后：人数 1 → 角色 1，人数 3 → 角色 1/2/3，可逐个改名并分配演员。"
                )
                hint.setObjectName("mutedLabel")
                hint.setWordWrap(True)
                self.slot_layout.addWidget(hint)
                return
            for index, role in enumerate(slots, start=1):
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(6)
                name_edit = QLineEdit(role.name)
                name_edit.setMaxLength(24)
                name_edit.setFixedWidth(104)
                name_edit.setToolTip(f"第 {index} 个角色的名称，可直接改")
                name_edit.editingFinished.connect(
                    lambda r=role, e=name_edit: self._slot_renamed(r.uuid, e.text())
                )
                row_layout.addWidget(name_edit)
                combo = QComboBox()
                combo.setMinimumWidth(118)
                combo.addItem("未分配演员", "")
                for actor in self.casting.actors:
                    combo.addItem(actor.name, actor.uuid)
                combo.setCurrentIndex(max(0, combo.findData(role.actor_uuid)))
                combo.currentIndexChanged.connect(
                    lambda _index, r=role, c=combo: self._slot_actor_changed(
                        r.uuid, c.currentData()
                    )
                )
                row_layout.addWidget(combo)
                style_dark_combo(combo)
                actor = self.casting.actor(role.actor_uuid)
                info = QLabel(self._actor_brief(actor))
                info.setObjectName("mutedLabel")
                info.setWordWrap(False)
                row_layout.addWidget(info, 1)
                if self._actor_conflict(group_uuid, actor):
                    tag = QLabel("档期冲突")
                    tag.setObjectName("conflictTag")
                    shot = self.casting.shot(group_uuid)
                    tag.setToolTip(
                        f"{actor.name} 的档期不包含拍摄日 {shot.shoot_date if shot else ''}"
                    )
                    row_layout.addWidget(tag)
                self.slot_layout.addWidget(row)
        finally:
            self._slot_syncing = False

    def _slot_renamed(self, role_uuid: str, text: str) -> None:
        if self._slot_syncing:
            return
        try:
            self.casting.rename_role(role_uuid, text)
        except ValueError:
            self.statusBar().showMessage("角色名不能为空")
            self._slot_state = None
            QTimer.singleShot(0, self._refresh_actor_panel)
            return
        self._mark_dirty()
        self._refresh_shooting_table()
        current = self._current_shot()
        if current:
            self.actor_schedule_label.setText(self._schedule_text(current))

    def _slot_actor_changed(self, role_uuid: str, actor_uuid: str) -> None:
        if self._slot_syncing:
            return
        try:
            self.casting.assign_actor(role_uuid, actor_uuid or "")
        except ValueError:
            return
        self._mark_dirty()
        self._refresh_shooting_table()
        self._refresh_actor_schedule()
        self._slot_state = None
        self._refresh_actor_panel()
        current = self._current_shot()
        if current:
            actor = self.casting.actor(actor_uuid)
            if actor is not None and self._actor_conflict(current, actor):
                self.statusBar().showMessage(
                    f"{self._shot_label(current)}：{actor.name} 与拍摄日档期冲突",
                    12000,
                )
            else:
                self.statusBar().showMessage(f"{self._shot_label(current)}：演员已更新")

    # -- page 1: local person detection ----------------------------------
    def start_person_detection(self, only_selected: bool = False) -> None:
        if self._busy_kind or self.project is None or not self.video_path:
            return
        groups = list(self.project.groups)
        if only_selected:
            chosen = set(self._selected_group_ids())
            groups = [group for group in groups if group.uuid in chosen]
        if not groups:
            self.statusBar().showMessage("请先在时间线选择要标记的镜头")
            return
        config = PersonDetectConfig(
            model_path=self._person_model_path,
            confidence=self.person_confidence_spin.value(),
            samples_per_shot=self.person_samples_spin.value(),
        )
        status = YoloPersonDetector(config).status
        if not status.available:
            QMessageBox.warning(self, "人物检测模型不可用", status.message)
            return
        intervals = [
            (group.uuid, *self.project.group_interval(group.uuid)) for group in groups
        ]
        self._task_serial += 1
        token = self._task_serial
        self._task_started = time.monotonic()
        worker = PersonDetectWorker(
            self.video_path,
            intervals,
            config,
            on_progress=lambda value: self._events.put(
                ("person_progress", token, value)
            ),
            on_shot=lambda group_id, count: self._events.put(
                ("person_shot", token, group_id, count)
            ),
            on_result=lambda counts: self._events.put(
                ("person_result", token, counts)
            ),
            on_error=lambda error: self._events.put(("person_error", token, error)),
        )
        self._person_worker = worker
        self._person_note = ""
        self._set_busy("person")
        self._progress(0.0, "正在本地识别画面人物…")
        self.person_status_label.setText(f"人物标记中 · 共 {len(intervals)} 镜")
        try:
            worker.start()
        except Exception:
            self._person_worker = None
            self._set_busy(None)
            raise

    # -- schedule reconciliation -----------------------------------------
    def _schedule_edited(self, group_uuid: str) -> None:
        """Page 2 date edit: reconcile after the widget signal returns."""

        QTimer.singleShot(0, lambda: self._reconcile_group(group_uuid))

    def _actor_window_changed(self, actor_uuid: str) -> None:
        """Page 3 window edit: re-check the shots already planned for that actor."""

        QTimer.singleShot(0, lambda: self._reconcile_actor(actor_uuid))

    def _reconcile_group(self, group_uuid: str) -> None:
        if self.project is None:
            return
        seeded = self.casting.fill_missing_windows(group_uuid)
        issues = self.casting.check_group_schedule(group_uuid)
        note = ""
        if seeded:
            note = f"已把镜头日期回填到页面 3 的演员档期：{'、'.join(seeded)}"
        if issues:
            dialog = ScheduleConflictDialog(self, issues)
            dialog.exec()
            applied = self._apply_schedule_choice(dialog.choice, issues)
            note = applied if applied else note
        self._refresh_project(select_group_id=self._current_shot() or group_uuid)
        if note:
            self.statusBar().showMessage(note, 12000)

    def _reconcile_actor(self, actor_uuid: str) -> None:
        if self.project is None:
            return
        issues = [
            issue
            for issue in self.casting.check_actor_schedule(actor_uuid)
            if issue.kind == "outside"
        ]
        if issues:
            actor = self.casting.actor(actor_uuid)
            dialog = ScheduleConflictDialog(
                self,
                issues,
                title=f"档期变更 · {actor.name if actor else '演员'}",
            )
            dialog.exec()
            self._apply_schedule_choice(dialog.choice, issues)
        self._refresh_project(select_group_id=self._current_shot())

    def _apply_schedule_choice(
        self, choice: str, issues: list[AvailabilityIssue]
    ) -> str:
        if choice == "extend":
            changed = self.casting.extend_for_issues(issues)
            return f"已延长 {changed} 位演员的档期以覆盖拍摄日" if changed else ""
        if choice == "clamp":
            moved = self.casting.clamp_for_issues(issues)
            if not moved:
                return ""
            labels = "、".join(
                f"{self._shot_label(group_id)} → {day}" for group_id, day in moved
            )
            return f"已把拍摄日改为档期内：{labels}"
        return "已保留现状，越界提示会继续显示"

    def _request_thumbnail(self, group_uuid: str) -> None:
        if self.project is None or not self.video_path:
            return
        key = (self._thumbnail_generation, group_uuid)
        if group_uuid in self._thumbnail_cache or key in self._thumbnail_pending:
            return
        self._thumbnail_pending.add(key)
        generation = self._thumbnail_generation
        start, end = self.project.group_interval(group_uuid)
        target = start + min(0.25, max(0.0, end - start) * 0.25)
        future = self._thumbnail_executor.submit(extract_thumbnail, self.video_path, target, (288, 162))

        def completed(task) -> None:
            try:
                image = task.result()
            except Exception as error:
                self._events.put(("thumbnail", generation, group_uuid, error))
            else:
                self._events.put(("thumbnail", generation, group_uuid, image))

        future.add_done_callback(completed)

    def _thumbnail_ready(self, generation: int, group_uuid: str, result) -> None:
        self._thumbnail_pending.discard((generation, group_uuid))
        if generation != self._thumbnail_generation or self.project is None:
            return
        if isinstance(result, BaseException):
            return
        image = np.ascontiguousarray(result)
        if image.ndim != 3 or image.shape[2] != 3:
            return
        height, width = image.shape[:2]
        qimage = QImage(image.data, width, height, image.strides[0], QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage).scaled(
            QSize(344, 194), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
        )
        self._thumbnail_cache[group_uuid] = pixmap
        self.timeline.set_thumbnail(group_uuid, pixmap)
        self.shooting_page.set_thumbnail(group_uuid, pixmap)

    def _selected_group_ids(self) -> list[str]:
        if self.project is None:
            return []
        known = {group.uuid for group in self.project.groups}
        return [group_id for group_id in self.timeline.selected_ids() if group_id in known]

    def _mutate_project(
        self, operation: Callable[[], object], select_group_id: str | None = None
    ) -> bool:
        if self.project is None or self._busy_kind:
            return False
        before = self.project.snapshot()
        transcripts_before = dict(self._transcripts)
        casting_before = self.casting.snapshot()
        try:
            result = operation()
        except (ValueError, KeyError) as error:
            QMessageBox.warning(self, "无法完成编辑", str(error))
            return False
        self._history.append((before, transcripts_before, casting_before))
        del self._history[:-100]
        self._mark_dirty()
        selected = select_group_id or (result if isinstance(result, str) else None)
        self._refresh_project(select_group_id=selected)
        return True

    def merge_selected(self) -> None:
        if self.project is None:
            return
        selected = self._selected_group_ids()
        if len(selected) < 2:
            self.statusBar().showMessage("请先用 Ctrl 或 Shift 选择至少两个相邻逻辑镜头")
            return

        def merge() -> str:
            texts = [self._transcripts.get(group_id, "").strip() for group_id in selected]
            merged = self.project.merge_groups(selected)
            self.casting.merge_shots(selected, merged.uuid)
            for group_id in selected:
                self._transcripts.pop(group_id, None)
            joined = " ".join(text for text in texts if text)
            if joined:
                self._transcripts[merged.uuid] = joined
            return merged.uuid

        if self._mutate_project(merge):
            self.statusBar().showMessage("已合成一个连续逻辑镜头；内部转场与台词已合并保留")

    def split_selected(self) -> None:
        if self.project is None:
            return
        selected = self._selected_group_ids()
        if not selected:
            return

        def split() -> str | None:
            first_restored = None
            for group_id in selected:
                group = next(group for group in self.project.groups if group.uuid == group_id)
                if len(group.source_ids) > 1:
                    restored = self.project.split_group(group_id)
                    self.casting.split_shot(
                        group_id, [group.uuid for group in restored]
                    )
                    if first_restored is None:
                        first_restored = restored[0].uuid
            return first_restored

        self._mutate_project(split)

    def _slider_scope(self) -> tuple[float, float] | None:
        """The focused shot's (start, end) so the slider spans only that shot."""

        current = self._current_shot()
        if current is None or self.project is None:
            return None
        try:
            start, end = self.project.group_interval(current)
        except (KeyError, ValueError):
            return None
        return (start, end) if end > start else None

    def _update_slider_scope(self) -> None:
        """Map the position slider onto the selected shot instead of the whole clip."""

        scope = self._slider_scope()
        if scope is None:
            start_ms, end_ms = 0, max(0, self.player.duration())
        else:
            start_ms, end_ms = round(scope[0] * 1000), round(scope[1] * 1000)
        if end_ms <= start_ms:
            end_ms = start_ms + 1
        position = min(max(self.player.position(), start_ms), end_ms)
        self.position_slider.blockSignals(True)
        self.position_slider.setRange(start_ms, end_ms)
        self.position_slider.setValue(position)
        self.position_slider.blockSignals(False)
        self._sync_slider_marks()
        self._position_changed(position)

    def _sync_slider_marks(self) -> None:
        if self.project is None:
            self.position_slider.set_positions([], [])
            self.mark_label.setText("手动切点：0 个（黄线＝待应用标记，蓝线＝已有切点）")
            self.apply_marks_button.setEnabled(False)
            self.unmark_button.setEnabled(False)
            return
        scope = self._slider_scope()
        if scope is not None:
            start, end = scope
            span = max(0.001, end - start)
            cuts = [
                (transition.cut_pts - start) / span
                for transition in self.project.transitions
                if start <= transition.cut_pts <= end
            ]
            marks = [(mark - start) / span for mark in self._manual_marks if start <= mark <= end]
        else:
            duration = self.player.duration() / 1000
            if duration <= 0:
                self.position_slider.set_positions([], [])
                self.mark_label.setText("手动切点：0 个（黄线＝待应用标记，蓝线＝已有切点）")
                self.apply_marks_button.setEnabled(False)
                self.unmark_button.setEnabled(False)
                return
            cuts = [transition.cut_pts / duration for transition in self.project.transitions]
            marks = [mark / duration for mark in self._manual_marks]
        self.position_slider.set_positions(cuts, marks)
        self.mark_label.setText(
            f"手动切点：{len(self._manual_marks)} 个 · "
            f"{'刻度＝当前片段' if scope is not None else '刻度＝整段视频'}"
            "（黄线＝待应用标记，蓝线＝已有切点）"
        )
        editable = bool(self._manual_marks) and not self._busy_kind
        self.apply_marks_button.setEnabled(editable)
        self.unmark_button.setEnabled(editable)

    def _group_at(self, seconds: float):
        if self.project is None:
            return None
        for group in self.project.groups:
            start, end = self.project.group_interval(group.uuid)
            if start <= seconds < end:
                return group
        return None

    def add_manual_mark(self) -> None:
        if self.project is None or self.video_path is None:
            self.statusBar().showMessage("请先导入并分析视频，再标记切点")
            return
        seconds = self.player.position() / 1000
        if seconds <= 0:
            self.statusBar().showMessage("请先把进度条拖到要切分的位置")
            return
        group = self._group_at(seconds)
        if group is None:
            self.statusBar().showMessage("当前位置不在任何镜头区间内")
            return
        start, end = self.project.group_interval(group.uuid)
        if seconds - start < 0.05 or end - seconds < 0.05:
            self.statusBar().showMessage("切点太靠近镜头边界，请稍向内移动")
            return
        if any(abs(seconds - mark) < 0.05 for mark in self._manual_marks):
            self.statusBar().showMessage("该位置已经标记过切点")
            return
        if any(abs(seconds - item.cut_pts) < 0.05 for item in self.project.transitions):
            self.statusBar().showMessage("该位置已存在检测切点")
            return
        self._manual_marks.append(seconds)
        self._manual_marks.sort()
        self._sync_slider_marks()
        self.statusBar().showMessage(f"已标记切点 {seconds:.3f}s，共 {len(self._manual_marks)} 个")

    def remove_last_mark(self) -> None:
        if not self._manual_marks:
            return
        removed = self._manual_marks.pop()
        self._sync_slider_marks()
        self.statusBar().showMessage(f"已撤销切点 {removed:.3f}s")

    def apply_manual_marks(self) -> None:
        if self.project is None or not self._manual_marks:
            return
        marks = sorted(self._manual_marks)

        def split() -> str | None:
            first_id = None
            for cut in marks:
                group = self._group_at(cut)
                if group is None:
                    continue
                split_groups = self.project.split_group_at(group.uuid, cut)
                self.casting.split_shot(
                    group.uuid, [item.uuid for item in split_groups]
                )
                if first_id is None:
                    first_id = split_groups[0].uuid
            return first_id

        if self._mutate_project(split):
            self._manual_marks.clear()
            self._refresh_project(select_first=True)
            self.statusBar().showMessage(
                f"已按 {len(marks)} 个手动切点拆成独立镜头，镜号与转场统计已重排"
            )

    def _slider_released(self) -> None:
        self._preview_end = None
        self._preview_label = None
        self._preview_timer.stop()
        self._update_badge()

    def _default_asr_model_path(self) -> str | None:
        override = os.environ.get("LENSPLIT_ASR_MODEL")
        candidates = [Path(override)] if override else []
        candidates.append(Path.cwd() / "models" / "whisper-small")
        candidates.append(Path(__file__).resolve().parents[3] / "models" / "whisper-small")
        for candidate in candidates:
            if (candidate / "model.bin").is_file():
                return str(candidate)
        return None

    def start_transcription(self) -> None:
        if self.project is None or self.video_path is None:
            self.statusBar().showMessage("请先导入并分析视频")
            return
        if self._busy_kind:
            return
        model_path = self._default_asr_model_path()
        if model_path is None:
            QMessageBox.warning(
                self,
                "缺少本地台词模型",
                "未找到本地 Whisper 模型。请把 CTranslate2 模型放到 models/whisper-small，"
                "或设置环境变量 LENSPLIT_ASR_MODEL 指向模型目录。",
            )
            return
        if (
            self._asr_transcriber is None
            or self._asr_transcriber.config.model_path != model_path
        ):
            self._asr_transcriber = LocalTranscriber(AsrConfig(model_path=model_path))
        if not self._asr_transcriber.status.available:
            QMessageBox.warning(
                self, "台词模型不可用", self._asr_transcriber.status.reason or ""
            )
            return
        clips = [
            (group.uuid, *self.project.group_interval(group.uuid))
            for group in self.project.groups
        ]
        self.asr_status_label.setText("准备识别…")
        self.asr_cancel_button.setEnabled(True)
        self._set_busy("asr")
        self._progress(0.0, "正在本地识别台词…")
        self._asr_worker = AsrWorker(
            self._asr_transcriber,
            self.video_path,
            clips,
            on_progress=lambda index, total, clip_id: self._events.put(
                ("asr_progress", self._task_serial, index, total)
            ),
            on_clip_result=lambda clip_id, result: self._events.put(
                ("asr_clip", self._task_serial, clip_id, result)
            ),
            on_result=lambda results: self._events.put(
                ("asr_result", self._task_serial, results)
            ),
            on_error=lambda error: self._events.put(
                ("asr_error", self._task_serial, error)
            ),
        )
        self._asr_worker.start()

    def cancel_transcription(self) -> None:
        if self._asr_worker is not None:
            self._asr_worker.cancel()
            self.asr_cancel_button.setEnabled(False)
            self.statusBar().showMessage("正在取消台词识别…")

    def _apply_transcript(self, clip_id: str, result: TranscriptResult) -> None:
        self._transcript_results[clip_id] = result
        self._transcripts[clip_id] = result.text.strip()
        self._mark_dirty()
        self._update_item_transcript(clip_id)

    def _update_item_transcript(self, group_id: str) -> None:
        if self.project is None:
            return
        self.timeline.set_transcript(
            group_id, self._transcripts.get(group_id, "").strip()
        )

    def save_transcript(self) -> None:
        group_id = self.timeline.current_id()
        if group_id is None or self.project is None:
            return
        self._transcripts[group_id] = self.transcript_edit.toPlainText().strip()
        self._mark_dirty()
        self._update_item_transcript(group_id)
        self.statusBar().showMessage("台词修改已保存到当前镜头的清单数据")

    def undo(self) -> None:
        if self.project is None or not self._history or self._busy_kind:
            return
        entry = self._history.pop()
        project_snapshot = entry
        transcripts = None
        casting = None
        if isinstance(entry, tuple):
            project_snapshot, transcripts = entry[0], entry[1]
            if len(entry) > 2:
                casting = entry[2]
        self.project.restore(project_snapshot)
        if transcripts is not None:
            self._transcripts = transcripts
        if casting is not None:
            self.casting.restore(casting)
        self._mark_dirty()
        self._refresh_project(select_first=True)
        self.statusBar().showMessage("已撤销上一步镜头编辑")

    def _refresh_transitions(self) -> None:
        previous = self.transition_combo.currentData()
        self.transition_combo.blockSignals(True)
        self.transition_combo.clear()
        if self.project is not None:
            shots = {shot.uuid: shot for shot in self.project.source_shots}
            for index, transition in enumerate(self.project.transitions, start=1):
                left = shots[transition.before_uuid].ordinal
                right = shots[transition.after_uuid].ordinal
                self.transition_combo.addItem(
                    f"转场 {index} · 原始 {left:03d} → {right:03d} · "
                    f"{transition.cut_pts:.6f}s · {transition.kind} {transition.confidence:.0%}",
                    transition.before_uuid,
                )
            restored = self.transition_combo.findData(previous)
            self.transition_combo.setCurrentIndex(restored if restored >= 0 else 0)
        self.transition_combo.blockSignals(False)
        self._transition_changed()

    def _transition_changed(self) -> None:
        if self.project is None or self._refreshing and self.transition_combo.count() == 0:
            self.cut_spin.setEnabled(False)
            self.apply_cut_button.setEnabled(False)
            return
        source_id = self.transition_combo.currentData()
        transition = next(
            (item for item in self.project.transitions if item.before_uuid == source_id), None
        )
        if transition is None:
            self.cut_spin.setEnabled(False)
            self.apply_cut_button.setEnabled(False)
            return
        shots = {shot.uuid: shot for shot in self.project.source_shots}
        left = shots[transition.before_uuid]
        right = shots[transition.after_uuid]
        epsilon = min(0.001, max(0.000001, (right.end_pts - left.start_pts) / 1000))
        self.cut_spin.setRange(left.start_pts + epsilon, right.end_pts - epsilon)
        self.cut_spin.setValue(transition.cut_pts)
        self.cut_spin.setEnabled(not self._busy_kind)
        self.apply_cut_button.setEnabled(not self._busy_kind)

    def apply_boundary(self) -> None:
        if self.project is None:
            return
        source_id = self.transition_combo.currentData()
        if source_id is None:
            return
        self._mutate_project(lambda: self.project.adjust_boundary(source_id, self.cut_spin.value()))
        self.statusBar().showMessage("已同步调整前后镜头的共同切点")

    def _update_action_states(self) -> None:
        busy = bool(self._busy_kind)
        has_video = bool(self.video_path)
        has_project = self.project is not None
        selected = self._selected_group_ids()
        can_split = False
        if has_project:
            can_split = any(
                len(group.source_ids) > 1
                for group in self.project.groups
                if group.uuid in selected
            )
        self.open_button.setEnabled(True)
        self.analyze_button.setEnabled(has_video and not busy)
        self.engine_combo.setEnabled(not busy)
        self.threshold_spin.setEnabled(not busy)
        self.min_shot_spin.setEnabled(not busy)
        self.model_button.setEnabled(not busy)
        self.clear_model_button.setEnabled(not busy and self._model_status.configured)
        self.export_button.setEnabled(has_video and has_project and not busy)
        self.merge_button.setEnabled(
            has_project and self._selection_is_contiguous(selected) and not busy
        )
        self.split_button.setEnabled(can_split and not busy)
        self.undo_button.setEnabled(bool(self._history) and not busy)
        self.play_button.setEnabled(has_video)
        self.play_shot_button.setEnabled(has_project and self.timeline.current_id() is not None)
        self.timeline.setEnabled(has_project and not busy)
        self._update_selection_label(selected)
        for control in self._numbering_controls:
            control.setEnabled(has_project and not busy)
        self.transition_combo.setEnabled(has_project and self.transition_combo.count() > 0 and not busy)
        self.cut_spin.setEnabled(self.transition_combo.isEnabled())
        self.apply_cut_button.setEnabled(self.transition_combo.isEnabled())
        self.mark_button.setEnabled(has_video and has_project and not busy)
        self.asr_button.setEnabled(has_video and has_project and not busy)
        self.save_transcript_button.setEnabled(has_project and not busy)
        self.transcript_edit.setReadOnly(busy)
        person_usable = has_video and has_project and not busy and self._person_ready()
        self.person_mark_button.setEnabled(person_usable)
        self.person_mark_selected_button.setEnabled(person_usable)
        shot_ready = has_project and not busy and self.timeline.current_id() is not None
        self.actor_count_spin.setEnabled(shot_ready)
        self.actor_date_edit.setEnabled(shot_ready)
        self.slot_scroll.setEnabled(shot_ready)
        self._sync_slider_marks()

    def _selection_is_contiguous(self, selected: list[str]) -> bool:
        if self.project is None or len(selected) < 2:
            return False
        positions = {group.uuid: index for index, group in enumerate(self.project.groups)}
        indices = sorted(positions[group_id] for group_id in selected if group_id in positions)
        if len(indices) != len(selected):
            return False
        return indices == list(range(indices[0], indices[-1] + 1))

    def _update_selection_label(self, selected: list[str]) -> None:
        if self.project is None or not selected:
            self.selection_label.setText("未选择镜头 · 单击卡片预览，Ctrl / Shift 可多选")
            return
        positions = {group.uuid: index for index, group in enumerate(self.project.groups)}
        indices = sorted(positions[group_id] for group_id in selected if group_id in positions)
        total = sum(
            end - start
            for start, end in (self.project.group_interval(group_id) for group_id in selected)
        )
        if len(selected) == 1:
            index = indices[0] if indices else -1
            label = self._group_labels[index] if 0 <= index < len(self._group_labels) else ""
            self.selection_label.setText(f"已选 {label} · {total:.2f} 秒 · Ctrl 单击可加选")
            return
        suffix = "相邻，可合并" if self._selection_is_contiguous(selected) else "不相邻，无法合并"
        self.selection_label.setText(
            f"已选 {len(selected)} 个镜头 · 合计 {total:.2f} 秒 · {suffix}"
        )

    def _timeline_selection_changed(self) -> None:
        current = self.timeline.current_id()
        if self.project is not None and current is not None:
            self.transcript_edit.setPlainText(self._transcripts.get(current, ""))
        self._update_action_states()
        self._refresh_actor_panel()
        self._update_slider_scope()

    def _preview_group(self, group_uuid: str) -> None:
        if self.project is None or not self.video_path:
            return
        start, end = self.project.group_interval(group_uuid)
        self._preview_end = end
        index = next(
            (
                index
                for index, group in enumerate(self.project.groups)
                if group.uuid == group_uuid
            ),
            None,
        )
        label = self._group_labels[index] if index is not None and index < len(self._group_labels) else None
        self._preview_label = label
        self.transcript_edit.setPlainText(self._transcripts.get(group_uuid, ""))
        self.player.setPosition(round(start * 1000))
        self._update_badge(label)
        self.player.play()
        self._preview_timer.start()

    def _check_preview_end(self) -> None:
        if self._preview_end is None:
            self._preview_timer.stop()
            return
        end_ms = round(self._preview_end * 1000)
        if self.player.playbackState() == QMediaPlayer.StoppedState:
            self._preview_timer.stop()
            self._preview_end = None
            return
        if self.player.position() >= end_ms - 30:
            self._finish_preview()

    def _finish_preview(self) -> None:
        label = self._preview_label
        end_ms = round((self._preview_end or 0.0) * 1000)
        self._preview_timer.stop()
        self._preview_end = None
        self.player.pause()
        if end_ms > 0:
            self.player.setPosition(max(0, end_ms - 40))
        self._update_badge(label)

    def play_selected_group(self) -> None:
        group_id = self.timeline.current_id()
        if group_id is not None:
            self._preview_group(group_id)

    def toggle_playback(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
            self._preview_timer.stop()
            return
        group_id = self.timeline.current_id()
        if self.project is not None and group_id is not None:
            start, end = self.project.group_interval(group_id)
            position = self.player.position() / 1000
            if position < start or position >= end - 0.001:
                self._preview_group(group_id)
                return
            index = next(
                (
                    index
                    for index, group in enumerate(self.project.groups)
                    if group.uuid == group_id
                ),
                None,
            )
            self._preview_end = end
            self._preview_label = (
                self._group_labels[index]
                if index is not None and index < len(self._group_labels)
                else None
            )
            self._update_badge(self._preview_label)
            self.player.play()
            self._preview_timer.start()
            return
        self._preview_end = None
        self._preview_label = None
        self.player.play()

    def _position_changed(self, position: int) -> None:
        if self._preview_end is not None and position >= round(self._preview_end * 1000) - 20:
            self._finish_preview()
            return
        self.position_slider.blockSignals(True)
        self.position_slider.setValue(position)
        self.position_slider.blockSignals(False)
        scope = self._slider_scope()
        if scope is not None:
            start, end = scope
            inside = min(max(position / 1000 - start, 0.0), end - start)
            self.position_label.setText(
                f"片段内 {_time_text(inside)} / {_time_text(end - start)}"
            )
        else:
            self.position_label.setText(
                f"{_time_text(position / 1000)} / {_time_text(self.player.duration() / 1000)}"
            )
        self._update_badge()

    def _badge_label_at(self, seconds: float) -> str | None:
        if not self._group_labels or not self._group_ranges:
            return None
        for label, (start, end) in zip(self._group_labels, self._group_ranges):
            if start <= seconds < end:
                return label
        return self._group_labels[-1] if seconds >= self._group_ranges[-1][1] else None

    def _update_badge(self, label: str | None = None) -> None:
        text = label if label is not None else self._badge_label_at(self.player.position() / 1000)
        if not text:
            self.shot_badge.setVisible(False)
            self.shot_badge_backdrop.setVisible(False)
            return
        self.shot_badge.setPlainText(text)
        self.shot_badge.setVisible(True)
        self.shot_badge_backdrop.setVisible(True)
        self._place_badge()

    def _place_badge(self) -> None:
        if not self.shot_badge.isVisible():
            return
        video_rect = self.video_item.boundingRect()
        if video_rect.isEmpty():
            video_rect = self.video_scene.sceneRect()
        if video_rect.isEmpty():
            return
        text_rect = self.shot_badge.boundingRect()
        padding = max(4.0, text_rect.height() * 0.22)
        margin_x = video_rect.width() * 0.035
        margin_y = video_rect.height() * 0.045
        key = self.badge_position_combo.currentData()
        # 位置键为 "top|middle|bottom - left|center|right"，第一段是纵向、第二段是横向。
        vertical, _, horizontal = str(key).partition("-")
        if horizontal == "left":
            x = video_rect.left() + margin_x
        elif horizontal == "center":
            x = video_rect.center().x() - text_rect.width() / 2
        else:
            x = video_rect.right() - margin_x - text_rect.width()
        if vertical == "top":
            y = video_rect.top() + margin_y
        elif vertical == "middle":
            y = video_rect.center().y() - text_rect.height() / 2
        else:
            y = video_rect.bottom() - margin_y - text_rect.height()
        self.shot_badge.setPos(x, y)
        self.shot_badge_backdrop.setRect(
            x - padding,
            y - padding / 2,
            text_rect.width() + padding * 2,
            text_rect.height() + padding,
        )

    def _badge_font_changed(self, points: int) -> None:
        font = self.shot_badge.font()
        font.setPointSize(points)
        font.setBold(True)
        self.shot_badge.setFont(font)
        self._place_badge()

    def _video_size_changed(self, _size) -> None:
        rect = self.video_item.boundingRect()
        if rect.isEmpty():
            return
        self.video_scene.setSceneRect(rect)
        self.video_view.fitInView(rect, Qt.KeepAspectRatio)
        self._place_badge()

    def eventFilter(self, watched, event) -> bool:
        if watched is self.video_view.viewport() and event.type() == QEvent.Resize:
            if not self.video_item.boundingRect().isEmpty():
                self.video_view.fitInView(self.video_item.boundingRect(), Qt.KeepAspectRatio)
            self._place_badge()
        return super().eventFilter(watched, event)

    def _duration_changed(self, duration: int) -> None:
        self._update_slider_scope()

    def _playback_state_changed(self, state) -> None:
        playing = state == QMediaPlayer.PlayingState
        self.play_button.setText("暂停" if playing else "播放")
        self.play_button.setIcon(
            self.style().standardIcon(QStyle.SP_MediaPause if playing else QStyle.SP_MediaPlay)
        )

    def _player_error(self, _error, description: str = "") -> None:
        if description:
            self.statusBar().showMessage(f"播放器提示：{description}", 10000)

    def start_export(self) -> None:
        if self._busy_kind or self.project is None or not self.video_path:
            return
        default = self._last_output_directory or str(
            Path(self.video_path).parent / f"{Path(self.video_path).stem}_lensplit"
        )
        dialog = ExportDialog(self, default)
        if dialog.exec() != QDialog.Accepted:
            return
        options = dialog.export_options(
            self.numbering_style(),
            self.merge_mode(),
            dict(self._analysis_metadata),
            dict(self._transcripts),
        )
        self._last_output_directory = options.output_dir
        self._task_serial += 1
        token = self._task_serial
        self._task_started = time.monotonic()
        project_copy = ShotProject.from_snapshot(self.project.snapshot())
        worker = ExportWorker(
            FFmpegExporter(self.video_path),
            project_copy,
            options,
            on_progress=lambda value: self._events.put(("export_progress", token, value)),
            on_result=lambda result: self._events.put(("export_result", token, result)),
            on_error=lambda error: self._events.put(("export_error", token, error)),
            on_warning=lambda text: self._events.put(("export_warning", token, text)),
        )
        self._export_worker = worker
        self._set_busy("export")
        self._progress(0.0, "正在本地导出镜头…")
        try:
            worker.start()
        except Exception:
            self._export_worker = None
            self._set_busy(None)
            raise

    def _export_finished(self, report: ExportReport) -> None:
        self.statusBar().showMessage(
            f"导出完成：{len(report.output_files)} 个镜头 · {report.actual_encoder} · {report.elapsed:.2f} 秒"
        )
        QMessageBox.information(
            self,
            "导出完成",
            f"已导出 {len(report.output_files)} 个镜头。\n"
            f"实际编码器：{report.actual_encoder}\n"
            f"耗时：{report.elapsed:.2f} 秒\n\n"
            f"JSON 清单：{report.manifest_json_path}\n"
            f"CSV 清单：{report.manifest_csv_path}",
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._busy_kind:
            answer = QMessageBox.question(
                self,
                "任务仍在运行",
                "关闭窗口将取消当前本地任务。确定退出吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            if self._analysis_worker is not None:
                self._analysis_worker.cancel()
            if self._export_worker is not None:
                self._export_worker.cancel()
        elif not self._confirm_discard():
            event.ignore()
            return
        self.player.stop()
        self.player.setSource(QUrl())
        self._event_timer.stop()
        self._thumbnail_executor.shutdown(wait=False, cancel_futures=True)
        event.accept()


__all__ = ["ExportDialog", "MainWindow"]
