"""Shooting table page: one row per logical shot, call-sheet style."""

from __future__ import annotations

import csv
from pathlib import Path

from PySide6.QtCore import QDate, QSignalBlocker, QSize, QStandardPaths, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPalette, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lensplit.core.casting import (
    ACTOR_CSV_HEADER,
    ACTOR_CSV_RULES,
    Actor,
    CastingConflict,
    CastingData,
    parse_actor_csv,
)
from lensplit.core.model import ShotProject

_UNSET_DATE = QDate(2000, 1, 1)


def style_dark_combo(combo: QComboBox) -> None:
    """Force the dark palette on a combo and its popup list.

    Windows renders the popup as a separate window whose palette is not always
    inherited from the stylesheet, which used to leave light text on white.
    """

    palette = combo.palette()
    for role, color in (
        (QPalette.Base, QColor("#1b2230")),
        (QPalette.Window, QColor("#1b2230")),
        (QPalette.Text, QColor("#edf2fa")),
        (QPalette.WindowText, QColor("#edf2fa")),
        (QPalette.ButtonText, QColor("#edf2fa")),
        (QPalette.Highlight, QColor("#3979dc")),
        (QPalette.HighlightedText, QColor("#ffffff")),
    ):
        palette.setColor(role, color)
    combo.setPalette(palette)
    view = combo.view()
    view.setPalette(palette)
    view.setStyleSheet(
        "QAbstractItemView { background: #1b2230; color: #edf2fa;"
        " selection-background-color: #3979dc; selection-color: #ffffff;"
        " outline: 0; }"
    )


def _default_save_dir() -> str:
    """A folder the current user can always write to."""

    for location in (
        QStandardPaths.DocumentsLocation,
        QStandardPaths.DesktopLocation,
        QStandardPaths.HomeLocation,
    ):
        folder = QStandardPaths.writableLocation(location)
        if folder and Path(folder).is_dir():
            return folder
    return str(Path.home())


def export_actor_template(parent: QWidget | None) -> bool:
    """Save a CSV template that matches the import rules."""

    suggested = str(Path(_default_save_dir()) / "演员导入模板.csv")
    path, _ = QFileDialog.getSaveFileName(
        parent, "保存演员导入模板", suggested, "CSV 文件 (*.csv)"
    )
    if not path:
        return False
    sample = "张三,2026-11-01,2026-11-05,主角 · 会骑摩托\n李四,,,配角 · 需化老年妆\n"
    try:
        Path(path).write_text(ACTOR_CSV_HEADER + "\n" + sample, encoding="utf-8-sig")
    except OSError as error:
        QMessageBox.warning(
            parent,
            "无法保存模板",
            f"写入失败：{error}\n\n请换一个你有权限的目录，例如桌面或文档文件夹。",
        )
        return False
    return True


def import_actors_from_csv(parent: QWidget | None, casting: CastingData) -> int:
    """Pick a CSV, import actors into the library, report the outcome."""

    path, _ = QFileDialog.getOpenFileName(
        parent, "批量导入演员", "", "CSV 文件 (*.csv);;所有文件 (*)"
    )
    if not path:
        return 0
    raw = Path(path).read_bytes()
    text = ""
    for codec in ("utf-8-sig", "gbk"):
        try:
            text = raw.decode(codec)
            break
        except UnicodeDecodeError:
            continue
    if not text:
        QMessageBox.warning(parent, "无法读取", "文件编码不是 UTF-8 或 GBK，请另存后再试。")
        return 0
    try:
        rows, errors = parse_actor_csv(text)
    except ValueError as error:
        QMessageBox.warning(
            parent, "CSV 格式不对", f"{error}\n\n导入规则：\n{ACTOR_CSV_RULES}"
        )
        return 0
    result = casting.import_actors(rows)
    notes = list(errors) + list(result.errors)
    detail = "\n".join(notes) if notes else "全部导入成功。"
    QMessageBox.information(
        parent,
        "批量导入完成",
        f"{result.summary()}\n\n{detail}\n\n导入规则：\n{ACTOR_CSV_RULES}",
    )
    return result.total


def _timecode(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    minutes, remainder = divmod(milliseconds, 60_000)
    secs, millis = divmod(remainder, 1000)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def _qdate(value: str) -> QDate:
    parsed = QDate.fromString(value, "yyyy-MM-dd")
    return parsed if parsed.isValid() else _UNSET_DATE


def _centered(widget: QWidget, vertical: int = 18) -> QWidget:
    holder = QWidget()
    layout = QHBoxLayout(holder)
    layout.setContentsMargins(4, vertical, 4, vertical)
    layout.addWidget(widget)
    return holder


class RoleAssignDialog(QDialog):
    """Name the shot's role slots and cast each one from the actor library."""

    def __init__(
        self,
        parent: QWidget | None,
        casting: CastingData,
        group_uuid: str,
        shot_label: str,
    ) -> None:
        super().__init__(parent)
        self.casting = casting
        self.group_uuid = group_uuid
        self._loading = False
        self._rows: list[tuple[str, QLineEdit, QComboBox]] = []
        self.setWindowTitle(f"角色与演员 · {shot_label or '当前镜头'}")
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)
        hint = QLabel(
            "角色槽由镜头人数决定（人数 2 → 角色 1 / 角色 2）。"
            "可直接改名，并从演员库挑人；档期取「演员与档期」页的登记值。"
        )
        hint.setObjectName("mutedLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["角色", "演员", "档期与备注"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setColumnWidth(0, 130)
        self.table.setColumnWidth(1, 170)
        self.table.setMinimumHeight(180)
        layout.addWidget(self.table, 1)

        self.count_label = QLabel()
        self.count_label.setObjectName("mutedLabel")
        layout.addWidget(self.count_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("确定")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._fill()

    def _fill(self) -> None:
        self._loading = True
        self._rows.clear()
        slots = self.casting.shot_slots(self.group_uuid)
        self.table.setRowCount(len(slots))
        for row, role in enumerate(slots):
            name_edit = QLineEdit(role.name)
            name_edit.setMaxLength(24)
            self.table.setCellWidget(row, 0, name_edit)
            combo = QComboBox()
            combo.addItem("未分配", "")
            for actor in self.casting.actors:
                combo.addItem(actor.name, actor.uuid)
            combo.setCurrentIndex(max(0, combo.findData(role.actor_uuid)))
            style_dark_combo(combo)
            self.table.setCellWidget(row, 1, combo)
            info = QLabel(self._brief(self.casting.actor(role.actor_uuid)))
            info.setObjectName("mutedLabel")
            self.table.setCellWidget(row, 2, info)
            self._rows.append((role.uuid, name_edit, combo))
        self.table.resizeRowsToContents()
        self.count_label.setText(
            f"本镜头 {len(slots)} 个角色槽"
            if slots
            else "本镜头还没有角色槽：请先在「剪辑」页标注人数（人物自动标记或手动填写）"
        )
        self._loading = False

    @staticmethod
    def _brief(actor: Actor | None) -> str:
        if actor is None:
            return "尚未分配演员"
        text = actor.window_text()
        return f"{text} · {actor.notes}" if actor.notes else text

    def accept(self) -> None:
        for role_uuid, name_edit, combo in self._rows:
            self.casting.rename_role(role_uuid, name_edit.text())
            self.casting.assign_actor(role_uuid, combo.currentData() or "")
        super().accept()


class CastingLibraryDialog(QDialog):
    """Maintain the actor library (with availability) and the role library."""

    def __init__(self, parent: QWidget | None, casting: CastingData) -> None:
        super().__init__(parent)
        self.casting = casting
        self._loading = False
        self.setWindowTitle("角色与演员")
        self.resize(880, 640)

        layout = QVBoxLayout(self)
        hint = QLabel(
            "演员档期是闭区间；留空表示不限。所有修改即时生效，并会同步到拍摄表。"
        )
        hint.setObjectName("mutedLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        actor_group = QGroupBox("演员档期")
        actor_layout = QVBoxLayout(actor_group)
        self.actor_table = QTableWidget(0, 4)
        self.actor_table.setHorizontalHeaderLabels(
            ["姓名", "可拍摄开始", "可拍摄结束", "备注"]
        )
        self._prepare_table(self.actor_table)
        header = self.actor_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        self.actor_table.setColumnWidth(0, 170)
        self.actor_table.setColumnWidth(1, 150)
        self.actor_table.setColumnWidth(2, 150)
        self.actor_table.itemChanged.connect(self._actor_text_changed)
        actor_layout.addWidget(self.actor_table, 1)
        actor_row = QHBoxLayout()
        add_actor = QPushButton("添加演员…")
        add_actor.clicked.connect(self._add_actor)
        actor_row.addWidget(add_actor)
        remove_actor = QPushButton("删除选中演员")
        remove_actor.clicked.connect(self._remove_actor)
        actor_row.addWidget(remove_actor)
        import_actor = QPushButton("批量导入演员…")
        import_actor.setToolTip(f"CSV 格式：{ACTOR_CSV_HEADER}")
        import_actor.clicked.connect(self._import_actors)
        actor_row.addWidget(import_actor)
        template_actor = QPushButton("下载模板")
        template_actor.clicked.connect(lambda: export_actor_template(self))
        actor_row.addWidget(template_actor)
        actor_row.addStretch(1)
        actor_layout.addLayout(actor_row)
        layout.addWidget(actor_group, 1)

        role_group = QGroupBox("角色与饰演演员")
        role_layout = QVBoxLayout(role_group)
        self.role_table = QTableWidget(0, 3)
        self.role_table.setHorizontalHeaderLabels(["角色", "饰演演员", "性格与表演要求"])
        self._prepare_table(self.role_table)
        role_header = self.role_table.horizontalHeader()
        role_header.setSectionResizeMode(0, QHeaderView.Fixed)
        role_header.setSectionResizeMode(1, QHeaderView.Fixed)
        role_header.setSectionResizeMode(2, QHeaderView.Stretch)
        self.role_table.setColumnWidth(0, 200)
        self.role_table.setColumnWidth(1, 220)
        self.role_table.itemChanged.connect(self._role_text_changed)
        role_layout.addWidget(self.role_table, 1)
        role_row = QHBoxLayout()
        add_role = QPushButton("添加角色…")
        add_role.clicked.connect(self._add_role)
        role_row.addWidget(add_role)
        remove_role = QPushButton("删除选中角色")
        remove_role.clicked.connect(self._remove_role)
        role_row.addWidget(remove_role)
        role_row.addStretch(1)
        role_layout.addLayout(role_row)
        layout.addWidget(role_group, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.button(QDialogButtonBox.Close).setText("关闭")
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
        for combo in self.findChildren(QComboBox):
            style_dark_combo(combo)
        self._fill()

    @staticmethod
    def _prepare_table(table: QTableWidget) -> None:
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(38)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setShowGrid(False)

    # -- rendering -------------------------------------------------------
    def _fill(self) -> None:
        self._loading = True
        self.actor_table.setRowCount(0)
        for actor in self.casting.actors:
            row = self.actor_table.rowCount()
            self.actor_table.insertRow(row)
            name_item = QTableWidgetItem(actor.name)
            name_item.setData(Qt.UserRole, actor.uuid)
            self.actor_table.setItem(row, 0, name_item)
            for column, value, field in (
                (1, actor.available_from, "start"),
                (2, actor.available_to, "end"),
            ):
                editor = QDateEdit()
                editor.setDisplayFormat("yyyy-MM-dd")
                editor.setCalendarPopup(True)
                editor.setSpecialValueText("不限")
                editor.setMinimumDate(_UNSET_DATE)
                editor.setMaximumDate(QDate(2100, 12, 31))
                editor.setDate(_qdate(value))
                editor.dateChanged.connect(
                    lambda qdate,
                    uuid=actor.uuid,
                    which=field,
                    widget=editor: self._actor_date_changed(uuid, which, qdate, widget)
                )
                self.actor_table.setCellWidget(row, column, _centered(editor, 4))
            self.actor_table.setItem(row, 3, QTableWidgetItem(actor.notes))
        self.role_table.setRowCount(0)
        for role in self.casting.roles:
            if role.is_slot:
                # Per-shot slots are edited on the shot itself, not in the library.
                continue
            row = self.role_table.rowCount()
            self.role_table.insertRow(row)
            name_item = QTableWidgetItem(role.name)
            name_item.setData(Qt.UserRole, role.uuid)
            self.role_table.setItem(row, 0, name_item)
            combo = QComboBox()
            combo.addItem("未分配演员", "")
            for actor in self.casting.actors:
                combo.addItem(f"{actor.name}（{actor.window_text()}）", actor.uuid)
            index = combo.findData(role.actor_uuid)
            combo.setCurrentIndex(max(0, index))
            style_dark_combo(combo)
            combo.currentIndexChanged.connect(
                lambda _index, uuid=role.uuid, widget=combo: self._role_actor_changed(
                    uuid, widget.currentData()
                )
            )
            self.role_table.setCellWidget(row, 1, _centered(combo, 4))
            self.role_table.setItem(row, 2, QTableWidgetItem(role.persona))
        self._loading = False

    # -- actor edits -----------------------------------------------------
    def _actor(self, actor_uuid: str) -> Actor | None:
        return self.casting.actor(actor_uuid)

    def _actor_text_changed(self, item: QTableWidgetItem) -> None:
        if self._loading:
            return
        actor = self._actor(item.data(Qt.UserRole) or "")
        if actor is None or item.column() not in (0, 3):
            return
        if item.column() == 0:
            try:
                actor.set_name(item.text())
            except ValueError as error:
                QMessageBox.warning(self, "姓名不可用", str(error))
                with QSignalBlocker(self.actor_table):
                    item.setText(actor.name)
                return
        else:
            actor.notes = item.text()

    def _actor_date_changed(
        self, actor_uuid: str, field: str, qdate: QDate, widget: QDateEdit
    ) -> None:
        if self._loading:
            return
        actor = self._actor(actor_uuid)
        if actor is None:
            return
        value = "" if qdate == _UNSET_DATE else qdate.toString("yyyy-MM-dd")
        start = value if field == "start" else actor.available_from
        end = value if field == "end" else actor.available_to
        try:
            actor.set_availability(start, end)
        except ValueError as error:
            QMessageBox.warning(self, "档期不合法", str(error))
            restored = actor.available_from if field == "start" else actor.available_to
            with QSignalBlocker(widget):
                widget.setDate(_qdate(restored))

    def _add_actor(self) -> None:
        name, accepted = QInputDialog.getText(
            self, "添加演员", "演员姓名：", text=f"演员{len(self.casting.actors) + 1}"
        )
        if not accepted or not name.strip():
            return
        try:
            self.casting.add_actor(name)
        except ValueError as error:
            QMessageBox.warning(self, "无法添加演员", str(error))
            return
        self._fill()

    def _remove_actor(self) -> None:
        row = self.actor_table.currentRow()
        if row < 0:
            return
        item = self.actor_table.item(row, 0)
        if item is None:
            return
        actor = self._actor(item.data(Qt.UserRole) or "")
        if actor is None:
            return
        answer = QMessageBox.question(
            self,
            "删除演员",
            f"删除演员「{actor.name}」？其角色会变为未分配演员。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self.casting.remove_actor(actor.uuid)
        self._fill()

    def _import_actors(self) -> None:
        if import_actors_from_csv(self, self.casting):
            self._fill()

    # -- role edits ------------------------------------------------------
    def _role_actor_changed(self, role_uuid: str, actor_uuid: str) -> None:
        if self._loading:
            return
        try:
            self.casting.assign_actor(role_uuid, actor_uuid or "")
        except ValueError as error:
            QMessageBox.warning(self, "无法分配演员", str(error))

    def _role_text_changed(self, item: QTableWidgetItem) -> None:
        if self._loading:
            return
        role = self.casting.role(item.data(Qt.UserRole) or "")
        if role is None:
            return
        if item.column() == 0:
            text = item.text().strip()
            if not text:
                QMessageBox.warning(self, "角色名不可用", "角色名称不能为空")
                with QSignalBlocker(self.role_table):
                    item.setText(role.name)
                return
            role.name = text
        elif item.column() == 2:
            role.persona = item.text()

    def _add_role(self) -> None:
        name, accepted = QInputDialog.getText(
            self, "添加角色", "角色名称：", text=f"角色{len(self.casting.roles) + 1}"
        )
        if not accepted or not name.strip():
            return
        try:
            self.casting.add_role(name)
        except ValueError as error:
            QMessageBox.warning(self, "无法添加角色", str(error))
            return
        self._fill()

    def _remove_role(self) -> None:
        row = self.role_table.currentRow()
        if row < 0:
            return
        item = self.role_table.item(row, 0)
        if item is None:
            return
        role = self.casting.role(item.data(Qt.UserRole) or "")
        if role is None:
            return
        answer = QMessageBox.question(
            self,
            "删除角色",
            f"删除角色「{role.name}」？该角色会从所有镜头移除。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self.casting.remove_role(role.uuid)
        self._fill()


class ShootingTablePage(QWidget):
    """Call-sheet style list of shots with casting, schedule, and dialogue."""

    preview_requested = Signal(str)
    transcript_changed = Signal(str, str)
    data_changed = Signal()
    schedule_changed = Signal(str)

    COLUMNS = (
        "镜号",
        "画面",
        "入点",
        "出点",
        "时长",
        "人数",
        "拍摄日期",
        "角色分配",
        "演员档期",
        "台词",
    )
    WIDTHS = {0: 82, 1: 138, 2: 98, 3: 98, 4: 66, 5: 86, 6: 126, 7: 190, 8: 162}
    ROW_HEIGHT = 78

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project: ShotProject | None = None
        self.casting = CastingData()
        self._labels: dict[str, str] = {}
        self._transcripts: dict[str, str] = {}
        self._thumbnails: dict[str, QPixmap] = {}
        self._rows: list[str] = []
        self._row_index: dict[str, int] = {}
        self._shot_issues: dict[str, list[CastingConflict]] = {}
        self._issues: list[CastingConflict] = []
        self._dirty = True
        self._populating = False
        self._export_path = str(Path.home() / "拍摄表.csv")
        self._build_ui()
        for combo in self.findChildren(QComboBox):
            style_dark_combo(combo)

    # -- construction ----------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 8)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(8)
        title = QLabel("拍摄表 · 每个逻辑镜头一行")
        title.setObjectName("statsLabel")
        header.addWidget(title)
        hint = QLabel("单击「画面」回到剪辑页播放该镜头；角色与档期改动即时写入镜头数据。")
        hint.setObjectName("mutedLabel")
        header.addWidget(hint, 1)
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("全部镜头", "all")
        self.filter_combo.addItem("仅看撞档", "conflict")
        self.filter_combo.addItem("未分配角色", "unassigned")
        self.filter_combo.currentIndexChanged.connect(self._filters_changed)
        header.addWidget(self.filter_combo)
        self.actor_filter = QComboBox()
        self.actor_filter.addItem("全部演员", "")
        self.actor_filter.currentIndexChanged.connect(self._filters_changed)
        header.addWidget(self.actor_filter)
        self.library_button = QPushButton("角色与演员…")
        self.library_button.clicked.connect(self._open_library)
        header.addWidget(self.library_button)
        self.export_button = QPushButton("导出拍摄表 CSV…")
        self.export_button.clicked.connect(self.export_csv)
        header.addWidget(self.export_button)
        layout.addLayout(header)

        summary_row = QHBoxLayout()
        self.summary_label = QLabel("尚未导入视频")
        self.summary_label.setObjectName("statsLabel")
        summary_row.addWidget(self.summary_label)
        summary_row.addStretch(1)
        layout.addLayout(summary_row)

        self.issue_label = QLabel()
        self.issue_label.setObjectName("conflictLabel")
        self.issue_label.setWordWrap(True)
        self.issue_label.setVisible(False)
        layout.addWidget(self.issue_label)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(list(self.COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(self.ROW_HEIGHT)
        self.table.setShowGrid(False)
        self.table.setWordWrap(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        table_header = self.table.horizontalHeader()
        table_header.setHighlightSections(False)
        table_header.setStretchLastSection(True)
        for column, width in self.WIDTHS.items():
            table_header.setSectionResizeMode(column, QHeaderView.Fixed)
            self.table.setColumnWidth(column, width)
        table_header.setSectionResizeMode(9, QHeaderView.Stretch)
        layout.addWidget(self.table, 1)

    # -- context ---------------------------------------------------------
    def set_context(
        self,
        project: ShotProject | None,
        casting: CastingData,
        labels: dict[str, str],
        transcripts: dict[str, str],
    ) -> None:
        self.project = project
        self.casting = casting
        self._labels = dict(labels)
        self._transcripts = transcripts
        self._dirty = True
        self._refresh_actor_filter()
        if self.isVisible():
            self.rebuild()

    def set_export_directory(self, path: str) -> None:
        self._export_path = path

    def clear(self) -> None:
        self.project = None
        self._labels = {}
        self._transcripts = {}
        self._thumbnails.clear()
        self._dirty = True
        if self.isVisible():
            self.rebuild()

    def clear_thumbnails(self) -> None:
        self._thumbnails.clear()

    def set_thumbnail(self, group_uuid: str, pixmap: QPixmap) -> None:
        self._thumbnails[group_uuid] = pixmap
        if self._dirty or not self.isVisible():
            return
        row = self._row_index.get(group_uuid)
        if row is None:
            return
        button = self.table.cellWidget(row, 1)
        if isinstance(button, QPushButton):
            self._apply_thumbnail(button, group_uuid)

    def is_dirty(self) -> bool:
        return self._dirty

    def activate(self) -> None:
        if self._dirty:
            self.rebuild()

    # -- rebuilding ------------------------------------------------------
    def rebuild(self) -> None:
        self._dirty = False
        self._issues = self.casting.conflicts(self._labels)
        self._shot_issues = {}
        for issue in self._issues:
            if issue.group_uuid:
                self._shot_issues.setdefault(issue.group_uuid, []).append(issue)

        self._populating = True
        self.table.setUpdatesEnabled(False)
        self.table.setRowCount(0)
        self._rows = []
        self._row_index = {}
        groups = self.project.groups if self.project is not None else []
        for group in groups:
            if not self._passes_filters(group.uuid):
                continue
            self._append_row(group.uuid)
        self.table.setUpdatesEnabled(True)
        self._populating = False
        self._update_summary()

    def _passes_filters(self, group_uuid: str) -> bool:
        shot = self.casting.shot(group_uuid)
        mode = self.filter_combo.currentData()
        if mode == "conflict" and group_uuid not in self._shot_issues:
            return False
        if mode == "unassigned" and shot is not None and shot.role_ids:
            return False
        actor_uuid = self.actor_filter.currentData()
        if actor_uuid:
            assigned = {
                role.actor_uuid for role in self.casting.roles_for_shot(group_uuid)
            }
            if actor_uuid not in assigned:
                return False
        return True

    def _append_row(self, group_uuid: str) -> None:
        if self.project is None:
            return
        shot = self.casting.shot(group_uuid)
        start, end = self.project.group_interval(group_uuid)
        label = self._labels.get(group_uuid, "")
        row = self.table.rowCount()
        self.table.insertRow(row)
        self._rows.append(group_uuid)
        self._row_index[group_uuid] = row
        issues = self._shot_issues.get(group_uuid, [])
        error = any(issue.is_error for issue in issues)

        label_item = QTableWidgetItem(label)
        label_item.setForeground(QColor("#e8edf5"))
        self.table.setItem(row, 0, label_item)

        button = QPushButton()
        button.setObjectName("shotThumbButton")
        button.setIconSize(QSize(100, 56))
        button.setMinimumHeight(64)
        button.setCursor(Qt.PointingHandCursor)
        button.setToolTip(
            f"{label} · {_timecode(start)} → {_timecode(end)}\n单击回到剪辑页播放该镜头"
        )
        button.clicked.connect(
            lambda _checked=False, group_id=group_uuid: self.preview_requested.emit(
                group_id
            )
        )
        self._apply_thumbnail(button, group_uuid)
        self.table.setCellWidget(row, 1, _centered(button, 4))

        for column, text in (
            (2, _timecode(start)),
            (3, _timecode(end)),
            (4, f"{max(0.0, end - start):.2f}s"),
        ):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            item.setForeground(QColor("#c4d3e6"))
            self.table.setItem(row, column, item)

        spin = QSpinBox()
        spin.setRange(0, 99)
        spin.setSpecialValueText("未标注")
        spin.setValue(shot.character_count if shot else 0)
        spin.setToolTip("本镜头画面中的角色人数（0 表示未标注，后续可由本地人检测自动填写）")
        spin.valueChanged.connect(
            lambda value, group_id=group_uuid: self._character_count_changed(
                group_id, value
            )
        )
        self.table.setCellWidget(row, 5, _centered(spin))

        date_edit = QDateEdit()
        date_edit.setDisplayFormat("yyyy-MM-dd")
        date_edit.setCalendarPopup(True)
        date_edit.setSpecialValueText("未定")
        date_edit.setMinimumDate(_UNSET_DATE)
        date_edit.setMaximumDate(QDate(2100, 12, 31))
        date_edit.setDate(_qdate(shot.shoot_date if shot else ""))
        date_edit.setToolTip("该镜头的拍摄日期；撞档检查与演员档期都基于此日期")
        date_edit.dateChanged.connect(
            lambda qdate, group_id=group_uuid: self._shoot_date_changed(group_id, qdate)
        )
        self.table.setCellWidget(row, 6, _centered(date_edit))

        role_button = QPushButton(self._role_text(group_uuid))
        role_button.setObjectName("roleButton")
        role_button.setToolTip("单击分配角色并标注由谁饰演")
        role_button.setMinimumHeight(40)
        role_button.clicked.connect(
            lambda _checked=False, group_id=group_uuid: self._assign_roles(group_id)
        )
        self.table.setCellWidget(row, 7, _centered(role_button, 16))

        actor_label = QLabel(self._actor_text(group_uuid))
        actor_label.setWordWrap(True)
        actor_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        actor_label.setContentsMargins(6, 4, 6, 4)
        if issues:
            actor_label.setToolTip("\n".join(issue.message for issue in issues))
            actor_label.setStyleSheet(
                "color: #ff8b8b;" if error else "color: #ffc35c;"
            )
        else:
            actor_label.setStyleSheet("color: #c4d3e6;")
        self.table.setCellWidget(row, 8, actor_label)

        editor = QPlainTextEdit(self._transcripts.get(group_uuid, ""))
        editor.setObjectName("cellEdit")
        editor.setPlaceholderText("无台词")
        editor.setFixedHeight(58)
        editor.setToolTip("可直接修改；改动会同步回剪辑页的台词清单")
        editor.textChanged.connect(
            lambda group_id=group_uuid, widget=editor: self._transcript_edited(
                group_id, widget
            )
        )
        self.table.setCellWidget(row, 9, _centered(editor, 8))

        if issues:
            background = QColor("#3a2130") if error else QColor("#3a3021")
            for column in (0, 2, 3, 4):
                item = self.table.item(row, column)
                if item is not None:
                    item.setBackground(background)

    def _apply_thumbnail(self, button: QPushButton, group_uuid: str) -> None:
        pixmap = self._thumbnails.get(group_uuid)
        if pixmap is None or pixmap.isNull():
            button.setText("▶ 播放")
            button.setIcon(QIcon())
            return
        button.setText("")
        button.setIcon(QIcon(pixmap))

    def _role_text(self, group_uuid: str) -> str:
        roles = self.casting.roles_for_shot(group_uuid)
        if not roles:
            shot = self.casting.shot(group_uuid)
            if shot is not None and shot.character_count:
                return f"＋ 分配演员（{shot.character_count} 人）"
            return "＋ 标注人数"
        parts = []
        for role in roles:
            actor = self.casting.actor(role.actor_uuid)
            parts.append(f"{role.name}（{actor.name}）" if actor else f"{role.name}（未分配）")
        text = "、".join(parts)
        return text if len(text) <= 26 else text[:25] + "…"

    def _actor_text(self, group_uuid: str) -> str:
        actors = self.casting.actors_for_shot(group_uuid)
        if not actors:
            return "—"
        shot = self.casting.shot(group_uuid)
        day = shot.shoot_date if shot else ""
        lines = []
        for actor in actors:
            suffix = ""
            if day:
                suffix = " √" if actor.is_available(day) else " × 档期冲突"
            lines.append(f"{actor.name} · {actor.window_text()}{suffix}")
        return "\n".join(lines)

    def _update_summary(self) -> None:
        total = len(self.project.groups) if self.project is not None else 0
        if total == 0:
            self.summary_label.setText("尚未导入视频 · 请先在「剪辑」页导入并分析")
            self.summary_label.setStyleSheet("color: #95a3b7;")
            self.issue_label.setVisible(False)
            return
        assigned = sum(
            1
            for group in self.project.groups
            if (shot := self.casting.shot(group.uuid)) is not None and shot.role_ids
        )
        errors = sum(1 for issue in self._issues if issue.is_error)
        warnings = len(self._issues) - errors
        self.summary_label.setText(
            f"共 {total} 镜 ｜ 已分配角色 {assigned} ｜ 撞档 {errors} ｜ 待确认 {warnings}"
        )
        self.summary_label.setStyleSheet(
            "color: #ff8b8b;" if errors else "color: #a9c8f2;"
        )
        if self._issues:
            shown = "\n".join(issue.message for issue in self._issues[:3])
            if len(self._issues) > 3:
                shown += f"\n…共 {len(self._issues)} 条提示"
            self.issue_label.setText(shown)
        self.issue_label.setVisible(bool(self._issues))

    # -- edits -----------------------------------------------------------
    def _character_count_changed(self, group_uuid: str, value: int) -> None:
        if self._populating:
            return
        self.casting.set_character_count(group_uuid, value)
        self.data_changed.emit()

    def _shoot_date_changed(self, group_uuid: str, qdate: QDate) -> None:
        if self._populating:
            return
        value = "" if qdate == _UNSET_DATE else qdate.toString("yyyy-MM-dd")
        self.casting.set_shoot_date(group_uuid, value)
        self._refresh_row(group_uuid)
        self.data_changed.emit()
        self.schedule_changed.emit(group_uuid)

    def apply_shoot_date(self, group_uuid: str, value: str) -> None:
        """Mirror a date set from page 1 without emitting change signals."""

        shot = self.casting.shot(group_uuid)
        row = self._row_index.get(group_uuid)
        if row is None:
            return
        holder = self.table.cellWidget(row, 6)
        editor = holder.findChild(QDateEdit) if holder is not None else None
        if editor is not None:
            with QSignalBlocker(editor):
                editor.setDate(_qdate(shot.shoot_date if shot else value))
        self._refresh_row(group_uuid)

    def apply_character_count(self, group_uuid: str, value: int) -> None:
        """Write an auto-detected headcount without rebuilding the table."""

        self.casting.set_character_count(group_uuid, value)
        row = self._row_index.get(group_uuid)
        if row is None:
            return
        holder = self.table.cellWidget(row, 5)
        spin = holder.findChild(QSpinBox) if holder is not None else None
        if spin is not None:
            with QSignalBlocker(spin):
                spin.setValue(value)

    def _refresh_row(self, group_uuid: str) -> None:
        """Recompute conflicts after an inline edit without rebuilding widgets."""

        self._issues = self.casting.conflicts(self._labels)
        self._shot_issues = {}
        for issue in self._issues:
            if issue.group_uuid:
                self._shot_issues.setdefault(issue.group_uuid, []).append(issue)
        row = self._row_index.get(group_uuid)
        if row is not None:
            issues = self._shot_issues.get(group_uuid, [])
            error = any(issue.is_error for issue in issues)
            label_widget = self.table.cellWidget(row, 8)
            if isinstance(label_widget, QLabel):
                label_widget.setText(self._actor_text(group_uuid))
                label_widget.setToolTip("\n".join(issue.message for issue in issues))
                label_widget.setStyleSheet(
                    "color: #ff8b8b;"
                    if error
                    else "color: #ffc35c;"
                    if issues
                    else "color: #c4d3e6;"
                )
            for column in (0, 2, 3, 4):
                item = self.table.item(row, column)
                if item is None:
                    continue
                item.setBackground(
                    QColor("#3a2130")
                    if error
                    else QColor("#3a3021")
                    if issues
                    else QColor("#00000000")
                )
        self._update_summary()

    def _transcript_edited(self, group_uuid: str, editor: QPlainTextEdit) -> None:
        if self._populating:
            return
        self.transcript_changed.emit(group_uuid, editor.toPlainText())

    def _assign_roles(self, group_uuid: str) -> None:
        dialog = RoleAssignDialog(
            self, self.casting, group_uuid, self._labels.get(group_uuid, "")
        )
        if dialog.exec() != QDialog.Accepted:
            return
        self.rebuild()
        self.data_changed.emit()

    def _open_library(self) -> None:
        dialog = CastingLibraryDialog(self, self.casting)
        dialog.exec()
        self._refresh_actor_filter()
        self.rebuild()
        self.data_changed.emit()

    def _filters_changed(self) -> None:
        if self.project is None:
            return
        self.rebuild()

    def _refresh_actor_filter(self) -> None:
        current = self.actor_filter.currentData()
        self.actor_filter.blockSignals(True)
        self.actor_filter.clear()
        self.actor_filter.addItem("全部演员", "")
        for actor in self.casting.actors:
            self.actor_filter.addItem(actor.name, actor.uuid)
        index = self.actor_filter.findData(current) if current else 0
        self.actor_filter.setCurrentIndex(index if index >= 0 else 0)
        self.actor_filter.blockSignals(False)

    # -- export ----------------------------------------------------------
    def export_csv(self) -> None:
        if self.project is None or not self.project.groups:
            QMessageBox.information(self, "暂无拍摄表", "请先导入并分析视频，再导出拍摄表。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出拍摄表", self._export_path, "CSV 文件 (*.csv)"
        )
        if not path:
            return
        self._export_path = path
        try:
            with open(path, "w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "镜号",
                        "原始镜头",
                        "入点",
                        "出点",
                        "时长(秒)",
                        "人数",
                        "拍摄日期",
                        "角色分配",
                        "演员与档期",
                        "台词",
                        "冲突提示",
                    ]
                )
                for group in self.project.groups:
                    start, end = self.project.group_interval(group.uuid)
                    shot = self.casting.shot(group.uuid)
                    source_shots = {
                        item.uuid: item for item in self.project.source_shots
                    }
                    source_text = "、".join(
                        f"{source_shots[source_id].ordinal:03d}"
                        for source_id in group.source_ids
                    )
                    roles = self.casting.roles_for_shot(group.uuid)
                    role_text = "、".join(
                        f"{role.name}（{self.casting.actor(role.actor_uuid).name}）"
                        if self.casting.actor(role.actor_uuid)
                        else role.name
                        for role in roles
                    )
                    actor_text = " / ".join(
                        f"{actor.name} {actor.window_text()}"
                        for actor in self.casting.actors_for_shot(group.uuid)
                    )
                    issue_text = "；".join(
                        issue.message
                        for issue in self._shot_issues.get(group.uuid, [])
                    )
                    writer.writerow(
                        [
                            self._labels.get(group.uuid, ""),
                            source_text,
                            _timecode(start),
                            _timecode(end),
                            f"{max(0.0, end - start):.3f}",
                            shot.character_count if shot and shot.character_count else "",
                            shot.shoot_date if shot else "",
                            role_text,
                            actor_text,
                            self._transcripts.get(group.uuid, ""),
                            issue_text,
                        ]
                    )
        except OSError as error:
            QMessageBox.critical(self, "导出失败", str(error))
            return
        QMessageBox.information(self, "导出完成", f"拍摄表已导出：\n{path}")


__all__ = ["CastingLibraryDialog", "RoleAssignDialog", "ShootingTablePage"]
