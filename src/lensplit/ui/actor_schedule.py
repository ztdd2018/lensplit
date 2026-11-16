"""Actor schedule page: the master availability board for the whole project."""

from __future__ import annotations

from datetime import date, timedelta

from PySide6.QtCore import QDate, QRectF, QSignalBlocker, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDateEdit,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lensplit.core.casting import (
    Actor,
    AvailabilityIssue,
    CastingData,
    parse_day,
)
from lensplit.core.model import ShotProject
from lensplit.ui.shooting_table import (
    CastingLibraryDialog,
    export_actor_template,
    import_actors_from_csv,
)

_UNSET_DATE = QDate(2000, 1, 1)
_COLUMNS = ("演员", "可拍摄开始", "可拍摄结束", "参演镜头", "档期占用条", "状态")
_WIDTHS = {0: 170, 1: 138, 2: 138, 3: 210, 5: 190}


def _qdate(value: str) -> QDate:
    parsed = QDate.fromString(value, "yyyy-MM-dd")
    return parsed if parsed.isValid() else _UNSET_DATE


def _centered(widget: QWidget, vertical: int = 18) -> QWidget:
    holder = QWidget()
    layout = QHBoxLayout(holder)
    layout.setContentsMargins(4, vertical, 4, vertical)
    layout.addWidget(widget)
    return holder


class AvailabilityBar(QWidget):
    """Paint one actor's window with a tick per planned shoot day."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._span_start: date | None = None
        self._span_end: date | None = None
        self._window: tuple[date | None, date | None] = (None, None)
        self._days: list[tuple[date, bool]] = []
        self.setFixedHeight(26)
        self.setMinimumWidth(150)

    def set_span(self, start: date, end: date) -> None:
        self._span_start = start
        self._span_end = end
        self.update()

    def set_data(
        self,
        window: tuple[date | None, date | None],
        days: list[tuple[date, bool]],
    ) -> None:
        self._window = window
        self._days = days
        self.update()

    def _ratio(self, day: date) -> float | None:
        if self._span_start is None or self._span_end is None:
            return None
        total = max(1, (self._span_end - self._span_start).days)
        offset = (day - self._span_start).days
        return min(1.0, max(0.0, offset / total))

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        track = QRectF(4.0, 7.0, max(1.0, self.width() - 8.0), self.height() - 14.0)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#232c3a"))
        painter.drawRoundedRect(track, 4.0, 4.0)

        start, end = self._window
        if start is not None or end is not None:
            left = self._ratio(start) if start is not None else 0.0
            right = self._ratio(end) if end is not None else 1.0
            if left is not None and right is not None:
                band = QRectF(
                    track.left() + left * track.width(),
                    track.top(),
                    max(2.0, (right - left) * track.width()),
                    track.height(),
                )
                painter.setBrush(QColor("#2f6b4f"))
                painter.drawRoundedRect(band, 3.0, 3.0)

        for day, inside in self._days:
            ratio = self._ratio(day)
            if ratio is None:
                continue
            x = track.left() + ratio * track.width()
            painter.setPen(QPen(QColor("#7ee2a8" if inside else "#ff8b8b"), 2))
            painter.drawLine(int(x), int(track.top()) - 2, int(x), int(track.bottom()) + 2)
        painter.setPen(QPen(QColor("#39455a"), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(track, 4.0, 4.0)
        painter.end()


class ActorSchedulePage(QWidget):
    """Master availability board; pages 1 and 2 follow whatever is set here."""

    data_changed = Signal()
    window_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project: ShotProject | None = None
        self.casting = CastingData()
        self._labels: dict[str, str] = {}
        self._rows: list[str] = []
        self._issues: dict[str, list[AvailabilityIssue]] = {}
        self._populating = False
        self._dirty = True
        self._build_ui()

    # -- construction ----------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 8)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(8)
        title = QLabel("演员与档期 · 主控档期表")
        title.setObjectName("statsLabel")
        header.addWidget(title)
        hint = QLabel("这里填写的档期是主蓝本：页面 1、2 的拍摄日期会以此校验，越界会立即提示。")
        hint.setObjectName("mutedLabel")
        header.addWidget(hint, 1)
        self.add_button = QPushButton("添加演员…")
        self.add_button.clicked.connect(self._add_actor)
        header.addWidget(self.add_button)
        self.import_button = QPushButton("批量导入演员…")
        self.import_button.setToolTip(
            "CSV 表头：姓名,可拍摄开始,可拍摄结束,备注（日期 YYYY-MM-DD，可留空）"
        )
        self.import_button.clicked.connect(self._import_actors)
        header.addWidget(self.import_button)
        self.template_button = QPushButton("下载模板")
        self.template_button.clicked.connect(lambda: export_actor_template(self))
        header.addWidget(self.template_button)
        self.library_button = QPushButton("角色与演员库…")
        self.library_button.clicked.connect(self._open_library)
        header.addWidget(self.library_button)
        layout.addLayout(header)

        self.summary_label = QLabel("尚未添加演员")
        self.summary_label.setObjectName("statsLabel")
        layout.addWidget(self.summary_label)

        self.issue_label = QLabel()
        self.issue_label.setObjectName("conflictLabel")
        self.issue_label.setWordWrap(True)
        self.issue_label.setVisible(False)
        layout.addWidget(self.issue_label)

        group = QGroupBox("档期看板")
        group_layout = QVBoxLayout(group)
        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(_COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(40)
        self.table.setShowGrid(False)
        self.table.setWordWrap(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header_view = self.table.horizontalHeader()
        header_view.setHighlightSections(False)
        for column, width in _WIDTHS.items():
            header_view.setSectionResizeMode(column, QHeaderView.Fixed)
            self.table.setColumnWidth(column, width)
        header_view.setSectionResizeMode(4, QHeaderView.Stretch)
        header_view.setSectionResizeMode(5, QHeaderView.Fixed)
        self.table.setColumnWidth(5, 190)
        self.table.itemChanged.connect(self._name_changed)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        group_layout.addWidget(self.table, 1)
        layout.addWidget(group, 2)

        detail_group = QGroupBox("镜头明细 · 选中演员的拍摄安排")
        detail_layout = QVBoxLayout(detail_group)
        self.detail_hint = QLabel("在上表选择一个演员，查看其镜头日期是否落在档期内。")
        self.detail_hint.setObjectName("mutedLabel")
        detail_layout.addWidget(self.detail_hint)
        self.detail_table = QTableWidget(0, 3)
        self.detail_table.setHorizontalHeaderLabels(["镜号", "拍摄日期", "状态"])
        self.detail_table.verticalHeader().setVisible(False)
        self.detail_table.verticalHeader().setDefaultSectionSize(30)
        self.detail_table.setShowGrid(False)
        self.detail_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.detail_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.detail_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        detail_header = self.detail_table.horizontalHeader()
        detail_header.setSectionResizeMode(0, QHeaderView.Fixed)
        detail_header.setSectionResizeMode(1, QHeaderView.Fixed)
        detail_header.setSectionResizeMode(2, QHeaderView.Stretch)
        self.detail_table.setColumnWidth(0, 150)
        self.detail_table.setColumnWidth(1, 140)
        detail_layout.addWidget(self.detail_table, 1)

        action_row = QHBoxLayout()
        self.extend_button = QPushButton("延长档期到该日")
        self.extend_button.clicked.connect(self._extend_selected)
        action_row.addWidget(self.extend_button)
        self.clamp_button = QPushButton("拍摄日改为档期内")
        self.clamp_button.clicked.connect(self._clamp_selected)
        action_row.addWidget(self.clamp_button)
        self.ignore_button = QPushButton("暂不处理")
        self.ignore_button.setToolTip("保留现状，只在页面上继续显示越界提示")
        self.ignore_button.clicked.connect(self._ignore_selected)
        action_row.addWidget(self.ignore_button)
        action_row.addStretch(1)
        detail_layout.addLayout(action_row)
        layout.addWidget(detail_group, 1)

    # -- context ---------------------------------------------------------
    def set_context(
        self,
        project: ShotProject | None,
        casting: CastingData,
        labels: dict[str, str],
    ) -> None:
        self.project = project
        self.casting = casting
        self._labels = dict(labels)
        self._dirty = True
        if self.isVisible():
            self.rebuild()

    def activate(self) -> None:
        if self._dirty:
            self.rebuild()

    def is_dirty(self) -> bool:
        return self._dirty

    # -- rendering -------------------------------------------------------
    def _span(self) -> tuple[date, date]:
        days: list[date] = []
        for actor in self.casting.actors:
            start = parse_day(actor.available_from)
            end = parse_day(actor.available_to)
            if start is not None:
                days.append(start)
            if end is not None:
                days.append(end)
            for _group, day in self.casting.shots_for_actor(actor.uuid):
                parsed = parse_day(day)
                if parsed is not None:
                    days.append(parsed)
        if not days:
            today = date.today()
            return today - timedelta(days=1), today + timedelta(days=14)
        return min(days) - timedelta(days=2), max(days) + timedelta(days=2)

    def _day_states(self, actor: Actor) -> list[tuple[date, bool]]:
        states: list[tuple[date, bool]] = []
        for _group, day in self.casting.shots_for_actor(actor.uuid):
            parsed = parse_day(day)
            if parsed is None:
                continue
            empty = not actor.available_from and not actor.available_to
            states.append((parsed, empty or actor.is_available(day)))
        return states

    def rebuild(self) -> None:
        self._dirty = False
        self._issues = {}
        for actor in self.casting.actors:
            issues = self.casting.check_actor_schedule(actor.uuid)
            if issues:
                self._issues[actor.uuid] = issues

        span = self._span()
        keep = self._selected_actor_uuid()
        self._populating = True
        self.table.setUpdatesEnabled(False)
        self.table.setRowCount(0)
        self._rows = []
        for actor in self.casting.actors:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._rows.append(actor.uuid)
            name_item = QTableWidgetItem(actor.name)
            name_item.setData(Qt.UserRole, actor.uuid)
            self.table.setItem(row, 0, name_item)
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
                    widget=editor: self._window_edited(uuid, which, qdate, widget)
                )
                self.table.setCellWidget(row, column, _centered(editor, 4))

            shots = self.casting.shots_for_actor(actor.uuid)
            days = sorted({day for _group, day in shots})
            shot_text = "—"
            if shots:
                shown = "、".join(day[5:] for day in days[:4])
                if len(days) > 4:
                    shown += " …"
                shot_text = f"{len(shots)} 镜 · {shown}"
            shot_item = QTableWidgetItem(shot_text)
            shot_item.setForeground(QColor("#c4d3e6"))
            self.table.setItem(row, 3, shot_item)

            bar = AvailabilityBar()
            bar.setMinimumHeight(26)
            bar.set_span(*span)
            window = (parse_day(actor.available_from), parse_day(actor.available_to))
            states = self._day_states(actor)
            bar.set_data(window, states)
            bar.setToolTip(self._bar_tooltip(actor, states))
            self.table.setCellWidget(row, 4, _centered(bar, 6))

            issues = self._issues.get(actor.uuid, [])
            status = QTableWidgetItem()
            if not actor.available_from and not actor.available_to:
                status.setText("未填写档期" if not issues else f"{len(issues)} 镜待定档期")
                status.setForeground(QColor("#ffc35c"))
            elif issues:
                status.setText(f"{len(issues)} 镜越界")
                status.setForeground(QColor("#ff8b8b"))
            elif shots:
                status.setText("✓ 全部在档期内")
                status.setForeground(QColor("#7ee2a8"))
            else:
                status.setText(actor.window_text())
                status.setForeground(QColor("#95a3b7"))
            if issues:
                status.setToolTip("\n".join(issue.message for issue in issues[:6]))
            self.table.setItem(row, 5, status)
        self.table.setUpdatesEnabled(True)
        if self._rows:
            target = self._rows.index(keep) if keep in self._rows else 0
            self.table.selectRow(target)
        self._populating = False
        self._update_summary()
        self._refresh_detail()

    def _selected_actor_uuid(self) -> str | None:
        row = self.table.currentRow()
        if 0 <= row < len(self._rows):
            return self._rows[row]
        return None

    def _bar_tooltip(self, actor: Actor, states: list[tuple[date, bool]]) -> str:
        lines = [f"{actor.name} · 档期 {actor.window_text()}"]
        for day, inside in states:
            lines.append(f"{day.isoformat()} {'✓ 在档期内' if inside else '✕ 档期外'}")
        return "\n".join(lines)

    def _update_summary(self) -> None:
        actors = len(self.casting.actors)
        planned = sum(
            1
            for actor in self.casting.actors
            if actor.available_from or actor.available_to
        )
        errors = sum(len(issues) for issues in self._issues.values())
        total_shots = len(self.project.groups) if self.project is not None else 0
        self.summary_label.setText(
            f"演员 {actors} 人 ｜ 已登记档期 {planned} ｜ 越界镜头 {errors} ｜ 全片 {total_shots} 镜"
        )
        self.summary_label.setStyleSheet(
            "color: #ff8b8b;" if errors else "color: #a9c8f2;"
        )
        if errors:
            lines = [
                f"{issue.actor_name}：{issue.message.split('，', 1)[-1]}"
                for issues in self._issues.values()
                for issue in issues[:2]
            ]
            self.issue_label.setText("\n".join(lines[:4]))
        self.issue_label.setVisible(bool(errors))

    # -- table edits -----------------------------------------------------
    def _name_changed(self, item: QTableWidgetItem) -> None:
        if self._populating or item.column() != 0:
            return
        actor = self.casting.actor(item.data(Qt.UserRole) or "")
        if actor is None:
            return
        try:
            actor.set_name(item.text())
        except ValueError as error:
            QMessageBox.warning(self, "姓名不可用", str(error))
            with QSignalBlocker(self.table):
                item.setText(actor.name)
            return
        self.data_changed.emit()

    def _window_edited(
        self, actor_uuid: str, field: str, qdate: QDate, widget: QDateEdit
    ) -> None:
        if self._populating:
            return
        actor = self.casting.actor(actor_uuid)
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
            return
        self._refresh_row(actor_uuid)
        self.window_changed.emit(actor_uuid)
        self.data_changed.emit()

    def _refresh_row(self, actor_uuid: str) -> None:
        """Update one row's bar and status without rebuilding the table."""

        actor = self.casting.actor(actor_uuid)
        if actor is None or actor_uuid not in self._rows:
            return
        row = self._rows.index(actor_uuid)
        issues = self.casting.check_actor_schedule(actor_uuid)
        if issues:
            self._issues[actor_uuid] = issues
        else:
            self._issues.pop(actor_uuid, None)

        holder = self.table.cellWidget(row, 4)
        bar = holder.findChild(AvailabilityBar) if holder is not None else None
        states = self._day_states(actor)
        if bar is not None:
            bar.set_span(*self._span())
            bar.set_data(
                (parse_day(actor.available_from), parse_day(actor.available_to)),
                states,
            )
            bar.setToolTip(self._bar_tooltip(actor, states))

        shots = self.casting.shots_for_actor(actor.uuid)
        shot_item = self.table.item(row, 3)
        if shot_item is not None:
            if not shots:
                shot_item.setText("—")
            else:
                days = sorted({day for _group, day in shots})
                shown = "、".join(day[5:] for day in days[:4])
                if len(days) > 4:
                    shown += " …"
                shot_item.setText(f"{len(shots)} 镜 · {shown}")

        status = self.table.item(row, 5)
        if status is not None:
            if not actor.available_from and not actor.available_to:
                status.setText("未填写档期" if not issues else f"{len(issues)} 镜待定档期")
                status.setForeground(QColor("#ffc35c"))
            elif issues:
                status.setText(f"{len(issues)} 镜越界")
                status.setForeground(QColor("#ff8b8b"))
            elif shots:
                status.setText("✓ 全部在档期内")
                status.setForeground(QColor("#7ee2a8"))
            else:
                status.setText(actor.window_text())
                status.setForeground(QColor("#95a3b7"))
            status.setToolTip("\n".join(issue.message for issue in issues[:6]))
        self._update_summary()
        self._refresh_detail()

    def _add_actor(self) -> None:
        name, accepted = QInputDialog.getText(
            self, "添加演员", "演员姓名：", text=f"演员{len(self.casting.actors) + 1}"
        )
        if not accepted or not name.strip():
            return
        try:
            actor = self.casting.add_actor(name)
        except ValueError as error:
            QMessageBox.warning(self, "无法添加演员", str(error))
            return
        self.rebuild()
        row = self._rows.index(actor.uuid) if actor.uuid in self._rows else -1
        if row >= 0:
            self.table.selectRow(row)
        self.data_changed.emit()

    def _import_actors(self) -> None:
        if not import_actors_from_csv(self, self.casting):
            return
        self.rebuild()
        self.data_changed.emit()

    def _open_library(self) -> None:
        dialog = CastingLibraryDialog(self, self.casting)
        dialog.exec()
        self.rebuild()
        self.data_changed.emit()

    # -- detail ----------------------------------------------------------
    def _selected_actor(self) -> Actor | None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self._rows):
            return None
        return self.casting.actor(self._rows[row])

    def _selected_issue(self) -> AvailabilityIssue | None:
        actor = self._selected_actor()
        if actor is None:
            return None
        row = self.detail_table.currentRow()
        issues = self._current_detail_issues()
        if 0 <= row < len(issues):
            return issues[row]
        return None

    def _current_detail_issues(self) -> list[AvailabilityIssue]:
        actor = self._selected_actor()
        if actor is None:
            return []
        return self.casting.check_actor_schedule(actor.uuid)

    def _refresh_detail(self) -> None:
        actor = self._selected_actor()
        self.detail_table.setRowCount(0)
        if actor is None:
            self.detail_hint.setText("在上表选择一个演员，查看其镜头日期是否落在档期内。")
            return
        issues = self.casting.check_actor_schedule(actor.uuid)
        by_group = {issue.group_uuid: issue for issue in issues}
        shots = self.casting.shots_for_actor(actor.uuid)
        if not shots:
            self.detail_hint.setText(
                f"{actor.name} 还没有被分配到任何镜头；可在页面 1 / 2 的「分配角色」里安排。"
            )
            return
        if issues:
            self.detail_hint.setText(
                f"{actor.name} 档期 {actor.window_text()} · {len(issues)} 个镜头越界，可逐条处理。"
            )
        else:
            self.detail_hint.setText(
                f"{actor.name} 档期 {actor.window_text()} · 全部 {len(shots)} 个镜头都在档期内。"
            )
        for group_uuid, day in shots:
            row = self.detail_table.rowCount()
            self.detail_table.insertRow(row)
            label = self._labels.get(group_uuid, group_uuid[:8])
            self.detail_table.setItem(row, 0, QTableWidgetItem(label))
            self.detail_table.setItem(row, 1, QTableWidgetItem(day))
            issue = by_group.get(group_uuid)
            status = QTableWidgetItem()
            if issue is None:
                status.setText("✓ 在档期内")
                status.setForeground(QColor("#7ee2a8"))
            elif issue.kind == "empty":
                status.setText("档期未填写")
                status.setForeground(QColor("#ffc35c"))
            else:
                status.setText("✕ 档期外")
                status.setForeground(QColor("#ff8b8b"))
            if issue is not None:
                status.setToolTip(issue.message)
            self.detail_table.setItem(row, 2, status)
        if issues:
            self.detail_table.selectRow(0)

    def _selection_changed(self) -> None:
        if self._populating:
            return
        self._refresh_detail()

    def _extend_selected(self) -> None:
        issue = self._selected_issue()
        if issue is None:
            QMessageBox.information(self, "请选择越界镜头", "请先在下方明细里选中一条越界记录。")
            return
        if self.casting.extend_availability(issue.actor_uuid, issue.day):
            self.window_changed.emit(issue.actor_uuid)
            QTimer.singleShot(0, self.rebuild)
            self.data_changed.emit()

    def _clamp_selected(self) -> None:
        issue = self._selected_issue()
        if issue is None:
            QMessageBox.information(self, "请选择越界镜头", "请先在下方明细里选中一条越界记录。")
            return
        moved = self.casting.clamp_for_issues([issue])
        if moved:
            QTimer.singleShot(0, self.rebuild)
            self.data_changed.emit()

    def _ignore_selected(self) -> None:
        issue = self._selected_issue()
        if issue is None:
            return
        self.detail_hint.setText(f"已保留现状：{issue.message}。该提示会一直显示，直到你调整档期或拍摄日。")


__all__ = ["ActorSchedulePage", "AvailabilityBar"]
