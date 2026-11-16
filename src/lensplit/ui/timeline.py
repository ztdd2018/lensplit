"""Compact horizontal timeline strip used by the desktop editor."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPoint, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QScrollArea, QWidget


def _clock(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    minutes, remainder = divmod(milliseconds, 60_000)
    secs, millis = divmod(remainder, 1000)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


@dataclass
class TimelineItem:
    """One logical shot rendered as a card on the strip."""

    uuid: str
    label: str
    start: float
    end: float
    source_count: int = 1
    transcript: str = ""
    recognized: bool = False
    source_text: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class TimelineCanvas(QWidget):
    """Paints the cards and owns selection state."""

    CARD_WIDTH = 172
    CARD_HEIGHT = 142
    GAP = 10
    PADDING = 10

    activated = Signal(str)
    selection_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._items: list[TimelineItem] = []
        self._thumbnails: dict[str, QPixmap] = {}
        self._selected: set[str] = set()
        self._current: str | None = None
        self._anchor: str | None = None
        self._hover: str | None = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFixedHeight(self.CARD_HEIGHT + self.PADDING * 2)

    # -- data -----------------------------------------------------------
    def items(self) -> list[TimelineItem]:
        return list(self._items)

    def item(self, group_uuid: str) -> TimelineItem | None:
        return next((item for item in self._items if item.uuid == group_uuid), None)

    def set_items(self, items: list[TimelineItem]) -> None:
        self._items = list(items)
        known = {item.uuid for item in self._items}
        self._selected &= known
        if self._current not in known:
            self._current = None
        if self._anchor not in known:
            self._anchor = None
        self.setMinimumWidth(self.content_width())
        self.update()

    def content_width(self) -> int:
        return self.PADDING * 2 + max(0, len(self._items)) * (self.CARD_WIDTH + self.GAP)

    def set_thumbnail(self, group_uuid: str, pixmap: QPixmap) -> None:
        self._thumbnails[group_uuid] = pixmap
        if any(item.uuid == group_uuid for item in self._items):
            self.update()

    def set_transcript(self, group_uuid: str, text: str, recognized: bool = True) -> None:
        for item in self._items:
            if item.uuid == group_uuid:
                item.transcript = text
                item.recognized = recognized
                self.update()
                return

    # -- selection ------------------------------------------------------
    def selected_ids(self) -> list[str]:
        return [item.uuid for item in self._items if item.uuid in self._selected]

    def current_id(self) -> str | None:
        return self._current

    def set_selection(self, ids: list[str], current: str | None = None) -> None:
        known = {item.uuid for item in self._items}
        self._selected = {item_id for item_id in ids if item_id in known}
        if current in known:
            self._current = current
        elif self._current not in known:
            self._current = None
        if self._anchor not in known:
            self._anchor = self._current
        self.update()

    def select_indices(self, indices: list[int], current_index: int | None = None) -> None:
        chosen = [
            self._items[index].uuid
            for index in indices
            if 0 <= index < len(self._items)
        ]
        current = None
        if current_index is not None and 0 <= current_index < len(self._items):
            current = self._items[current_index].uuid
        elif chosen:
            current = chosen[-1]
        self._selected = set(chosen)
        self._current = current
        self._anchor = current
        self.update()
        self.selection_changed.emit()

    def index_of(self, group_uuid: str | None) -> int | None:
        return next(
            (index for index, item in enumerate(self._items) if item.uuid == group_uuid),
            None,
        )

    # -- geometry -------------------------------------------------------
    def card_rect(self, index: int) -> QRect:
        return QRect(
            self.PADDING + index * (self.CARD_WIDTH + self.GAP),
            self.PADDING,
            self.CARD_WIDTH,
            self.CARD_HEIGHT,
        )

    def _index_at(self, position: QPoint) -> int | None:
        for index in range(len(self._items)):
            if self.card_rect(index).contains(position):
                return index
        return None

    # -- painting -------------------------------------------------------
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        painter.fillRect(event.rect(), QColor("#11161e"))
        for index, item in enumerate(self._items):
            rect = self.card_rect(index)
            if rect.intersects(event.rect()):
                self._paint_card(painter, rect, item)
        painter.end()

    def _paint_card(self, painter: QPainter, rect: QRect, item: TimelineItem) -> None:
        selected = item.uuid in self._selected
        current = item.uuid == self._current
        hover = item.uuid == self._hover
        if selected:
            background = QColor("#223a5c")
        elif hover:
            background = QColor("#202b3b")
        else:
            background = QColor("#1a2230")
        painter.setPen(Qt.NoPen)
        painter.setBrush(background)
        painter.drawRoundedRect(rect, 8, 8)
        border = QColor("#66a1f0") if selected else QColor("#4a5d78") if current else QColor("#2c3646")
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(border, 2 if (selected or current) else 1))
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 8, 8)

        thumb = QRect(
            rect.left() + 6,
            rect.top() + 6,
            rect.width() - 12,
            (rect.width() - 12) * 9 // 16,
        )
        path = QPainterPath()
        path.addRoundedRect(QRectF(thumb), 5, 5)
        painter.save()
        painter.setClipPath(path)
        pixmap = self._thumbnails.get(item.uuid)
        if pixmap is None or pixmap.isNull():
            painter.fillRect(thumb, QColor("#0d1219"))
        else:
            scaled = pixmap.scaled(
                thumb.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
            )
            source = QRect(
                max(0, (scaled.width() - thumb.width()) // 2),
                max(0, (scaled.height() - thumb.height()) // 2),
                thumb.width(),
                thumb.height(),
            )
            painter.drawPixmap(thumb, scaled, source)
        painter.restore()

        self._paint_pill(painter, thumb.adjusted(4, 4, -4, -4), item.label, bold=True)
        if item.source_count > 1:
            self._paint_chip(
                painter,
                thumb.adjusted(4, 4, -4, -4),
                f"×{item.source_count}",
                QColor(255, 186, 66, 210),
                align_right=True,
            )

        caption_top = thumb.bottom() + 6
        text_width = rect.width() - 12
        font = QFont(self.font())
        font.setPointSizeF(9.0)
        painter.setFont(font)
        painter.setPen(QColor("#9fb3cc"))
        painter.drawText(
            QRect(rect.left() + 6, caption_top, text_width, 16),
            Qt.AlignLeft | Qt.AlignVCenter,
            f"{item.duration:.2f}s",
        )
        status = "无台词" if item.recognized and not item.transcript else ("台词" if item.transcript else "")
        if status:
            painter.setPen(QColor("#6f7a89") if status == "无台词" else QColor("#7fb6f2"))
            painter.drawText(
                QRect(rect.left() + 6, caption_top, text_width, 16),
                Qt.AlignRight | Qt.AlignVCenter,
                status,
            )
        detail_top = caption_top + 17
        detail = item.transcript
        painter.setPen(QColor("#c4d3e6") if detail else QColor("#6f7a89"))
        if not detail and not item.recognized:
            detail = ""
        metrics = QFontMetrics(font)
        if detail:
            detail = metrics.elidedText(detail, Qt.ElideRight, text_width)
        painter.drawText(
            QRect(rect.left() + 6, detail_top, text_width, 16),
            Qt.AlignLeft | Qt.AlignVCenter,
            detail,
        )

    def _paint_pill(
        self, painter: QPainter, area: QRect, text: str, bold: bool = False
    ) -> None:
        font = QFont(self.font())
        font.setPointSizeF(9.5)
        font.setBold(bold)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        width = min(area.width(), metrics.horizontalAdvance(text) + 12)
        pill = QRect(area.left(), area.top(), width, metrics.height() + 4)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 175))
        painter.drawRoundedRect(pill, 4, 4)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(
            pill, Qt.AlignCenter, metrics.elidedText(text, Qt.ElideRight, max(0, width - 8))
        )

    def _paint_chip(
        self,
        painter: QPainter,
        area: QRect,
        text: str,
        color: QColor,
        align_right: bool = False,
    ) -> None:
        font = QFont(self.font())
        font.setPointSizeF(9.0)
        font.setBold(True)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        width = metrics.horizontalAdvance(text) + 10
        left = area.right() - width if align_right else area.left()
        chip = QRect(left, area.top(), width, metrics.height() + 4)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 175))
        painter.drawRoundedRect(chip, 4, 4)
        painter.setPen(color)
        painter.drawText(chip, Qt.AlignCenter, text)

    # -- interaction ----------------------------------------------------
    def mousePressEvent(self, event) -> None:
        index = self._index_at(event.position().toPoint())
        modifiers = event.modifiers()
        if index is None:
            if not modifiers & Qt.ControlModifier and self._selected:
                self._selected.clear()
                self._selected_changed()
            return
        item = self._items[index]
        if modifiers & Qt.ShiftModifier and self._anchor is not None:
            anchor = self.index_of(self._anchor)
            if anchor is not None:
                low, high = sorted((anchor, index))
                self._selected = {entry.uuid for entry in self._items[low : high + 1]}
        elif modifiers & Qt.ControlModifier:
            if item.uuid in self._selected:
                self._selected.discard(item.uuid)
            else:
                self._selected.add(item.uuid)
        else:
            self._selected = {item.uuid}
        self._current = item.uuid
        self._anchor = item.uuid
        self.setFocus()
        self._selected_changed()
        if not modifiers & (Qt.ShiftModifier | Qt.ControlModifier):
            self.activated.emit(item.uuid)

    def _selected_changed(self) -> None:
        self.update()
        self.selection_changed.emit()

    def mouseMoveEvent(self, event) -> None:
        index = self._index_at(event.position().toPoint())
        hovered = self._items[index].uuid if index is not None else None
        if hovered != self._hover:
            self._hover = hovered
            self.setCursor(Qt.PointingHandCursor if hovered else Qt.ArrowCursor)
            self.setToolTip(self._tooltip_text(self._items[index]) if index is not None else "")
            self.update()

    @staticmethod
    def _tooltip_text(item: TimelineItem) -> str:
        lines = [
            f"{item.label} · {item.duration:.3f} 秒",
            f"原始镜头：{item.source_text or '—'}",
            f"{_clock(item.start)} → {_clock(item.end)} · 内部转场 {item.source_count - 1}",
        ]
        if item.transcript:
            lines.append(f"台词：{item.transcript}")
        elif item.recognized:
            lines.append("台词：（无台词）")
        else:
            lines.append("台词：（未识别）")
        return "\n".join(lines)

    def leaveEvent(self, event) -> None:
        if self._hover is not None:
            self._hover = None
            self.update()

    def keyPressEvent(self, event) -> None:
        if not self._items:
            super().keyPressEvent(event)
            return
        index = self.index_of(self._current)
        if event.key() in (Qt.Key_Left, Qt.Key_Right):
            step = -1 if event.key() == Qt.Key_Left else 1
            target = 0 if index is None else max(0, min(len(self._items) - 1, index + step))
            self.select_indices([target], target)
            self._ensure_visible(target)
            return
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and self._current:
            self.activated.emit(self._current)
            return
        if event.matches(QKeySequence.SelectAll) or (
            event.key() == Qt.Key_A and event.modifiers() & Qt.ControlModifier
        ):
            self.select_indices(list(range(len(self._items))))
            return
        super().keyPressEvent(event)

    def _ensure_visible(self, index: int) -> None:
        strip = self.parentWidget()
        while strip is not None and not isinstance(strip, QScrollArea):
            strip = strip.parentWidget()
        if isinstance(strip, QScrollArea):
            strip.ensureVisible(self.card_rect(index).center().x(), 0, 40, 0)


class TimelineStrip(QScrollArea):
    """Scrollable filmstrip with quick single/toggle/range selection."""

    activated = Signal(str)
    selection_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.canvas = TimelineCanvas(self)
        self.setWidget(self.canvas)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        self.viewport().setAutoFillBackground(False)
        self.viewport().setStyleSheet("background: transparent;")
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setFixedHeight(TimelineCanvas.CARD_HEIGHT + TimelineCanvas.PADDING * 2 + 18)
        self.canvas.activated.connect(self.activated)
        self.canvas.selection_changed.connect(self.selection_changed)

    def items(self) -> list[TimelineItem]:
        return self.canvas.items()

    def item(self, group_uuid: str) -> TimelineItem | None:
        return self.canvas.item(group_uuid)

    def set_items(self, items: list[TimelineItem]) -> None:
        self.canvas.set_items(items)

    def set_thumbnail(self, group_uuid: str, pixmap: QPixmap) -> None:
        self.canvas.set_thumbnail(group_uuid, pixmap)

    def set_transcript(self, group_uuid: str, text: str, recognized: bool = True) -> None:
        self.canvas.set_transcript(group_uuid, text, recognized)

    def selected_ids(self) -> list[str]:
        return self.canvas.selected_ids()

    def current_id(self) -> str | None:
        return self.canvas.current_id()

    def set_selection(self, ids: list[str], current: str | None = None) -> None:
        self.canvas.set_selection(ids, current)
        index = self.canvas.index_of(current)
        if index is not None:
            self.canvas._ensure_visible(index)

    def select_indices(self, indices: list[int], current_index: int | None = None) -> None:
        self.canvas.select_indices(indices, current_index)

    def wheelEvent(self, event) -> None:
        bar = self.horizontalScrollBar()
        if bar.maximum() > 0 and not event.modifiers() & Qt.ControlModifier:
            bar.setValue(bar.value() - event.angleDelta().y() - event.angleDelta().x())
            event.accept()
            return
        super().wheelEvent(event)

    def sizeHint(self) -> QSize:
        return QSize(640, TimelineCanvas.CARD_HEIGHT + TimelineCanvas.PADDING * 2 + 18)
