import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QDate, Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
)

from lensplit.analysis.asr import TranscriptResult
from lensplit.core.casting import AvailabilityIssue, CastingData
from lensplit.core.model import ShotProject, SourceShot
from lensplit.core.numbering import MergeMode
from lensplit.export.exporter import ExportOptions
from lensplit.ui.actor_schedule import AvailabilityBar
from lensplit.ui.main_window import ExportDialog, MainWindow, ScheduleConflictDialog
from lensplit.ui.shooting_table import CastingLibraryDialog, RoleAssignDialog


def make_project():
    return ShotProject(
        [
            SourceShot(f"s{ordinal}", ordinal, ordinal - 1.0, float(ordinal))
            for ordinal in range(1, 4)
        ]
    )


class MainWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        self.window.set_project(make_project())

    def tearDown(self):
        with mock.patch.object(
            QMessageBox, "question", return_value=QMessageBox.Discard
        ):
            self.window.close()
        self.app.processEvents()

    def select_rows(self, *rows):
        self.window.timeline.select_indices(list(rows), list(rows)[-1])

    def labels(self):
        return [item.label for item in self.window.timeline.items()]

    def test_missing_local_model_disables_only_ai_mode(self):
        self.window.clear_model()
        self.assertFalse(self.window.engine_combo.model().item(1).isEnabled())
        self.assertTrue(self.window.engine_combo.model().item(0).isEnabled())
        self.assertIn("AI 模式已禁用", self.window.model_status_label.text())
        self.assertFalse(self.window.export_button.isEnabled())

    def test_project_model_is_loaded_with_threshold_on_startup(self):
        model = Path(__file__).resolve().parents[1] / "models" / "transnetv2.onnx"
        if not model.is_file():
            self.skipTest("models/transnetv2.onnx 未放置")
        self.assertEqual(self.window.engine_combo.currentData(), "transnet")
        self.assertAlmostEqual(self.window.threshold_spin.value(), 0.40)
        self.assertIn("模型就绪", self.window.model_status_label.text())

    def test_busy_state_keeps_import_clickable(self):
        self.window._set_busy("analysis")
        self.assertTrue(self.window.open_button.isEnabled())
        self.assertFalse(self.window.analyze_button.isEnabled())
        self.assertIn("分析中", self.window.analyze_button.text())

        self.window._set_busy(None)
        self.assertEqual(self.window.analyze_button.text(), "开始分析")

    def test_manual_mark_splits_group_and_renumbers(self):
        self.window._manual_marks = [1.5]
        self.window.apply_manual_marks()

        self.assertEqual(len(self.window.project.source_shots), 4)
        self.assertEqual(len(self.window.project.groups), 4)
        self.assertEqual(len(self.window.timeline.items()), 4)
        self.assertEqual(self.window._group_labels, ["001", "002", "003", "004"])
        self.assertEqual(self.window._manual_marks, [])
        self.assertEqual(
            [transition.engine for transition in self.window.project.transitions].count(
                "manual"
            ),
            1,
        )

    def test_position_slider_only_spans_the_selected_shot(self):
        self.select_rows(1)
        group = self.window.project.groups[1]
        start, end = self.window.project.group_interval(group.uuid)
        slider = self.window.position_slider
        self.assertEqual(slider.minimum(), round(start * 1000))
        self.assertEqual(slider.maximum(), round(end * 1000))
        self.assertIn("片段内", self.window.position_label.text())
        self.assertIn("刻度＝当前片段", self.window.mark_label.text())

    def test_session_can_be_saved_and_reopened(self):
        actor = self.window.casting.add_actor("苏婉晴", "2026-11-01", "2026-11-05")
        group_id = self.window.project.groups[0].uuid
        self.window.casting.set_character_count(group_id, 2)
        self.window.casting.assign_actor(
            self.window.casting.shot_slots(group_id)[0].uuid, actor.uuid
        )
        self.window._transcripts[group_id] = "第一句"
        self.window._analysis_metadata = {"engine": "fast"}

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "工程.lensplit"
            self.assertTrue(self.window.save_session_to(str(path)))
            self.assertFalse(self.window._dirty)

            reopened = MainWindow()
            try:
                self.assertTrue(reopened.open_session_from(str(path)))
                self.assertEqual(len(reopened.project.groups), 3)
                self.assertEqual(reopened._transcripts[group_id], "第一句")
                self.assertEqual(reopened._analysis_metadata["engine"], "fast")
                slots = reopened.casting.shot_slots(group_id)
                self.assertEqual([role.name for role in slots], ["角色 1", "角色 2"])
                self.assertEqual(reopened.casting.actor(slots[0].actor_uuid).name, "苏婉晴")
                self.assertEqual(reopened._session_path, str(path))
            finally:
                reopened.close()

    def test_conflicting_actor_is_tagged_on_the_first_page(self):
        actor = self.window.casting.add_actor("苏婉晴", "2026-11-01", "2026-11-05")
        group_id = self.window.project.groups[0].uuid
        self.window.casting.set_character_count(group_id, 1)
        self.window.casting.set_shoot_date(group_id, "2026-11-09")
        self.window.casting.assign_actor(
            self.window.casting.shot_slots(group_id)[0].uuid, actor.uuid
        )
        self.select_rows(0)
        self.window._refresh_actor_panel()

        self.assertIn("档期冲突", self.window.actor_schedule_label.text())
        tags = [label.text() for label in self.window.slot_area.findChildren(QLabel)]
        self.assertIn("档期冲突", tags)
        self.assertNotIn("11-01 ~ 11-05", self.window.actor_schedule_label.text())

    def test_actor_notes_are_shown_instead_of_window(self):
        actor = self.window.casting.add_actor("苏婉晴", "2026-11-01", "2026-11-05")
        actor.notes = "会骑马"
        group_id = self.window.project.groups[0].uuid
        self.window.casting.set_character_count(group_id, 1)
        self.window.casting.assign_actor(
            self.window.casting.shot_slots(group_id)[0].uuid, actor.uuid
        )
        self.select_rows(0)
        self.window._refresh_actor_panel()

        texts = [label.text() for label in self.window.slot_area.findChildren(QLabel)]
        self.assertIn("会骑马", texts)

    def test_transcript_edit_is_shown_and_saved(self):
        group_id = self.window.project.groups[0].uuid
        self.window._apply_transcript(
            group_id, TranscriptResult(text="测试台词", language="zh", segments=[])
        )
        self.assertEqual(self.window.timeline.item(group_id).transcript, "测试台词")

        self.window._transcripts[group_id] = ""
        self.window._update_item_transcript(group_id)
        item = self.window.timeline.item(group_id)
        self.assertEqual(item.transcript, "")
        self.assertTrue(item.recognized)

    def test_merge_joins_transcripts_and_undo_restores_them(self):
        first_id, second_id = [group.uuid for group in self.window.project.groups[:2]]
        self.window._transcripts[first_id] = "第一句"
        self.window._transcripts[second_id] = "第二句"
        self.select_rows(0, 1)
        self.window.merge_selected()

        merged_id = self.window.project.groups[0].uuid
        self.assertEqual(len(self.window.project.groups), 2)
        self.assertNotIn(first_id, self.window._transcripts)
        self.assertEqual(self.window.timeline.item(merged_id).transcript, "第一句 第二句")
        self.assertTrue(self.window.timeline.item(merged_id).recognized)

        self.window.undo()
        self.assertEqual(len(self.window.project.groups), 3)
        self.assertEqual(self.window._transcripts[first_id], "第一句")
        self.assertEqual(self.window._transcripts[second_id], "第二句")

    def test_merge_supports_all_three_numbering_modes_and_split(self):
        self.select_rows(0, 1)
        self.window.merge_selected()
        self.assertEqual(self.labels(), ["001+002", "003"])
        self.assertIn("有效转场 1", self.window.stats_label.text())
        self.assertIn("内部转场 1", self.window.stats_label.text())

        for mode, expected in (
            (MergeMode.SEQUENTIAL, ["001", "002"]),
            (MergeMode.FIRST, ["001", "003"]),
            (MergeMode.COMBINED, ["001+002", "003"]),
        ):
            self.window.merge_mode_combo.setCurrentIndex(
                self.window.merge_mode_combo.findData(mode.value)
            )
            self.assertEqual(self.labels(), expected)

        self.window.timeline.select_indices([0], 0)
        self.window.split_selected()
        self.assertEqual(self.labels(), ["001", "002", "003"])
        self.assertEqual(len(self.window.project.groups), 3)

    def test_merge_and_boundary_adjustment_are_undoable(self):
        original = self.window.project.snapshot()
        self.select_rows(0, 1)
        self.window.merge_selected()
        self.window.undo()
        self.assertEqual(self.window.project.snapshot(), original)

        self.window.cut_spin.setValue(0.5)
        self.window.apply_boundary()
        self.assertEqual(self.window.project.source_shots[0].end_pts, 0.5)
        self.assertEqual(self.window.project.source_shots[1].start_pts, 0.5)
        self.assertEqual(self.window.project.transitions[0].cut_pts, 0.5)
        self.window.undo()
        self.assertEqual(self.window.project.snapshot(), original)

    def test_numbering_controls_change_labels_without_mutating_sources(self):
        before = self.window.project.snapshot()
        self.window.prefix_edit.setText("S")
        self.window.suffix_edit.setText("#")
        self.window.width_spin.setValue(2)
        self.window.sequence_combo.setCurrentIndex(1)
        self.assertEqual(self.labels(), ["S0a#", "S0b#", "S0c#"])
        self.assertEqual(self.window.project.snapshot(), before)


class ShootingTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        self.window.set_project(make_project())

    def tearDown(self):
        with mock.patch.object(
            QMessageBox, "question", return_value=QMessageBox.Discard
        ):
            self.window.close()
        self.app.processEvents()

    def open_page(self):
        self.window.tabs.setCurrentIndex(1)
        self.window.shooting_page.activate()
        self.app.processEvents()
        return self.window.shooting_page

    def test_rows_follow_logical_shots(self):
        page = self.open_page()
        self.assertEqual(page.table.rowCount(), 3)
        self.assertEqual(page.table.item(0, 0).text(), "001")
        self.assertEqual(page.table.item(0, 2).text(), "00:00.000")
        self.assertEqual(page.table.item(0, 3).text(), "00:01.000")
        self.assertIn("共 3 镜", page.summary_label.text())

    def test_role_assignment_and_conflict_are_rendered(self):
        casting = self.window.casting
        actor = casting.add_actor("张三", "2026-11-01", "2026-11-05")
        role = casting.add_role("主角", actor.uuid)
        first = self.window.project.groups[0].uuid
        casting.set_roles(first, [role.uuid])
        casting.set_shoot_date(first, "2026-11-09")
        self.window._refresh_shooting_table()

        page = self.open_page()
        button = page.table.cellWidget(0, 7).findChild(QPushButton)
        self.assertEqual(button.text(), "主角（张三）")
        schedule = page.table.cellWidget(0, 8)
        self.assertIn("张三", schedule.text())
        self.assertIn("档期冲突", schedule.text())
        self.assertIn("撞档 1", page.summary_label.text())
        self.assertFalse(page.issue_label.isHidden())

    def test_transcript_edited_on_page_reaches_timeline(self):
        page = self.open_page()
        editor = page.table.cellWidget(0, 9).findChild(QPlainTextEdit)
        editor.setPlainText("第一句台词")
        group_id = self.window.project.groups[0].uuid
        self.assertEqual(self.window._transcripts[group_id], "第一句台词")
        self.assertEqual(self.window.timeline.item(group_id).transcript, "第一句台词")

    def test_merge_migrates_casting_and_undo_restores_it(self):
        casting = self.window.casting
        actor = casting.add_actor("张三")
        role = casting.add_role("主角", actor.uuid)
        first, _second = [group.uuid for group in self.window.project.groups[:2]]
        casting.set_roles(first, [role.uuid])
        self.window.timeline.select_indices([0, 1], 1)
        self.window.merge_selected()

        merged = self.window.project.groups[0].uuid
        self.assertEqual(self.window.casting.shot(merged).role_ids, [role.uuid])
        self.assertIn("拍摄表", self.window.tabs.tabText(1))

        self.window.undo()
        self.assertEqual(len(self.window.project.groups), 3)
        self.assertEqual(self.window.casting.shot(first).role_ids, [role.uuid])

    def test_thumbnail_click_requests_preview(self):
        page = self.open_page()
        requested = []
        page.preview_requested.connect(requested.append)
        page.table.cellWidget(0, 1).findChild(QPushButton).click()
        self.assertEqual(requested, [self.window.project.groups[0].uuid])

    def test_export_csv_writes_call_sheet_columns(self):
        casting = self.window.casting
        actor = casting.add_actor("张三", "2026-11-01", "2026-11-05")
        role = casting.add_role("主角", actor.uuid)
        casting.set_roles(self.window.project.groups[0].uuid, [role.uuid])
        self.window._transcripts[self.window.project.groups[0].uuid] = "你好"
        self.window._refresh_shooting_table()
        page = self.open_page()

        with tempfile.TemporaryDirectory() as directory:
            target = str(Path(directory) / "拍摄表.csv")
            with (
                mock.patch(
                    "lensplit.ui.shooting_table.QFileDialog.getSaveFileName",
                    return_value=(target, ""),
                ),
                mock.patch.object(QMessageBox, "information", return_value=QMessageBox.Ok),
            ):
                page.export_csv()
            text = Path(target).read_text(encoding="utf-8-sig")
        self.assertIn("镜号", text)
        self.assertIn("角色分配", text)
        self.assertIn("001", text)
        self.assertIn("主角（张三）", text)
        self.assertIn("你好", text)


class CastingDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_role_assign_dialog_edits_slots(self):
        casting = CastingData()
        actor = casting.add_actor("张三", "2026-11-01", "2026-11-05")
        casting.set_character_count("g1", 2)
        dialog = RoleAssignDialog(None, casting, "g1", "001")
        self.assertEqual(dialog.table.rowCount(), 2)
        dialog.table.cellWidget(0, 0).setText("阿强")
        combo = dialog.table.cellWidget(0, 1)
        combo.setCurrentIndex(combo.findData(actor.uuid))
        dialog.accept()
        slots = casting.shot_slots("g1")
        self.assertEqual([slot.name for slot in slots], ["阿强", "角色 2"])
        self.assertEqual(slots[0].actor_uuid, actor.uuid)
        self.assertEqual(slots[1].actor_uuid, "")

    def test_library_dialog_writes_actor_name_through(self):
        casting = CastingData()
        casting.add_actor("张三")
        dialog = CastingLibraryDialog(None, casting)
        self.assertEqual(dialog.actor_table.rowCount(), 1)
        dialog.actor_table.item(0, 0).setText("李四")
        self.assertEqual(casting.actors[0].name, "李四")


class ExportDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_copy_mode_explains_keyframe_alignment(self):
        dialog = ExportDialog(None, tempfile.gettempdir())
        dialog.mode_combo.setCurrentIndex(1)
        self.assertFalse(dialog.encoder_combo.isEnabled())
        self.assertIn("关键帧", dialog.boundary_hint.text())
        dialog.close()


class ActorScheduleUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        self.window.set_project(make_project())

    def tearDown(self):
        with mock.patch.object(
            QMessageBox, "question", return_value=QMessageBox.Discard
        ):
            self.window.close()
        self.app.processEvents()

    def open_board(self):
        self.window._refresh_actor_schedule()
        self.window.tabs.setCurrentIndex(2)
        page = self.window.actor_page
        page.activate()
        self.app.processEvents()
        return page

    def add_conflicting_actor(self, day="2026-11-09"):
        casting = self.window.casting
        actor = casting.add_actor("张三", "2026-11-01", "2026-11-05")
        role = casting.add_role("主角", actor.uuid)
        first = self.window.project.groups[0].uuid
        casting.set_roles(first, [role.uuid])
        casting.set_shoot_date(first, day)
        return first, actor

    def test_three_tabs_include_actor_board(self):
        self.assertEqual(self.window.tabs.count(), 3)
        self.assertEqual(self.window.tabs.tabText(2), "演员与档期")

    def test_board_shows_actor_window_and_conflict(self):
        self.add_conflicting_actor()
        page = self.open_board()
        self.assertEqual(page.table.rowCount(), 1)
        self.assertEqual(page.table.item(0, 0).text(), "张三")
        self.assertIn("越界", page.table.item(0, 5).text())
        self.assertIsNotNone(page.table.cellWidget(0, 4).findChild(AvailabilityBar))
        self.assertEqual(page.detail_table.rowCount(), 1)
        self.assertIn("档期外", page.detail_table.item(0, 2).text())
        self.assertIn("1 镜越界", self.window.tabs.tabText(2))
        self.assertIn("越界镜头 1", page.summary_label.text())

    def test_board_clamp_moves_shot_into_window(self):
        first, _actor = self.add_conflicting_actor()
        page = self.open_board()
        page.table.selectRow(0)
        page._clamp_selected()
        self.app.processEvents()
        self.assertEqual(self.window.casting.shot(first).shoot_date, "2026-11-05")
        self.assertNotIn("越界", self.window.tabs.tabText(2))

    def test_page1_panel_tracks_current_shot(self):
        first = self.window.project.groups[0].uuid
        self.window.timeline.set_selection([first], first)
        self.app.processEvents()
        self.assertEqual(self.window.actor_shot_label.text(), "001")

        self.window.actor_count_spin.setValue(3)
        self.assertEqual(self.window.casting.shot(first).character_count, 3)

        self.window.tabs.setCurrentIndex(1)
        self.window.shooting_page.activate()
        self.app.processEvents()
        holder = self.window.shooting_page.table.cellWidget(0, 5)
        self.assertEqual(holder.findChild(QSpinBox).value(), 3)

    def test_page1_date_seeds_missing_actor_window(self):
        casting = self.window.casting
        actor = casting.add_actor("李四")
        role = casting.add_role("替身", actor.uuid)
        first = self.window.project.groups[0].uuid
        casting.set_roles(first, [role.uuid])
        self.window.timeline.set_selection([first], first)
        self.app.processEvents()

        self.window.actor_date_edit.setDate(QDate(2026, 11, 8))
        self.window._reconcile_group(first)
        self.app.processEvents()

        self.assertEqual(self.window.casting.shot(first).shoot_date, "2026-11-08")
        self.assertEqual(actor.available_from, "2026-11-08")
        self.assertEqual(actor.available_to, "2026-11-08")
        self.assertIn("拍摄日 2026-11-08", self.window.actor_schedule_label.text())
        self.assertIn("档期无冲突", self.window.actor_schedule_label.text())


class ScheduleDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_choices_are_recorded(self):
        issues = [
            AvailabilityIssue("g1", "a1", "2026-11-26", "11-20 ~ 11-25", "outside", "张三")
        ]
        dialog = ScheduleConflictDialog(None, issues)
        self.assertIn("张三", dialog.findChild(QPlainTextEdit).toPlainText())
        dialog._choose("extend")
        self.assertEqual(dialog.choice, "extend")
        self.assertIn("不含拍摄日", issues[0].message)


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required for desktop workflow")
class DesktopWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def wait_until(self, predicate, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return
            time.sleep(0.01)
        self.app.processEvents()
        self.fail("Timed out waiting for the desktop background task")

    def test_import_detect_merge_and_export(self):
        with tempfile.TemporaryDirectory(prefix="镜头桌面测试_") as directory:
            root = Path(directory)
            video = root / "保密样例.mp4"
            command = [shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y"]
            for color in ("black", "white", "black"):
                command.extend(
                    ["-f", "lavfi", "-i", f"color=c={color}:s=64x64:r=10:d=1"]
                )
            command.extend(
                [
                    "-filter_complex",
                    "[0:v][1:v][2:v]concat=n=3:v=1:a=0[out]",
                    "-map",
                    "[out]",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-g",
                    "10",
                    str(video),
                ]
            )
            subprocess.run(command, check=True, capture_output=True, timeout=60)

            window = MainWindow()
            try:
                window.load_video(str(video))
                self.assertTrue(window.analyze_button.isEnabled())
                self.assertTrue(window.play_button.isEnabled())
                window.engine_combo.setCurrentIndex(0)
                window.threshold_spin.setValue(0.3)
                window.min_shot_spin.setValue(0.2)
                with mock.patch.object(QMessageBox, "critical", return_value=QMessageBox.Ok):
                    window.start_analysis()
                    self.wait_until(
                        lambda: window.project is not None and window._analysis_worker is None
                    )
                self.assertIsNotNone(window.project)
                self.assertEqual(len(window.project.source_shots), 3)
                self.assertGreater(window._analysis_metadata["elapsed_seconds"], 0)

                window.timeline.select_indices([0, 1], 1)
                window.merge_selected()
                options = ExportOptions(
                    output_dir=root / "导出结果",
                    scope="logical",
                    mode="exact",
                    encoder="cpu",
                    numbering_style=window.numbering_style(),
                    merge_mode=window.merge_mode(),
                    analysis_metadata=window._analysis_metadata,
                )
                with (
                    mock.patch.object(ExportDialog, "exec", return_value=QDialog.Accepted),
                    mock.patch.object(ExportDialog, "export_options", return_value=options),
                    mock.patch.object(QMessageBox, "information", return_value=QMessageBox.Ok),
                    mock.patch.object(QMessageBox, "critical", return_value=QMessageBox.Ok),
                ):
                    window.start_export()
                    self.wait_until(lambda: window._export_worker is None)
                output = root / "导出结果"
                self.assertTrue((output / "001+002.mp4").is_file())
                self.assertTrue((output / "003.mp4").is_file())
                self.assertTrue((output / "lensplit_manifest.json").is_file())
                self.assertTrue((output / "lensplit_manifest.csv").is_file())
            finally:
                if window._analysis_worker is not None:
                    window._analysis_worker.cancel()
                if window._export_worker is not None:
                    window._export_worker.cancel()
                with mock.patch.object(
                    QMessageBox, "question", return_value=QMessageBox.Discard
                ):
                    window.close()
                self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
