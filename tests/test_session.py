import json
import tempfile
import unittest
from pathlib import Path

from lensplit.core.model import ShotProject, SourceShot
from lensplit.core.session import PROJECT_SUFFIX, ProjectSession


def make_project():
    return ShotProject(
        [SourceShot(f"s{ordinal}", ordinal, ordinal - 1.0, float(ordinal)) for ordinal in range(1, 4)]
    )


class ProjectSessionTests(unittest.TestCase):
    def setUp(self):
        self.session = ProjectSession(video_path=r"C:\clips\demo.mp4", project=make_project())
        self.zhang = self.session.casting.add_actor("张三", "2026-11-01", "2026-11-05")
        self.group_id = self.session.project.groups[0].uuid
        self.session.casting.set_character_count(self.group_id, 2)
        self.session.casting.set_shoot_date(self.group_id, "2026-11-03")
        self.session.casting.assign_actor(
            self.session.casting.shot_slots(self.group_id)[0].uuid, self.zhang.uuid
        )
        self.session.transcripts = {self.group_id: "第一句台词"}
        self.session.analysis = {"engine": "fast"}
        self.session.numbering = {"start": 5, "prefix": "S"}

    def test_roundtrip_keeps_shots_casting_and_text(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.session.save(Path(folder) / "工程")
            self.assertTrue(path.name.endswith(PROJECT_SUFFIX))
            rebuilt = ProjectSession.load(path)

        self.assertEqual(rebuilt.video_path, self.session.video_path)
        self.assertEqual(len(rebuilt.project.groups), 3)
        self.assertEqual(rebuilt.transcripts[self.group_id], "第一句台词")
        self.assertEqual(rebuilt.analysis["engine"], "fast")
        self.assertEqual(rebuilt.numbering["start"], 5)
        self.assertEqual(rebuilt.numbering["merge_mode"], "combined")
        shot = rebuilt.casting.shot(self.group_id)
        self.assertEqual(shot.character_count, 2)
        self.assertEqual(shot.shoot_date, "2026-11-03")
        slots = rebuilt.casting.shot_slots(self.group_id)
        self.assertEqual(len(slots), 2)
        self.assertEqual(rebuilt.casting.actor(slots[0].actor_uuid).name, "张三")

    def test_unknown_terms_are_dropped_on_load(self):
        data = self.session.to_dict()
        data["transcripts"]["missing-group"] = "过期台词"
        data["casting"]["shots"].append(
            {"group_uuid": "missing-group", "character_count": 1, "shoot_date": ""}
        )
        rebuilt = ProjectSession.from_dict(data)
        self.assertNotIn("missing-group", rebuilt.transcripts)
        self.assertIsNone(rebuilt.casting.shot("missing-group"))

    def test_foreign_or_broken_files_are_rejected(self):
        with self.assertRaises(ValueError):
            ProjectSession.from_dict({"app": "other"})
        with tempfile.TemporaryDirectory() as folder:
            broken = Path(folder) / "broken.lensplit"
            broken.write_text("{not json", encoding="utf-8")
            with self.assertRaises(ValueError):
                ProjectSession.load(broken)

    def test_newer_version_is_rejected(self):
        data = self.session.to_dict()
        data["version"] = 99
        with self.assertRaises(ValueError):
            ProjectSession.from_dict(data)

    def test_saved_file_is_readable_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.session.save(Path(folder) / "工程.lensplit")
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["app"], "lensplit")
        self.assertEqual(len(data["project"]["groups"]), 3)


if __name__ == "__main__":
    unittest.main()
