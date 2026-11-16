import json
import math
import random
import unittest
from pathlib import Path

from lensplit.core.model import LogicalGroup, ShotProject, SourceShot
from lensplit.core.numbering import MergeMode, format_groups, transition_statistics


def make_shots(count):
    return [
        SourceShot(
            uuid=f"s{ordinal}",
            ordinal=ordinal,
            start_pts=float(ordinal - 1),
            end_pts=float(ordinal),
        )
        for ordinal in range(1, count + 1)
    ]


class SourceShotTests(unittest.TestCase):
    def test_optional_frame_and_thumbnail_metadata(self):
        shot = SourceShot(
            uuid="s1",
            ordinal=1,
            start_pts=0,
            end_pts=1.5,
            start_frame=0,
            end_frame=36,
            thumbnail_path=Path("thumbs/s1.png"),
        )
        self.assertEqual(shot.start_pts, 0.0)
        self.assertEqual(shot.end_pts, 1.5)
        self.assertEqual(shot.start_frame, 0)
        self.assertEqual(shot.end_frame, 36)
        self.assertEqual(shot.thumbnail_path, str(Path("thumbs/s1.png")))


class ShotProjectTests(unittest.TestCase):
    def setUp(self):
        self.project = ShotProject(make_shots(5))

    def test_detection_state_has_single_groups_and_all_adjacent_transitions(self):
        self.assertEqual(
            [group.source_ids for group in self.project.groups],
            [[shot.uuid] for shot in self.project.source_shots],
        )
        self.assertEqual(len(self.project.transitions), 4)
        for index, transition in enumerate(self.project.transitions, start=1):
            self.assertEqual(transition.before_uuid, f"s{index}")
            self.assertEqual(transition.after_uuid, f"s{index + 1}")
            self.assertEqual(transition.cut_pts, float(index))

    def test_chain_merge_keeps_full_interval_and_internal_cuts(self):
        original_group_ids = [group.uuid for group in self.project.groups]
        merged = self.project.merge_groups(original_group_ids[:2])
        merged = self.project.merge_groups([merged.uuid, original_group_ids[2]])

        self.assertEqual(merged.source_ids, ["s1", "s2", "s3"])
        self.assertEqual(self.project.group_interval(merged.uuid), (0.0, 3.0))
        self.assertEqual(
            [transition.cut_pts for transition in self.project.transitions],
            [1.0, 2.0, 3.0, 4.0],
        )
        self.assertEqual(
            transition_statistics(self.project),
            {"total": 4, "internal": 2, "effective": 2},
        )

    def test_only_adjacent_logical_groups_can_be_merged(self):
        group_ids = [group.uuid for group in self.project.groups]
        before = self.project.snapshot()
        with self.assertRaises(ValueError):
            self.project.merge_groups([group_ids[0], group_ids[2]])
        self.assertEqual(self.project.snapshot(), before)

        merged = self.project.merge_groups(group_ids[:2])
        before = self.project.snapshot()
        with self.assertRaises(ValueError):
            self.project.merge_groups([merged.uuid, group_ids[4]])
        self.assertEqual(self.project.snapshot(), before)

    def test_split_restores_original_single_shot_groups(self):
        group_ids = [group.uuid for group in self.project.groups]
        merged = self.project.merge_groups(group_ids[:3])
        restored = self.project.split_group(merged.uuid)

        self.assertEqual(
            [group.source_ids for group in restored], [["s1"], ["s2"], ["s3"]]
        )
        self.assertEqual(
            [group.source_ids for group in self.project.groups],
            [[shot.uuid] for shot in self.project.source_shots],
        )
        self.assertEqual(len(self.project.transitions), 4)
        self.assertEqual(
            transition_statistics(self.project),
            {"total": 4, "internal": 0, "effective": 4},
        )

    def test_boundary_adjustment_updates_both_shots_and_transition(self):
        transition = self.project.transitions[0]
        transition.kind = "dissolve"
        transition.confidence = 0.42
        transition.engine = "fast"

        adjusted = self.project.adjust_boundary("s1", 1.5)
        self.assertEqual(self.project.source_shots[0].end_pts, 1.5)
        self.assertEqual(self.project.source_shots[1].start_pts, 1.5)
        self.assertIs(adjusted, transition)
        self.assertEqual(adjusted.cut_pts, 1.5)
        self.assertEqual(adjusted.kind, "dissolve")
        self.assertEqual(adjusted.confidence, 0.42)
        self.assertEqual(adjusted.engine, "fast")
        self.assertEqual(self.project.group_interval(self.project.groups[0].uuid), (0.0, 1.5))

        rebuilt = self.project.rebuild_transitions()
        self.assertEqual(rebuilt[0].kind, "dissolve")
        self.assertEqual(rebuilt[0].confidence, 0.42)
        self.assertEqual(rebuilt[0].engine, "fast")

    def test_invalid_boundary_adjustments_do_not_partially_mutate_state(self):
        invalid_cuts = [0, -1, math.nan, math.inf, 0.0, 2.0]
        for new_cut in invalid_cuts:
            with self.subTest(new_cut=new_cut):
                before = self.project.snapshot()
                with self.assertRaises(ValueError):
                    self.project.adjust_boundary("s1", new_cut)
                self.assertEqual(self.project.snapshot(), before)

        for source_uuid in ["missing", "s5"]:
            with self.subTest(source_uuid=source_uuid):
                before = self.project.snapshot()
                with self.assertRaises(ValueError):
                    self.project.adjust_boundary(source_uuid, 1.5)
                self.assertEqual(self.project.snapshot(), before)

    def test_json_snapshot_can_undo_merges_and_boundary_adjustments(self):
        before = self.project.snapshot()
        serialized = json.dumps(before)
        group_ids = [group.uuid for group in self.project.groups]
        self.project.merge_groups(group_ids[:3])
        self.project.adjust_boundary("s3", 3.5)
        self.assertNotEqual(self.project.snapshot(), before)

        self.project.restore(json.loads(serialized))
        self.assertEqual(self.project.snapshot(), before)
        restored_copy = ShotProject.from_snapshot(json.loads(serialized))
        self.assertEqual(restored_copy.snapshot(), before)


class RandomGroupingInvariantTests(unittest.TestCase):
    def test_random_contiguous_groups_preserve_numbering_and_transition_invariants(self):
        rng = random.Random(20261114)
        for count in range(1, 20):
            for _ in range(10):
                cuts = sorted(rng.sample(range(1, count), rng.randrange(count)))
                bounds = [0, *cuts, count]
                groups = []
                first_ordinals = []
                for group_index, (start, end) in enumerate(
                    zip(bounds, bounds[1:]), start=1
                ):
                    groups.append(
                        LogicalGroup(
                            uuid=f"g{group_index}",
                            source_ids=[f"s{index}" for index in range(start + 1, end + 1)],
                        )
                    )
                    first_ordinals.append(start + 1)
                project = ShotProject(make_shots(count), groups=groups)
                stats = transition_statistics(project)

                with self.subTest(count=count, cuts=cuts):
                    self.assertEqual(stats["total"], count - 1)
                    self.assertEqual(stats["internal"], count - len(groups))
                    self.assertEqual(stats["effective"], len(groups) - 1)
                    self.assertEqual(
                        stats["total"], stats["internal"] + stats["effective"]
                    )
                    self.assertEqual(
                        len(format_groups(groups, mode=MergeMode.COMBINED)), len(groups)
                    )
                    self.assertEqual(
                        format_groups(groups, mode=MergeMode.SEQUENTIAL),
                        [f"{index:03d}" for index in range(1, len(groups) + 1)],
                    )
                    self.assertEqual(
                        format_groups(groups, mode=MergeMode.FIRST),
                        [f"{ordinal:03d}" for ordinal in first_ordinals],
                    )
                    restored = ShotProject.from_snapshot(
                        json.loads(json.dumps(project.snapshot()))
                    )
                    self.assertEqual(transition_statistics(restored), stats)


class ManualSplitTests(unittest.TestCase):
    def test_cut_inside_shot_splits_group_and_renumbers(self):
        project = ShotProject(make_shots(3))
        labels_before = format_groups(project.groups)
        self.assertEqual(labels_before, ["001", "002", "003"])

        groups = project.split_group_at(project.groups[0].uuid, 0.5)

        self.assertEqual(len(groups), 2)
        self.assertEqual(len(project.groups), 4)
        self.assertEqual(len(project.source_shots), 4)
        self.assertEqual([shot.ordinal for shot in project.source_shots], [1, 2, 3, 4])
        self.assertEqual(format_groups(project.groups), ["001", "002", "003", "004"])
        manual = [
            transition
            for transition in project.transitions
            if transition.engine == "manual"
        ]
        self.assertEqual(len(manual), 1)
        self.assertAlmostEqual(manual[0].cut_pts, 0.5)
        self.assertEqual(len(project.transitions), 3)

    def test_cut_on_existing_boundary_only_splits_group(self):
        project = ShotProject(make_shots(3))
        shots = project.source_shots
        project.groups = [LogicalGroup("g1", [shots[0].uuid, shots[1].uuid])]
        project.groups.append(LogicalGroup("g2", [shots[2].uuid]))

        groups = project.split_group_at("g1", 1.0)

        self.assertEqual(len(groups), 2)
        self.assertEqual(len(project.source_shots), 3)
        self.assertEqual(groups[0].source_ids, [shots[0].uuid])
        self.assertEqual(groups[1].source_ids, [shots[1].uuid])

    def test_cut_outside_group_is_rejected(self):
        project = ShotProject(make_shots(3))
        with self.assertRaises(ValueError):
            project.split_group_at(project.groups[0].uuid, 1.5)

    def test_three_way_split_keeps_contiguous_coverage(self):
        project = ShotProject(make_shots(4))
        project.split_group_at(project.groups[0].uuid, 0.25)
        project.split_group_at(project.groups[1].uuid, 0.75)
        flattened = [
            source_id for group in project.groups for source_id in group.source_ids
        ]
        self.assertEqual(flattened, [shot.uuid for shot in project.source_shots])
        self.assertEqual(transition_statistics(project)["total"], 5)


if __name__ == "__main__":
    unittest.main()
