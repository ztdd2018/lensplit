import unittest

import numpy as np

from lensplit.analysis.fast import FastDetectionConfig, FastDetector
from lensplit.video.decoder import DecodedFrame


def make_frames(values, pts, durations=None):
    if durations is None:
        durations = [0.1] * len(values)
    return [
        DecodedFrame(
            index=index,
            pts=pts[index],
            duration=durations[index],
            image=np.full((12, 16, 3), value, dtype=np.uint8),
        )
        for index, value in enumerate(values)
    ]


class FastDetectorTests(unittest.TestCase):
    def test_hard_cuts_use_actual_vfr_pts_and_last_frame_duration(self):
        config = FastDetectionConfig(
            threshold=0.8,
            gradual_threshold=0.1,
            min_shot_seconds=0.0,
            analysis_size=(16, 12),
        )
        frames = make_frames(
            [0, 0, 255, 255],
            [0.0, 0.1, 0.7, 1.4],
            [0.1, 0.1, 0.2, 0.1],
        )

        project = FastDetector(config).detect_frames(frames)

        self.assertEqual(len(project.source_shots), 2)
        self.assertEqual(project.source_shots[0].start_pts, 0.0)
        self.assertEqual(project.source_shots[0].end_pts, 0.7)
        self.assertEqual(project.source_shots[1].start_pts, 0.7)
        self.assertAlmostEqual(project.source_shots[1].end_pts, 1.5)
        self.assertEqual(project.source_shots[1].start_frame, 2)
        self.assertEqual(project.source_shots[1].end_frame, 4)
        self.assertEqual(project.transitions[0].kind, "cut")
        self.assertEqual(project.transitions[0].engine, "fast")
        self.assertGreaterEqual(project.transitions[0].confidence, 0.0)
        self.assertLessEqual(project.transitions[0].confidence, 1.0)

    def test_accumulated_medium_changes_form_a_gradual_transition(self):
        config = FastDetectionConfig(
            threshold=0.9,
            gradual_threshold=0.05,
            min_shot_seconds=0.0,
            analysis_size=(16, 12),
        )
        frames = make_frames([0, 30, 60, 90, 120], [0, 1, 2, 3, 4])

        project = FastDetector(config).detect_frames(frames)

        self.assertEqual(len(project.source_shots), 2)
        self.assertEqual(project.transitions[0].kind, "gradual")
        self.assertEqual(project.transitions[0].cut_pts, 3.0)

    def test_dense_candidates_keep_the_stronger_change(self):
        config = FastDetectionConfig(
            threshold=0.4,
            gradual_threshold=0.05,
            min_shot_seconds=1.0,
            analysis_size=(16, 12),
        )
        frames = make_frames([0, 40, 255, 255], [0.0, 1.0, 1.1, 2.1])

        project = FastDetector(config).detect_frames(frames)

        self.assertEqual(len(project.source_shots), 2)
        self.assertEqual(project.transitions[0].cut_pts, 1.1)
        self.assertGreater(project.transitions[0].confidence, 0.85)

    def test_no_change_still_returns_one_complete_shot(self):
        frames = make_frames([80, 80, 80], [2.0, 2.2, 2.5], [0.2, 0.3, 0.4])

        project = FastDetector().detect_frames(frames)

        self.assertEqual(len(project.source_shots), 1)
        self.assertEqual(project.transitions, [])
        self.assertEqual(project.source_shots[0].start_pts, 2.0)
        self.assertAlmostEqual(project.source_shots[0].end_pts, 2.9)


if __name__ == "__main__":
    unittest.main()
