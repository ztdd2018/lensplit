import unittest
from unittest import mock

import numpy as np

from lensplit.analysis.person import (
    PersonDetectConfig,
    PersonModelError,
    YoloPersonDetector,
    count_persons_per_shot,
    decode_person_boxes,
)


def output_with(boxes):
    """Build a (1, 84, N) YOLOv8-style output from (cx, cy, w, h, score) rows."""

    matrix = np.zeros((84, len(boxes)), dtype=np.float32)
    for index, (cx, cy, width, height, score) in enumerate(boxes):
        matrix[0:4, index] = (cx, cy, width, height)
        matrix[4, index] = score
    return matrix[None, ...]


class PersonDecodeTests(unittest.TestCase):
    def test_person_box_is_mapped_back_to_source_pixels(self):
        output = output_with([(320.0, 320.0, 100.0, 200.0, 0.9)])
        boxes = decode_person_boxes(
            output, 1.0, 0.0, 0.0, (640, 640), 0.35, 0.45, 0.06
        )
        self.assertEqual(len(boxes), 1)
        np.testing.assert_allclose(boxes[0], [270.0, 220.0, 370.0, 420.0], atol=0.5)

    def test_low_score_and_tiny_boxes_are_dropped(self):
        output = output_with(
            [
                (320.0, 320.0, 100.0, 200.0, 0.20),
                (100.0, 100.0, 20.0, 10.0, 0.95),
            ]
        )
        boxes = decode_person_boxes(
            output, 1.0, 0.0, 0.0, (640, 640), 0.35, 0.45, 0.06
        )
        self.assertEqual(len(boxes), 0)

    def test_overlapping_boxes_are_suppressed(self):
        output = output_with(
            [
                (320.0, 320.0, 100.0, 200.0, 0.92),
                (322.0, 322.0, 104.0, 204.0, 0.90),
                (60.0, 320.0, 80.0, 180.0, 0.80),
            ]
        )
        boxes = decode_person_boxes(
            output, 1.0, 0.0, 0.0, (640, 640), 0.35, 0.45, 0.06
        )
        self.assertEqual(len(boxes), 2)

    def test_letterbox_padding_is_subtracted(self):
        output = output_with([(100.0, 100.0, 40.0, 80.0, 0.9)])
        boxes = decode_person_boxes(
            output, 0.5, 20.0, 40.0, (1280, 720), 0.35, 0.45, 0.06
        )
        self.assertEqual(len(boxes), 1)
        np.testing.assert_allclose(boxes[0], [120.0, 40.0, 200.0, 200.0], atol=0.5)


class PersonConfigTests(unittest.TestCase):
    def test_config_clamps_values(self):
        config = PersonDetectConfig(confidence=2.0, samples_per_shot=0, iou=-1)
        self.assertLessEqual(config.confidence, 0.99)
        self.assertEqual(config.samples_per_shot, 1)
        self.assertGreater(config.iou, 0.0)

    def test_missing_model_reports_unavailable(self):
        detector = YoloPersonDetector(PersonDetectConfig(model_path=""))
        self.assertFalse(detector.status.available)
        self.assertIn("未配置", detector.status.message)

        missing = YoloPersonDetector(
            PersonDetectConfig(model_path="models/does-not-exist.onnx")
        )
        self.assertTrue(missing.status.configured)
        self.assertFalse(missing.status.available)
        with self.assertRaises(PersonModelError):
            missing.count(np.zeros((32, 32, 3), dtype=np.uint8))

    def test_rejects_bad_frame_shape(self):
        detector = YoloPersonDetector(PersonDetectConfig(model_path=""))
        with self.assertRaises(ValueError):
            detector.count(np.zeros((32, 32), dtype=np.uint8))


class FakeDetector:
    def __init__(self, counts):
        self._counts = list(counts)
        self.calls = 0

    def count(self, image):
        value = self._counts[min(self.calls, len(self._counts) - 1)]
        self.calls += 1
        return value


class PersonShotSamplingTests(unittest.TestCase):
    def test_median_of_samples_is_reported_per_shot(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        progress = []
        shots = []
        with mock.patch(
            "lensplit.analysis.person.extract_thumbnail", return_value=frame
        ):
            result = count_persons_per_shot(
                "dummy.mp4",
                [("a", 0.0, 2.0), ("b", 2.0, 4.0)],
                PersonDetectConfig(samples_per_shot=3),
                detector=FakeDetector([1, 3, 3, 0]),
                progress=progress.append,
                on_shot=lambda group, count: shots.append((group, count)),
            )
        self.assertEqual(result, {"a": 3, "b": 0})
        self.assertEqual(shots, [("a", 3), ("b", 0)])
        self.assertEqual(progress, [0.5, 1.0])

    def test_undecodable_frames_do_not_break_the_run(self):
        from lensplit.video.decoder import VideoDecoderError

        with mock.patch(
            "lensplit.analysis.person.extract_thumbnail",
            side_effect=VideoDecoderError("bad frame"),
        ):
            result = count_persons_per_shot(
                "dummy.mp4",
                [("a", 0.0, 1.0)],
                PersonDetectConfig(samples_per_shot=2),
                detector=FakeDetector([2]),
            )
        self.assertEqual(result, {"a": 0})


if __name__ == "__main__":
    unittest.main()
