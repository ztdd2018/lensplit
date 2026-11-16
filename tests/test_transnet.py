import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lensplit.analysis.transnet import (
    TransNetConfig,
    TransNetDetector,
    TransNetModelError,
)
from lensplit.video.decoder import DecodedFrame


class FakeNode:
    def __init__(self, name, shape=None, input_type="tensor(float)"):
        self.name = name
        self.shape = shape
        self.type = input_type


class FakeSession:
    def __init__(
        self,
        shape=None,
        input_type="tensor(float)",
        multiple_outputs=False,
        fail_runtime=False,
    ):
        self.shape = shape or [1, 100, 2, 3, 3]
        self.input_type = input_type
        self.multiple_outputs = multiple_outputs
        self.fail_runtime = fail_runtime
        self.providers = ["CPUExecutionProvider"]
        self.calls = []

    def get_inputs(self):
        return [FakeNode("images", self.shape, self.input_type)]

    def get_outputs(self):
        if self.multiple_outputs:
            return [FakeNode("all_frame_output"), FakeNode("single_frame_output")]
        return [FakeNode("single_frame_output")]

    def get_providers(self):
        return self.providers

    def run(self, output_names, feed):
        batch = feed["images"]
        self.calls.append(batch.copy())
        if self.fail_runtime and len(self.calls) >= 2:
            raise RuntimeError("injected GPU runtime failure")
        single = batch[0, :, 0, 0, 0].astype(np.float32) / 255.0
        single = single.reshape(1, -1)
        if self.multiple_outputs:
            all_frames = np.ones_like(single)
            return [all_frames, single]
        return [single]


def numbered_frames(count, height=2, width=3):
    frames = np.empty((count, height, width, 3), dtype=np.uint8)
    for index in range(count):
        frames[index].fill(index + 1)
    return frames


class TransNetWindowTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.model_path = Path(self.temporary_directory.name) / "model.onnx"
        self.model_path.write_bytes(b"deterministic fake ONNX data")

    def make_detector(self, session, config=None):
        return TransNetDetector(
            config or TransNetConfig(model_path=self.model_path),
            session_factory=lambda path, providers: session,
            available_providers=["CPUExecutionProvider"],
        )

    def test_window_padding_and_output_lengths_cover_boundary_sizes(self):
        session = FakeSession()
        detector = self.make_detector(session)

        for count in [1, 25, 49, 50, 51, 99, 100, 101]:
            with self.subTest(count=count):
                probabilities = detector.predict_frames(numbered_frames(count))
                expected = (np.arange(count, dtype=np.float32) + 1) / 255.0
                self.assertEqual(probabilities.shape, (count,))
                np.testing.assert_allclose(probabilities, expected, atol=1e-6)

        first_real_window = session.calls[1]
        self.assertEqual(first_real_window.shape, (1, 100, 2, 3, 3))
        self.assertEqual(np.count_nonzero(first_real_window[0, :25]), 0)
        self.assertEqual(np.count_nonzero(first_real_window[0, 26:]), 0)
        self.assertEqual(np.count_nonzero(first_real_window), 18)
        self.assertEqual(first_real_window[0, 25, 0, 0, 0], 1)

    def test_dynamic_window_hw_and_float16_input_are_adapted(self):
        session = FakeSession(
            shape=["batch", "frames", 4, 5, 3], input_type="tensor(float16)"
        )
        config = TransNetConfig(
            window=6, overlap=2, model_path=self.model_path
        )
        detector = self.make_detector(session, config)

        probabilities = detector.predict_frames(numbered_frames(13, 4, 5))

        self.assertEqual(len(probabilities), 13)
        self.assertTrue(all(call.dtype == np.float16 for call in session.calls))
        np.testing.assert_allclose(
            probabilities, (np.arange(13) + 1) / 255.0, atol=1e-3
        )

    def test_single_frame_output_is_preferred_over_all_frame_output(self):
        session = FakeSession(multiple_outputs=True)
        detector = self.make_detector(session)

        probabilities = detector.predict_frames(numbered_frames(3))

        np.testing.assert_allclose(probabilities, [1 / 255, 2 / 255, 3 / 255])

    def test_model_status_hashes_local_file_and_unconfigured_never_falls_back(self):
        detector = self.make_detector(FakeSession())
        self.assertTrue(detector.status.available)
        self.assertEqual(
            detector.status.sha256, hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        )

        missing = TransNetDetector(available_providers=["CPUExecutionProvider"])
        self.assertFalse(missing.status.configured)
        with self.assertRaisesRegex(TransNetModelError, "No local"):
            missing.detect("not-opened.mp4")

    def test_adjacent_probability_peaks_merge_and_keep_the_maximum(self):
        detector = self.make_detector(
            FakeSession(),
            TransNetConfig(min_shot_seconds=0.0, model_path=self.model_path),
        )
        probabilities = np.array([0.1, 0.6, 0.8, 0.1, 0.7, 0.75, 0.2])
        pts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

        project = detector._probabilities_to_project(
            probabilities, pts, [0.1] * len(pts)
        )

        self.assertEqual(len(project.source_shots), 3)
        self.assertEqual([item.cut_pts for item in project.transitions], [0.2, 0.5])
        self.assertEqual([item.confidence for item in project.transitions], [0.8, 0.75])
        self.assertTrue(all(item.engine == "transnetv2" for item in project.transitions))


class TransNetCpuRestartTests(unittest.TestCase):
    def test_gpu_runtime_failure_restarts_decode_and_reruns_entire_input_on_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.onnx"
            model_path.write_bytes(b"fake")
            sessions = []

            def session_factory(path, providers):
                session = FakeSession(fail_runtime=providers[0] == "CUDAExecutionProvider")
                session.providers = providers
                sessions.append((providers, session))
                return session

            decoded_counts = []

            def fake_decode(path, size, cancel, progress):
                decoded_counts.append(0)
                for index in range(101):
                    decoded_counts[-1] += 1
                    yield DecodedFrame(
                        index=index,
                        pts=index * 0.04,
                        duration=0.04,
                        image=np.full((2, 3, 3), index + 1, dtype=np.uint8),
                    )

            warnings = []
            detector = TransNetDetector(
                TransNetConfig(model_path=model_path),
                session_factory=session_factory,
                available_providers=[
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ],
                warning=warnings.append,
            )
            with patch("lensplit.analysis.transnet.decode_frames", fake_decode):
                project = detector.detect("fake.mp4")

            self.assertEqual(decoded_counts, [75, 101])
            self.assertEqual(len(sessions), 2)
            self.assertEqual(sessions[0][0][0], "CUDAExecutionProvider")
            self.assertEqual(sessions[1][0], ["CPUExecutionProvider"])
            self.assertEqual(len(sessions[0][1].calls), 2)  # dummy, then failed real run
            self.assertGreaterEqual(len(sessions[1][1].calls), 4)  # dummy + full CPU rerun
            self.assertEqual(len(project.source_shots), 1)
            self.assertAlmostEqual(project.source_shots[0].end_pts, 4.04)
            self.assertTrue(any("restarting" in message for message in warnings))


if __name__ == "__main__":
    unittest.main()
