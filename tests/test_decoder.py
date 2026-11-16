import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from lensplit.video.decoder import (
    DecodeCancelledError,
    VideoDecoderError,
    decode_frames,
    extract_thumbnail,
    probe_video,
)


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.path = Path(self.temporary_directory.name) / "vfr.mp4"
        self.source_pts = [0, 40, 80, 200, 360, 420]
        self._write_vfr_video()

    def _write_vfr_video(self):
        container = av.open(str(self.path), mode="w", format="mp4")
        try:
            stream = container.add_stream("mpeg4", rate=25)
            stream.width = 64
            stream.height = 48
            stream.pix_fmt = "yuv420p"
            stream.time_base = Fraction(1, 1000)
            stream.codec_context.time_base = Fraction(1, 1000)
            for index, pts in enumerate(self.source_pts):
                pixels = np.full((48, 64, 3), 20 + index * 35, dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                frame.pts = pts
                frame.time_base = Fraction(1, 1000)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()

    def test_decode_uses_frame_pts_and_only_shrinks_rgb_images(self):
        info = probe_video(self.path)
        progress = []

        frames = list(decode_frames(self.path, size=(32, 24), progress=progress.append))

        self.assertEqual((info.width, info.height), (64, 48))
        self.assertEqual([frame.index for frame in frames], list(range(6)))
        self.assertEqual(frames[0].image.shape, (24, 32, 3))
        for frame, expected_ticks in zip(frames, self.source_pts):
            self.assertAlmostEqual(frame.pts, expected_ticks / 1000.0, places=6)
            self.assertGreater(frame.duration, 0.0)
        self.assertEqual(progress[0], 0.0)
        self.assertEqual(progress[-1], 1.0)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in progress))

    def test_decode_does_not_upscale(self):
        frames = list(decode_frames(self.path, size=(128, 96)))
        self.assertEqual(frames[0].image.shape, (48, 64, 3))

    def test_cancel_callback_stops_iteration(self):
        checks = 0

        def cancel():
            nonlocal checks
            checks += 1
            return checks > 3

        with self.assertRaises(DecodeCancelledError):
            list(decode_frames(self.path, cancel=cancel))
        self.assertGreater(checks, 1)

    def test_pts_seek_returns_the_containing_rgb_thumbnail(self):
        frames = list(decode_frames(self.path, size=(32, 24)))

        thumbnail = extract_thumbnail(self.path, 0.21, size=(32, 24))

        self.assertEqual(thumbnail.shape, (24, 32, 3))
        difference = np.mean(np.abs(thumbnail.astype(float) - frames[3].image.astype(float)))
        self.assertLess(difference, 2.0)

    def test_damaged_file_has_a_clear_decoder_error(self):
        damaged = Path(self.temporary_directory.name) / "damaged.mp4"
        damaged.write_bytes(b"not a video")
        with self.assertRaisesRegex(VideoDecoderError, "Cannot open video"):
            probe_video(damaged)


if __name__ == "__main__":
    unittest.main()
