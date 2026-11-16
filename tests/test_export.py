import csv
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from lensplit.core.model import LogicalGroup, ShotProject, SourceShot
from lensplit.core.numbering import NumberingStyle
from lensplit.export import (
    ExportCancelledError,
    ExportFailedError,
    ExportOptions,
    ExportWorker,
    FFmpegExporter,
    FFmpegNotFoundError,
    sanitize_filename,
)


class FakeProcess:
    def __init__(self, command, *, returncode=0, output="", waiting=False):
        self.command = command
        self.returncode = None if waiting else returncode
        self.stdout = io.StringIO(output)
        self.output = output
        self.waiting = waiting
        self.terminated = False
        self.killed = False

    def communicate(self, timeout=None):
        return self.output, None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None and timeout is not None:
            raise subprocess.TimeoutExpired(self.command, timeout)
        if self.returncode is None:
            raise AssertionError("A waiting fake process must be terminated first")
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -1

    def kill(self):
        self.killed = True
        self.returncode = -1


class FakeFFmpegRunner:
    def __init__(self, *, trials=None, encode_returncode=0, waiting=False):
        self.trials = trials or {}
        self.encode_returncode = encode_returncode
        self.waiting = waiting
        self.commands = []
        self.processes = []

    def __call__(self, command, **kwargs):
        self.commands.append(list(command))
        self.last_kwargs = kwargs
        if command[1:] == ["-hide_banner", "-encoders"]:
            output = (
                "Encoders:\n"
                " V..... h264_nvenc           NVIDIA NVENC H.264\n"
                " V..... h264_qsv             Intel Quick Sync Video H.264\n"
                " V..... libx264              libx264 H.264\n"
            )
            process = FakeProcess(command, output=output)
        elif "color=c=black:s=64x64:r=25:d=0.04" in command:
            encoder = command[command.index("-c:v") + 1]
            returncode = self.trials.get(encoder, 0)
            output = "" if returncode == 0 else f"{encoder}: simulated device failure\n"
            process = FakeProcess(command, returncode=returncode, output=output)
        elif "-progress" in command:
            part = Path(command[-1])
            part.write_bytes(b"fake mp4")
            duration = float(command[command.index("-t") + 1])
            output = (
                f"out_time_ms={int(duration * 500000)}\n"
                f"out_time=00:00:{duration / 2:09.6f}\n"
                "progress=continue\n"
            )
            process = FakeProcess(
                command,
                returncode=self.encode_returncode,
                output=output + ("simulated encode failure\n" if self.encode_returncode else ""),
                waiting=self.waiting,
            )
        else:
            raise AssertionError(f"Unexpected fake FFmpeg command: {command}")
        self.processes.append(process)
        return process


class FilenameTests(unittest.TestCase):
    def test_windows_safety_preserves_chinese(self):
        self.assertEqual(sanitize_filename('镜头:一/二*?"<>|'), "镜头_一_二______")
        self.assertEqual(sanitize_filename("name. \x00"), "name. _")
        self.assertEqual(sanitize_filename("CON"), "_CON")
        self.assertEqual(sanitize_filename("com1.txt"), "_com1.txt")
        self.assertEqual(sanitize_filename("...  "), "shot")


class ExportUnitTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.video = self.root / "input.mp4"
        self.video.write_bytes(b"fake source")
        self.ffmpeg = self.root / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        self.ffprobe = self.root / ("ffprobe.exe" if os.name == "nt" else "ffprobe")
        self.ffmpeg.write_bytes(b"fake")
        self.ffprobe.write_bytes(b"fake")
        self.output = self.root / "output"
        self.project = ShotProject(
            source_shots=[
                SourceShot("s1", 1, 0.0, 1.0),
                SourceShot("s2", 2, 1.0, 2.0),
                SourceShot("s3", 3, 2.0, 3.0),
            ],
            groups=[LogicalGroup("group-a", ["s1", "s2"]), LogicalGroup("group-b", ["s3"])],
        )
        self.project.transitions[0].kind = "dissolve"
        self.project.transitions[0].confidence = 0.7

    def exporter(self, runner):
        return FFmpegExporter(
            self.video,
            self.ffmpeg,
            self.ffprobe,
            process_runner=runner,
        )

    def test_logical_ranges_numbering_collision_and_manifest_data(self):
        self.output.mkdir()
        (self.output / "镜头_001+002.mp4").write_bytes(b"existing")
        (self.output / "lensplit_manifest.json").write_text("{}", encoding="utf-8")
        (self.output / "lensplit_manifest.csv").write_text("old", encoding="utf-8")
        runner = FakeFFmpegRunner()
        progress = []
        options = ExportOptions(
            self.output,
            encoder="cpu",
            numbering_style=NumberingStyle(prefix="镜头:"),
            analysis_metadata={"模型": {"hash": "abc123"}, "threshold": 0.25},
            transcripts={"group-a": "今天先试拍一条", "group-b": ""},
        )

        report = self.exporter(runner).export_project(
            self.project, options, progress=progress.append
        )

        self.assertEqual(report.actual_encoder, "libx264")
        self.assertEqual(len(report.records), 2)
        first, second = report.records
        self.assertEqual(first.group_uuid, "group-a")
        self.assertIsNone(first.source_uuid)
        self.assertEqual(first.source_ids, ["s1", "s2"])
        self.assertEqual(first.source_ordinals, [1, 2])
        self.assertEqual((first.start_pts, first.end_pts), (0.0, 2.0))
        self.assertEqual(first.internal_cuts, [1.0])
        self.assertEqual(second.internal_cuts, [])
        self.assertEqual(Path(first.output_path).name, "镜头_001+002_1.mp4")
        self.assertEqual(Path(second.output_path).name, "镜头_003.mp4")
        self.assertTrue(all(Path(path).is_file() for path in report.output_files))
        self.assertEqual(list(self.output.glob("*.part.mp4")), [])
        self.assertEqual(Path(report.manifest_json_path).name, "lensplit_manifest_1.json")
        self.assertEqual(Path(report.manifest_csv_path).name, "lensplit_manifest_1.csv")
        self.assertEqual(progress[0], 0.0)
        self.assertEqual(progress[-1], 1.0)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in progress))
        self.assertTrue(all(a <= b for a, b in zip(progress, progress[1:])))

        command = next(command for command in runner.commands if "-progress" in command)
        self.assertLess(command.index("-ss"), command.index("-i"))
        self.assertIn("-accurate_seek", command)
        self.assertEqual(command[command.index("-t") + 1], "2")
        self.assertEqual(command[command.index("-c:v") + 1], "libx264")
        self.assertEqual(command[command.index("-c:a") + 1], "aac")
        self.assertIn("0:a:0?", command)
        self.assertIn("+faststart", command)
        self.assertEqual(runner.last_kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(runner.last_kwargs["stderr"], subprocess.STDOUT)

        json_text = Path(report.manifest_json_path).read_text(encoding="utf-8")
        self.assertIn("模型", json_text)
        self.assertNotIn("\\u6a21\\u578b", json_text)
        manifest = json.loads(json_text)
        self.assertEqual(manifest["analysis_metadata"]["模型"]["hash"], "abc123")
        self.assertEqual(
            manifest["transition_statistics"],
            {"total": 2, "internal": 1, "effective": 1},
        )
        self.assertEqual(manifest["records"][0]["internal_cuts"], [1.0])
        self.assertEqual(manifest["records"][0]["source_ids"], ["s1", "s2"])
        self.assertEqual(manifest["records"][0]["group_uuid"], "group-a")
        self.assertEqual(manifest["transitions"][0]["kind"], "dissolve")
        self.assertEqual(manifest["export_parameters"]["actual_encoder"], "libx264")

        with Path(report.manifest_csv_path).open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["source_ids"], "s1|s2")
        self.assertEqual(rows[0]["source_ordinals"], "1|2")
        self.assertEqual(rows[0]["internal_cuts"], "1.000000000")
        self.assertEqual(rows[0]["actual_encoder"], "libx264")
        self.assertEqual(rows[0]["transcript"], "今天先试拍一条")
        self.assertEqual(manifest["transcripts"]["group-a"], "今天先试拍一条")

    def test_source_scope_ignores_merged_groups_and_keeps_original_ordinals(self):
        runner = FakeFFmpegRunner()
        options = ExportOptions(
            self.output,
            scope="source",
            encoder="cpu",
            numbering_style=NumberingStyle(start=4, width=2, prefix="原始"),
        )

        report = self.exporter(runner).export_project(self.project, options)

        self.assertEqual([record.display_number for record in report.records], ["原始04", "原始05", "原始06"])
        self.assertEqual([record.source_uuid for record in report.records], ["s1", "s2", "s3"])
        self.assertTrue(all(record.group_uuid is None for record in report.records))
        self.assertEqual([record.source_ordinals for record in report.records], [[1], [2], [3]])
        self.assertEqual([(record.start_pts, record.end_pts) for record in report.records], [(0, 1), (1, 2), (2, 3)])
        self.assertTrue(all(record.internal_cuts == [] for record in report.records))

    def test_auto_trials_listed_hardware_and_falls_back_to_libx264(self):
        runner = FakeFFmpegRunner(trials={"h264_nvenc": 1, "h264_qsv": 1})
        callback_warnings = []

        report = self.exporter(runner).export_project(
            self.project,
            ExportOptions(self.output, encoder="auto"),
            warning=callback_warnings.append,
        )

        self.assertEqual(report.actual_encoder, "libx264")
        trials = [command for command in runner.commands if "-frames:v" in command]
        self.assertEqual(len(trials), 2)
        self.assertTrue(all(command[command.index("-frames:v") + 1] == "1" for command in trials))
        self.assertTrue(any("listed but its one-frame trial failed" in item for item in report.warnings))
        self.assertTrue(any("h264_amf is not advertised" in item for item in report.warnings))
        self.assertEqual(report.warnings, callback_warnings)
        self.assertTrue(all(record.encoder == "libx264" for record in report.records))
        self.assertEqual(list(self.output.glob("*.part.mp4")), [])

    def test_cancel_terminates_ffmpeg_and_removes_partial_output(self):
        runner = FakeFFmpegRunner(waiting=True)
        progress = []

        with self.assertRaises(ExportCancelledError):
            self.exporter(runner).export_project(
                self.project,
                ExportOptions(self.output, encoder="cpu"),
                progress=progress.append,
                cancel=lambda: any(value > 0 for value in progress),
            )

        self.assertTrue(any(value > 0 for value in progress))
        self.assertTrue(runner.processes[-1].terminated)
        self.assertEqual(list(self.output.glob("*.part.mp4")), [])
        self.assertEqual(list(self.output.glob("*.mp4")), [])
        self.assertEqual(list(self.output.glob("lensplit_manifest*")), [])

    def test_failed_encode_removes_partial_output_and_writes_no_manifest(self):
        runner = FakeFFmpegRunner(encode_returncode=1)

        with self.assertRaisesRegex(ExportFailedError, "simulated encode failure"):
            self.exporter(runner).export_project(
                self.project, ExportOptions(self.output, encoder="cpu")
            )

        self.assertEqual(list(self.output.glob("*.part.mp4")), [])
        self.assertEqual(list(self.output.glob("*.mp4")), [])
        self.assertEqual(list(self.output.glob("lensplit_manifest*")), [])

    def test_missing_explicit_tool_has_a_clear_error(self):
        with self.assertRaisesRegex(FFmpegNotFoundError, "Explicit ffmpeg path"):
            FFmpegExporter(
                self.video,
                self.root / "missing-ffmpeg",
                self.ffprobe,
                process_runner=FakeFFmpegRunner(),
            ).export_project(self.project, ExportOptions(self.output, encoder="cpu"))

    def test_worker_transfers_result_through_future(self):
        one_shot = ShotProject([SourceShot("only", 1, 0.0, 1.0)])
        worker = ExportWorker(
            self.exporter(FakeFFmpegRunner()),
            one_shot,
            ExportOptions(self.output, encoder="cpu"),
        ).start()

        report = worker.result(timeout=5)
        worker.join(timeout=1)

        self.assertEqual(len(report.records), 1)
        self.assertFalse(worker.is_running())


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools are not installed")
class RealFFmpegIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary_directory.name)
        cls.ffmpeg = shutil.which("ffmpeg")
        cls.ffprobe = shutil.which("ffprobe")
        cls.video = cls.root / "sample.mp4"
        subprocess.run(
            [
                cls.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=160x90:rate=25:duration=2.4",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-g",
                "5",
                "-keyint_min",
                "5",
                "-sc_threshold",
                "0",
                str(cls.video),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary_directory.cleanup()

    def project(self):
        return ShotProject(
            source_shots=[
                SourceShot("real-1", 1, 0.0, 0.8),
                SourceShot("real-2", 2, 0.8, 1.6),
                SourceShot("real-3", 3, 1.6, 2.4),
            ],
            groups=[LogicalGroup("real-group-a", ["real-1", "real-2"]), LogicalGroup("real-group-b", ["real-3"])],
        )

    def exporter(self):
        return FFmpegExporter(self.video, self.ffmpeg, self.ffprobe)

    def probe(self, path):
        result = subprocess.run(
            [
                self.ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_type,codec_name",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return json.loads(result.stdout)

    def assert_decodes_without_audio(self, record):
        probe = self.probe(record.output_path)
        self.assertIn("video", [stream["codec_type"] for stream in probe["streams"]])
        self.assertNotIn("audio", [stream["codec_type"] for stream in probe["streams"]])
        duration = float(probe["format"]["duration"])
        self.assertGreater(duration, 0.0)
        subprocess.run(
            [self.ffmpeg, "-v", "error", "-i", record.output_path, "-f", "null", "-"],
            check=True,
            capture_output=True,
            timeout=30,
        )
        return duration

    def test_real_exact_export_is_frame_accurate_and_decodable_without_audio(self):
        report = self.exporter().export_project(
            self.project(),
            ExportOptions(self.root / "exact", encoder="cpu"),
        )

        self.assertEqual(report.actual_encoder, "libx264")
        for record in report.records:
            with self.subTest(output=record.output_path):
                duration = self.assert_decodes_without_audio(record)
                self.assertAlmostEqual(duration, record.duration, delta=0.12)
                probe = self.probe(record.output_path)
                video = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
                self.assertEqual(video["codec_name"], "h264")

    def test_real_copy_export_decodes_and_warns_about_keyframes(self):
        report = self.exporter().export_project(
            self.project(),
            ExportOptions(self.root / "copy", scope="source", mode="copy"),
        )

        self.assertEqual(report.actual_encoder, "copy")
        self.assertTrue(any("keyframe-aligned" in warning for warning in report.warnings))
        for record in report.records:
            with self.subTest(output=record.output_path):
                duration = self.assert_decodes_without_audio(record)
                self.assertLessEqual(duration, record.duration + 1.0)
                self.assertLess(duration, 3.0)


if __name__ == "__main__":
    unittest.main()
