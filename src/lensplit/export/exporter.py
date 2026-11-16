"""Local, atomic FFmpeg export with manifests and hardware fallback."""

from __future__ import annotations

import csv
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from lensplit.core.model import ShotProject, SourceShot
from lensplit.core.numbering import (
    MergeMode,
    NumberingStyle,
    format_groups,
    format_ordinal,
    transition_statistics,
)

PathLike = str | os.PathLike[str]
ProgressCallback = Callable[[float], None]
CancelCallback = Callable[[], bool]
WarningCallback = Callable[[str], None]

_HARDWARE_ENCODERS = ("h264_nvenc", "h264_qsv", "h264_amf")
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_INVALID_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_ENCODER_LINE = re.compile(r"^\s*[A-Z.]{6}\s+([A-Za-z0-9_]+)\s", re.MULTILINE)
_EOF = object()


class ExportError(RuntimeError):
    """Base error for local export failures."""


class FFmpegNotFoundError(ExportError):
    """Raised when ffmpeg or ffprobe cannot be resolved locally."""


class ExportFailedError(ExportError):
    """Raised when FFmpeg does not produce a requested output."""


class ExportCancelledError(ExportError):
    """Raised after cancellation has stopped the active FFmpeg process."""


@dataclass
class ExportOptions:
    """Options needed by the UI to export one complete project."""

    output_dir: PathLike
    scope: str = "logical"
    mode: str = "exact"
    encoder: str = "auto"
    numbering_style: NumberingStyle = field(default_factory=NumberingStyle)
    merge_mode: MergeMode | str = MergeMode.COMBINED
    analysis_metadata: dict[str, Any] = field(default_factory=dict)
    transcripts: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.output_dir = str(self.output_dir)
        self.scope = str(self.scope).strip().lower()
        self.mode = str(self.mode).strip().lower()
        self.encoder = str(self.encoder).strip().lower()
        if self.scope not in {"logical", "source"}:
            raise ValueError("Export scope must be 'logical' or 'source'")
        if self.mode not in {"exact", "copy"}:
            raise ValueError("Export mode must be 'exact' or 'copy'")
        if not self.encoder:
            raise ValueError("Export encoder cannot be empty")
        if not isinstance(self.numbering_style, NumberingStyle):
            raise TypeError("numbering_style must be a NumberingStyle")
        self.merge_mode = MergeMode(self.merge_mode)
        if not isinstance(self.analysis_metadata, dict):
            raise TypeError("analysis_metadata must be a dictionary")
        self.analysis_metadata = dict(self.analysis_metadata)
        if not isinstance(self.transcripts, dict):
            raise TypeError("transcripts must be a dictionary")
        self.transcripts = {
            str(key): str(value) for key, value in self.transcripts.items()
        }


@dataclass(frozen=True)
class ExportRecord:
    """Stable project identifiers and actual output details for one interval."""

    display_number: str
    group_uuid: str | None
    source_uuid: str | None
    source_ids: list[str]
    source_ordinals: list[int]
    start_pts: float
    end_pts: float
    internal_cuts: list[float]
    output_path: str
    encoder: str

    @property
    def label(self) -> str:
        return self.display_number

    @property
    def actual_encoder(self) -> str:
        return self.encoder

    @property
    def duration(self) -> float:
        return self.end_pts - self.start_pts

    def to_dict(self) -> dict[str, Any]:
        return {
            "display_number": self.display_number,
            "label": self.display_number,
            "group_uuid": self.group_uuid,
            "source_uuid": self.source_uuid,
            "source_ids": list(self.source_ids),
            "source_ordinals": list(self.source_ordinals),
            "start_pts": self.start_pts,
            "end_pts": self.end_pts,
            "duration_pts": self.duration,
            "internal_cuts": list(self.internal_cuts),
            "output_path": self.output_path,
            "encoder": self.encoder,
            "actual_encoder": self.encoder,
        }


@dataclass(frozen=True)
class ExportReport:
    """Result returned after every output and both manifests succeed."""

    records: list[ExportRecord]
    output_files: list[str]
    actual_encoder: str
    warnings: list[str]
    elapsed: float
    manifest_json_path: str
    manifest_csv_path: str

    @property
    def json_manifest_path(self) -> str:
        return self.manifest_json_path

    @property
    def csv_manifest_path(self) -> str:
        return self.manifest_csv_path


@dataclass
class _ExportItem:
    display_number: str
    group_uuid: str | None
    source_uuid: str | None
    shots: list[SourceShot]
    start_pts: float
    end_pts: float
    internal_cuts: list[float]


class _PathAllocator:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.reserved = {path.name.casefold() for path in directory.iterdir()}

    def output(self, label: str) -> tuple[Path, Path]:
        stem = sanitize_filename(label)
        index = 0
        while True:
            suffix = "" if index == 0 else f"_{index}"
            target = self.directory / f"{stem}{suffix}.mp4"
            part = self.directory / f"{stem}{suffix}.part.mp4"
            names = {target.name.casefold(), part.name.casefold()}
            if not names & self.reserved and not target.exists() and not part.exists():
                self.reserved.update(names)
                return target, part
            index += 1

    def manifests(self) -> tuple[Path, Path]:
        index = 0
        while True:
            suffix = "" if index == 0 else f"_{index}"
            json_path = self.directory / f"lensplit_manifest{suffix}.json"
            csv_path = self.directory / f"lensplit_manifest{suffix}.csv"
            temporary = [
                json_path.with_name(json_path.name + ".part"),
                csv_path.with_name(csv_path.name + ".part"),
            ]
            paths = [json_path, csv_path, *temporary]
            names = {path.name.casefold() for path in paths}
            if not names & self.reserved and not any(path.exists() for path in paths):
                self.reserved.update(names)
                return json_path, csv_path
            index += 1


def sanitize_filename(name: object) -> str:
    """Make one Windows-safe filename component without removing Unicode."""

    cleaned = _INVALID_FILENAME.sub("_", str(name)).rstrip(" .")
    if not cleaned:
        cleaned = "shot"
    if cleaned.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
        cleaned = f"_{cleaned}"
    return cleaned


def _executable_name(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


def _usable_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.R_OK)


def _resolve_executable(
    explicit: PathLike | None,
    *,
    binary: str,
    environment_names: tuple[str, ...],
) -> str:
    filename = _executable_name(binary)

    def from_value(value: PathLike, origin: str) -> str:
        candidate = Path(value).expanduser()
        if candidate.is_dir():
            candidate = candidate / filename
        if _usable_executable(candidate):
            return str(candidate.resolve())
        if not candidate.is_absolute() and candidate.parent == Path("."):
            located = shutil.which(str(value))
            if located:
                return str(Path(located).resolve())
        raise FFmpegNotFoundError(
            f"{origin} does not point to a readable {binary} executable: '{value}'"
        )

    if explicit is not None:
        return from_value(explicit, f"Explicit {binary} path")
    for variable in environment_names:
        value = os.environ.get(variable)
        if value:
            return from_value(value, f"Environment variable {variable}")

    package_root = Path(__file__).resolve().parents[1]
    roots = [
        package_root,
        package_root.parent,
        package_root.parent.parent,
        Path.cwd(),
    ]
    bundled_root = getattr(sys, "_MEIPASS", None)
    if bundled_root:
        roots.insert(0, Path(bundled_root))
    for root in roots:
        for candidate in (
            root / "resources" / "bin" / filename,
            root / "bin" / filename,
        ):
            if _usable_executable(candidate):
                return str(candidate.resolve())

    located = shutil.which(binary)
    if located:
        return str(Path(located).resolve())
    variables = " or ".join(environment_names)
    raise FFmpegNotFoundError(
        f"Cannot find {binary}. Set {variables}, place {filename} in resources/bin, "
        f"or add it to PATH."
    )


def _seconds(value: float) -> str:
    return f"{value:.9f}".rstrip("0").rstrip(".") or "0"


def _progress_seconds(value: str) -> float | None:
    try:
        pieces = value.strip().split(":")
        if len(pieces) != 3:
            return None
        result = int(pieces[0]) * 3600 + int(pieces[1]) * 60 + float(pieces[2])
        return max(0.0, result)
    except (TypeError, ValueError):
        return None


def _json_default(value: object) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return str(value)


def _remove_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


class FFmpegExporter:
    """Export local video intervals without sending media or metadata anywhere."""

    def __init__(
        self,
        video_path: PathLike,
        ffmpeg_path: PathLike | None = None,
        ffprobe_path: PathLike | None = None,
        *,
        process_runner: Callable[..., Any] | None = None,
    ) -> None:
        self.video_path = str(Path(video_path).expanduser())
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self._process_runner = process_runner or subprocess.Popen

    def _resolve_tools(self) -> tuple[str, str]:
        ffmpeg = _resolve_executable(
            self.ffmpeg_path,
            binary="ffmpeg",
            environment_names=(
                "LENSPLIT_FFMPEG",
                "LENSPLIT_FFMPEG_PATH",
                "FFMPEG_PATH",
                "FFMPEG",
            ),
        )
        ffprobe = _resolve_executable(
            self.ffprobe_path,
            binary="ffprobe",
            environment_names=(
                "LENSPLIT_FFPROBE",
                "LENSPLIT_FFPROBE_PATH",
                "FFPROBE_PATH",
                "FFPROBE",
            ),
        )
        self.ffmpeg_path = ffmpeg
        self.ffprobe_path = ffprobe
        return ffmpeg, ffprobe

    def _popen(self, command: list[str]) -> Any:
        try:
            return self._process_runner(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except (OSError, ValueError) as error:
            raise ExportFailedError(
                f"Cannot start local FFmpeg process '{command[0]}': {error}"
            ) from error

    @staticmethod
    def _stop_process(process: Any, timeout: float = 3.0) -> None:
        if process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass

    def _run_capture(self, command: list[str], timeout: float = 30.0) -> tuple[int, str]:
        process = self._popen(command)
        try:
            output, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            self._stop_process(process)
            raise ExportFailedError(
                f"FFmpeg did not respond within {timeout:g} seconds"
            ) from error
        finally:
            if process.stdout is not None:
                process.stdout.close()
        return int(process.returncode or 0), output or ""

    def _select_encoder(self, requested: str, warn: WarningCallback) -> str:
        if requested == "cpu":
            return "libx264"
        if requested != "auto":
            return requested

        returncode, output = self._run_capture(
            [str(self.ffmpeg_path), "-hide_banner", "-encoders"]
        )
        if returncode != 0:
            warn(
                "Cannot query FFmpeg encoders; automatic hardware encoding is "
                "unavailable. Falling back to libx264."
            )
            return "libx264"
        listed = set(_ENCODER_LINE.findall(output))
        for encoder in _HARDWARE_ENCODERS:
            if encoder not in listed:
                warn(f"{encoder} is not advertised by FFmpeg; trying the next encoder.")
                continue
            command = [
                str(self.ffmpeg_path),
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=64x64:r=25:d=0.04",
                "-frames:v",
                "1",
                "-an",
                "-c:v",
                encoder,
                "-pix_fmt",
                "yuv420p",
                "-f",
                "null",
                "-",
            ]
            try:
                trial_code, trial_output = self._run_capture(command, timeout=20.0)
            except ExportFailedError as error:
                warn(f"{encoder} one-frame trial failed: {error}; trying the next encoder.")
                continue
            if trial_code == 0:
                return encoder
            detail = trial_output.strip().splitlines()
            suffix = f" ({detail[-1]})" if detail else ""
            warn(
                f"{encoder} is listed but its one-frame trial failed{suffix}; "
                "trying the next encoder."
            )
        warn("No usable hardware H.264 encoder was found; falling back to libx264.")
        return "libx264"

    def _build_items(self, project: ShotProject, options: ExportOptions) -> list[_ExportItem]:
        shots = {shot.uuid: shot for shot in project.source_shots}
        transitions = {
            (transition.before_uuid, transition.after_uuid): transition
            for transition in project.transitions
        }

        def cuts_for(source_ids: list[str]) -> list[float]:
            return [
                transitions[(before, after)].cut_pts
                for before, after in zip(source_ids, source_ids[1:])
                if (before, after) in transitions
            ]

        if options.scope == "source":
            return [
                _ExportItem(
                    display_number=format_ordinal(shot.ordinal, options.numbering_style),
                    group_uuid=None,
                    source_uuid=shot.uuid,
                    shots=[shot],
                    start_pts=shot.start_pts,
                    end_pts=shot.end_pts,
                    internal_cuts=[],
                )
                for shot in project.source_shots
            ]

        labels = format_groups(
            project.groups, options.numbering_style, options.merge_mode
        )
        items: list[_ExportItem] = []
        for group, label in zip(project.groups, labels):
            group_shots = [shots[source_id] for source_id in group.source_ids]
            start, end = project.group_interval(group.uuid)
            items.append(
                _ExportItem(
                    display_number=label,
                    group_uuid=group.uuid,
                    source_uuid=None,
                    shots=group_shots,
                    start_pts=start,
                    end_pts=end,
                    internal_cuts=cuts_for(group.source_ids),
                )
            )
        return items

    def _command(self, item: _ExportItem, options: ExportOptions, encoder: str, part: Path) -> list[str]:
        command = [
            str(self.ffmpeg_path),
            "-hide_banner",
            "-nostdin",
            "-y",
            "-ss",
            _seconds(item.start_pts),
            "-accurate_seek",
            "-i",
            str(Path(self.video_path).resolve()),
            "-t",
            _seconds(item.end_pts - item.start_pts),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-sn",
            "-dn",
        ]
        if options.mode == "copy":
            command.extend(["-c", "copy"])
        else:
            command.extend(
                ["-c:v", encoder, "-pix_fmt", "yuv420p", "-c:a", "aac"]
            )
        command.extend(
            [
                "-movflags",
                "+faststart",
                "-avoid_negative_ts",
                "make_zero",
                "-progress",
                "pipe:1",
                "-nostats",
                "-f",
                "mp4",
                str(part),
            ]
        )
        return command

    def _run_encode(
        self,
        command: list[str],
        *,
        duration: float,
        shot_index: int,
        shot_count: int,
        progress: ProgressCallback | None,
        cancel: CancelCallback | None,
    ) -> None:
        process = self._popen(command)
        if process.stdout is None:
            self._stop_process(process)
            raise ExportFailedError("FFmpeg progress pipe could not be opened")
        lines: queue.Queue[object] = queue.Queue()

        def read_output() -> None:
            try:
                for line in process.stdout:
                    lines.put(line)
            except BaseException as error:
                lines.put(error)
            finally:
                process.stdout.close()
                lines.put(_EOF)

        reader = threading.Thread(
            target=read_output, name="lensplit-ffmpeg-output", daemon=True
        )
        reader.start()
        output_tail: list[str] = []
        furthest_seconds = 0.0
        reached_eof = False
        try:
            while not reached_eof or not lines.empty() or process.poll() is None:
                if cancel is not None and cancel():
                    self._stop_process(process)
                    raise ExportCancelledError("Video export was cancelled")
                try:
                    entry = lines.get(timeout=0.05)
                except queue.Empty:
                    continue
                if entry is _EOF:
                    reached_eof = True
                    continue
                if isinstance(entry, BaseException):
                    raise ExportFailedError(f"Cannot read FFmpeg output: {entry}")
                line = str(entry).strip()
                if not line:
                    continue
                output_tail.append(line)
                del output_tail[:-80]
                key, separator, value = line.partition("=")
                parsed: float | None = None
                if separator and key in {"out_time_ms", "out_time_us"}:
                    try:
                        parsed = max(0.0, int(value) / 1_000_000.0)
                    except ValueError:
                        parsed = None
                elif separator and key == "out_time":
                    parsed = _progress_seconds(value)
                if parsed is not None:
                    furthest_seconds = max(furthest_seconds, parsed)
                    if progress is not None:
                        shot_fraction = min(1.0, furthest_seconds / duration)
                        progress((shot_index + shot_fraction) / shot_count)
            returncode = process.wait()
        except ExportCancelledError:
            raise
        except BaseException:
            self._stop_process(process)
            raise
        if cancel is not None and cancel():
            self._stop_process(process)
            raise ExportCancelledError("Video export was cancelled")
        if returncode != 0:
            detail = "\n".join(output_tail[-12:])
            raise ExportFailedError(
                f"FFmpeg exited with status {returncode} while exporting interval "
                f"{shot_index + 1}/{shot_count}.\n{detail}"
            )

    def export_project(
        self,
        project: ShotProject,
        options: ExportOptions,
        progress: ProgressCallback | None = None,
        cancel: CancelCallback | None = None,
        warning: WarningCallback | None = None,
    ) -> ExportReport:
        """Export all requested intervals, then atomically write JSON and CSV."""

        started = time.monotonic()
        if not isinstance(options, ExportOptions):
            raise TypeError("options must be ExportOptions")
        if not project.source_shots:
            raise ExportError("Cannot export a project without source shots")
        source_path = Path(self.video_path).expanduser()
        if not source_path.is_file():
            raise ExportError(f"Source video does not exist: '{source_path}'")
        self.video_path = str(source_path.resolve())
        ffmpeg, ffprobe = self._resolve_tools()
        if cancel is not None and cancel():
            raise ExportCancelledError("Video export was cancelled before it started")

        output_dir = Path(options.output_dir).expanduser().resolve()
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ExportError(f"Cannot create export directory '{output_dir}': {error}") from error

        warnings: list[str] = []

        def warn(message: str) -> None:
            warnings.append(message)
            if warning is not None:
                warning(message)

        encoder = "copy" if options.mode == "copy" else self._select_encoder(options.encoder, warn)
        if options.mode == "copy":
            warn(
                "Stream copy is keyframe-aligned; requested shot boundaries may shift "
                "to nearby keyframes. Use exact mode for frame-accurate cuts."
            )
        items = self._build_items(project, options)
        allocator = _PathAllocator(output_dir)
        records: list[ExportRecord] = []
        if progress is not None:
            progress(0.0)

        for index, item in enumerate(items):
            if cancel is not None and cancel():
                raise ExportCancelledError("Video export was cancelled")
            target, part = allocator.output(item.display_number)
            command = self._command(item, options, encoder, part)
            try:
                self._run_encode(
                    command,
                    duration=item.end_pts - item.start_pts,
                    shot_index=index,
                    shot_count=len(items),
                    progress=progress,
                    cancel=cancel,
                )
                if not part.is_file():
                    raise ExportFailedError(
                        f"FFmpeg reported success but did not create '{part}'"
                    )
                os.replace(part, target)
            except BaseException:
                _remove_file(part)
                raise
            records.append(
                ExportRecord(
                    display_number=item.display_number,
                    group_uuid=item.group_uuid,
                    source_uuid=item.source_uuid,
                    source_ids=[shot.uuid for shot in item.shots],
                    source_ordinals=[shot.ordinal for shot in item.shots],
                    start_pts=item.start_pts,
                    end_pts=item.end_pts,
                    internal_cuts=list(item.internal_cuts),
                    output_path=str(target),
                    encoder=encoder,
                )
            )
            if progress is not None:
                progress((index + 1) / len(items))

        json_path, csv_path = allocator.manifests()
        manifest = {
            "schema_version": 1,
            "source_video": self.video_path,
            "records": [record.to_dict() for record in records],
            "transition_statistics": dict(transition_statistics(project)),
            "transitions": [transition.to_dict() for transition in project.transitions],
            "export_parameters": {
                "output_dir": str(output_dir),
                "scope": options.scope,
                "mode": options.mode,
                "requested_encoder": options.encoder,
                "actual_encoder": encoder,
                "numbering_style": {
                    "start": options.numbering_style.start,
                    "width": options.numbering_style.width,
                    "prefix": options.numbering_style.prefix,
                    "suffix": options.numbering_style.suffix,
                    "sequence": options.numbering_style.sequence.value,
                },
                "merge_mode": options.merge_mode.value,
                "ffmpeg_path": ffmpeg,
                "ffprobe_path": ffprobe,
            },
            "analysis_metadata": options.analysis_metadata,
            "transcripts": dict(options.transcripts),
        }
        try:
            self._write_manifests(json_path, csv_path, manifest, records)
        except BaseException:
            _remove_file(json_path)
            _remove_file(csv_path)
            raise
        if progress is not None:
            progress(1.0)
        return ExportReport(
            records=records,
            output_files=[record.output_path for record in records],
            actual_encoder=encoder,
            warnings=warnings,
            elapsed=time.monotonic() - started,
            manifest_json_path=str(json_path),
            manifest_csv_path=str(csv_path),
        )

    @staticmethod
    def _write_manifests(
        json_path: Path,
        csv_path: Path,
        manifest: dict[str, Any],
        records: list[ExportRecord],
    ) -> None:
        json_part = json_path.with_name(json_path.name + ".part")
        csv_part = csv_path.with_name(csv_path.name + ".part")
        try:
            with json_part.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(
                    manifest,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    default=_json_default,
                )
                handle.write("\n")
            with csv_part.open("w", encoding="utf-8-sig", newline="") as handle:
                fields = [
                    "display_number",
                    "group_uuid",
                    "source_uuid",
                    "start_pts",
                    "end_pts",
                    "duration_pts",
                    "source_ids",
                    "source_ordinals",
                    "internal_cuts",
                    "output_path",
                    "actual_encoder",
                    "transcript",
                ]
                transcripts = manifest.get("transcripts", {}) if isinstance(manifest, dict) else {}
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for record in records:
                    writer.writerow(
                        {
                            "display_number": record.display_number,
                            "group_uuid": record.group_uuid or "",
                            "source_uuid": record.source_uuid or "",
                            "start_pts": f"{record.start_pts:.9f}",
                            "end_pts": f"{record.end_pts:.9f}",
                            "duration_pts": f"{record.duration:.9f}",
                            "source_ids": "|".join(record.source_ids),
                            "source_ordinals": "|".join(
                                str(value) for value in record.source_ordinals
                            ),
                            "internal_cuts": "|".join(
                                f"{value:.9f}" for value in record.internal_cuts
                            ),
                            "output_path": record.output_path,
                            "actual_encoder": record.encoder,
                            "transcript": transcripts.get(record.group_uuid or "", ""),
                        }
                    )
            os.replace(json_part, json_path)
            os.replace(csv_part, csv_path)
        except (OSError, TypeError, ValueError) as error:
            raise ExportError(f"Cannot write export manifests: {error}") from error
        finally:
            _remove_file(json_part)
            _remove_file(csv_part)


__all__ = [
    "ExportCancelledError",
    "ExportError",
    "ExportFailedError",
    "ExportOptions",
    "ExportRecord",
    "ExportReport",
    "FFmpegExporter",
    "FFmpegNotFoundError",
    "sanitize_filename",
]
