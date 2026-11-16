"""Streaming PyAV decoding with presentation-time-based frame metadata."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterator, Sequence

import av
import numpy as np


PathLike = str | os.PathLike[str]
CancelCallback = Callable[[], bool]
ProgressCallback = Callable[[float], None]


class VideoDecoderError(RuntimeError):
    """Base error for unreadable or invalid local video."""


class NoVideoStreamError(VideoDecoderError):
    """Raised when a container has no video stream."""


class EmptyVideoError(VideoDecoderError):
    """Raised when the selected video stream yields no frames."""


class DecodeCancelledError(VideoDecoderError):
    """Raised when the caller cancels decoding."""


@dataclass(frozen=True)
class VideoInfo:
    """Metadata for the first video stream."""

    path: str
    stream_index: int
    width: int
    height: int
    duration: float | None
    average_fps: float | None
    frame_count: int | None
    time_base: Fraction

    @property
    def fps(self) -> float | None:
        """Average rate for display only; never used to synthesize frame PTS."""

        return self.average_fps


@dataclass(frozen=True)
class DecodedFrame:
    """One RGB frame and its exact timing information."""

    index: int
    pts: float
    image: np.ndarray
    duration: float

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("Frame index cannot be negative")
        object.__setattr__(self, "pts", float(self.pts))
        object.__setattr__(self, "duration", float(self.duration))
        if not math.isfinite(self.pts):
            raise ValueError("Frame PTS must be finite")
        if not math.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("Frame duration must be finite and positive")
        if not isinstance(self.image, np.ndarray) or self.image.ndim != 3:
            raise ValueError("Decoded image must be an HxWx3 ndarray")

    @property
    def pts_seconds(self) -> float:
        return self.pts

    @property
    def frame_duration(self) -> float:
        return self.duration


def _display_path(path: PathLike) -> str:
    return str(Path(path).expanduser())


def _open_first_video(path: PathLike):
    filename = _display_path(path)
    try:
        container = av.open(filename, mode="r")
    except (OSError, ValueError, av.error.FFmpegError) as error:
        raise VideoDecoderError(f"Cannot open video '{filename}': {error}") from error

    stream = next((item for item in container.streams if item.type == "video"), None)
    if stream is None:
        container.close()
        raise NoVideoStreamError(f"No video stream found in '{filename}'")
    try:
        stream.thread_type = "AUTO"
    except (AttributeError, ValueError):
        pass
    return container, stream


def _stream_duration(container, stream) -> float | None:
    if stream.duration is not None and stream.time_base is not None:
        value = float(stream.duration * stream.time_base)
        if math.isfinite(value) and value >= 0:
            return value
    if container.duration is not None:
        value = float(container.duration / av.time_base)
        if math.isfinite(value) and value >= 0:
            return value
    return None


def probe_video(path: PathLike) -> VideoInfo:
    """Read metadata for the first video stream without decoding all frames."""

    filename = _display_path(path)
    container, stream = _open_first_video(filename)
    try:
        rate = stream.average_rate
        average_fps = float(rate) if rate is not None and rate > 0 else None
        frame_count = int(stream.frames) if stream.frames and stream.frames > 0 else None
        time_base = Fraction(stream.time_base)
        return VideoInfo(
            path=filename,
            stream_index=stream.index,
            width=int(stream.codec_context.width),
            height=int(stream.codec_context.height),
            duration=_stream_duration(container, stream),
            average_fps=average_fps,
            frame_count=frame_count,
            time_base=time_base,
        )
    finally:
        container.close()


def _output_size(frame, size: Sequence[int] | None) -> tuple[int, int] | None:
    if size is None:
        return None
    if len(size) != 2:
        raise ValueError("Decode size must be a (width, height) pair")
    requested_width, requested_height = (int(size[0]), int(size[1]))
    if requested_width <= 0 or requested_height <= 0:
        raise ValueError("Decode width and height must be positive")
    return (
        min(int(frame.width), requested_width),
        min(int(frame.height), requested_height),
    )


def _rgb_image(frame, size: Sequence[int] | None) -> np.ndarray:
    target = _output_size(frame, size)
    if target is None:
        converted = frame.reformat(format="rgb24")
    else:
        converted = frame.reformat(
            width=target[0], height=target[1], format="rgb24"
        )
    return converted.to_ndarray()


def _frame_pts(frame, stream) -> float:
    if frame.pts is None:
        raise VideoDecoderError(
            "A decoded frame has no PTS; variable-rate timestamps cannot be inferred from average FPS"
        )
    time_base = frame.time_base or stream.time_base
    if time_base is None:
        raise VideoDecoderError("A decoded frame has no time base")
    return float(frame.pts * time_base)


def _frame_duration(frame, stream) -> float:
    time_base = frame.time_base or stream.time_base
    if frame.duration is not None and frame.duration > 0 and time_base is not None:
        value = float(frame.duration * time_base)
        if math.isfinite(value) and value > 0:
            return value
    rate = stream.average_rate
    if rate is not None and rate > 0:
        return float(1 / rate)
    if time_base is not None and float(time_base) > 0:
        return float(time_base)
    raise VideoDecoderError("A decoded frame has no usable duration")


def decode_frames(
    path: PathLike,
    size: Sequence[int] | None = None,
    *,
    cancel: CancelCallback | None = None,
    progress: ProgressCallback | None = None,
) -> Iterator[DecodedFrame]:
    """Stream RGB frames, retaining each frame's actual presentation timestamp.

    ``size`` is a maximum width and height. Frames are only reduced, never enlarged
    on disk or in memory by this decoder. Cancellation raises ``DecodeCancelledError``.
    """

    filename = _display_path(path)
    container, stream = _open_first_video(filename)
    info_duration = _stream_duration(container, stream)
    index = 0
    try:
        if cancel is not None and cancel():
            raise DecodeCancelledError("Video decoding was cancelled")
        if progress is not None:
            progress(0.0)
        try:
            packets = container.decode(stream)
            for frame in packets:
                if cancel is not None and cancel():
                    raise DecodeCancelledError("Video decoding was cancelled")
                pts = _frame_pts(frame, stream)
                duration = _frame_duration(frame, stream)
                decoded = DecodedFrame(
                    index=index,
                    pts=pts,
                    image=_rgb_image(frame, size),
                    duration=duration,
                )
                yield decoded
                index += 1
                if progress is not None:
                    fraction = 1.0
                    if info_duration is not None and info_duration > 0:
                        fraction = min(1.0, max(0.0, (pts + duration) / info_duration))
                    progress(fraction)
        except DecodeCancelledError:
            raise
        except (OSError, av.error.FFmpegError) as error:
            raise VideoDecoderError(f"Cannot decode video '{filename}': {error}") from error
        if index == 0:
            raise EmptyVideoError(f"Video stream contains no decodable frames: '{filename}'")
        if progress is not None:
            progress(1.0)
    finally:
        container.close()


def extract_thumbnail(
    path: PathLike,
    pts: float,
    size: Sequence[int] | None = (320, 180),
) -> np.ndarray:
    """Seek by presentation time and return the containing frame as an RGB array."""

    target = float(pts)
    if not math.isfinite(target) or target < 0:
        raise ValueError("Thumbnail PTS must be a finite, non-negative number")
    filename = _display_path(path)
    container, stream = _open_first_video(filename)
    saw_frame = False
    try:
        if stream.time_base is None:
            raise VideoDecoderError("The video stream has no seek time base")
        offset = math.floor(target / float(stream.time_base))
        try:
            container.seek(offset, stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                saw_frame = True
                frame_pts = _frame_pts(frame, stream)
                duration = _frame_duration(frame, stream)
                if frame_pts + duration > target:
                    return _rgb_image(frame, size)
        except (OSError, av.error.FFmpegError) as error:
            raise VideoDecoderError(
                f"Cannot seek or decode thumbnail in '{filename}': {error}"
            ) from error
        if not saw_frame:
            raise EmptyVideoError(f"Video stream contains no decodable frames: '{filename}'")
        raise VideoDecoderError(
            f"Thumbnail PTS {target:.6f}s is outside the decodable video range"
        )
    finally:
        container.close()


# Readable aliases for callers that prefer explicit video-oriented names.
read_video_info = probe_video
seek_thumbnail = extract_thumbnail
