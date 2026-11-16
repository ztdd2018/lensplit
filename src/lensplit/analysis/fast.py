"""Streaming shot detection using grayscale and normalized-histogram changes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence
from uuid import uuid4

import numpy as np

from lensplit.core.model import ShotProject, SourceShot, Transition
from lensplit.video.decoder import DecodedFrame, decode_frames


@dataclass(frozen=True)
class FastDetectionConfig:
    threshold: float = 0.60
    gradual_threshold: float = 0.16
    min_shot_seconds: float = 0.50
    analysis_size: tuple[int, int] = (160, 90)

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Fast cut threshold must be between 0 and 1")
        if not 0.0 <= self.gradual_threshold <= self.threshold:
            raise ValueError("Gradual threshold must be between 0 and the cut threshold")
        if not math.isfinite(self.min_shot_seconds) or self.min_shot_seconds < 0:
            raise ValueError("Minimum shot duration must be finite and non-negative")
        if len(self.analysis_size) != 2 or any(
            int(value) <= 0 for value in self.analysis_size
        ):
            raise ValueError("Analysis size must contain positive width and height")
        object.__setattr__(
            self,
            "analysis_size",
            (int(self.analysis_size[0]), int(self.analysis_size[1])),
        )


@dataclass(frozen=True)
class _Boundary:
    index: int
    pts: float
    score: float
    kind: str


@dataclass
class _Feature:
    gray: np.ndarray
    histogram: np.ndarray


def _resize_for_analysis(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = image.shape[:2]
    target_width = min(width, size[0])
    target_height = min(height, size[1])
    if (target_width, target_height) == (width, height):
        return image
    x_indices = np.linspace(0, width - 1, target_width).astype(np.intp)
    y_indices = np.linspace(0, height - 1, target_height).astype(np.intp)
    return image[np.ix_(y_indices, x_indices)]


def _extract_feature(image: np.ndarray, size: tuple[int, int]) -> _Feature:
    array = np.asarray(image)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("Analysis frames must be non-empty HxWxC arrays")
    if array.shape[2] < 1:
        raise ValueError("Analysis frames must have at least one channel")
    array = _resize_for_analysis(array, size)
    if np.issubdtype(array.dtype, np.floating):
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        if array.size and float(np.max(array)) <= 1.0:
            array = array * 255.0
    if array.shape[2] >= 3:
        rgb = array[:, :, :3].astype(np.float32, copy=False)
        gray = (
            rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
        )
    else:
        gray = array[:, :, 0].astype(np.float32, copy=False)
    gray = np.clip(gray, 0.0, 255.0)
    histogram = np.histogram(gray, bins=32, range=(0.0, 256.0))[0].astype(np.float64)
    histogram /= float(gray.size)
    return _Feature(gray=gray, histogram=histogram)


def _feature_change(left: _Feature, right: _Feature) -> float:
    gray_change = float(np.mean(np.abs(left.gray - right.gray))) / 255.0
    histogram_change = 0.5 * float(np.sum(np.abs(left.histogram - right.histogram)))
    return float(np.clip(0.65 * gray_change + 0.35 * histogram_change, 0.0, 1.0))


def _select_boundaries(
    candidates: Iterable[_Boundary],
    start_pts: float,
    end_pts: float,
    min_shot_seconds: float,
) -> list[_Boundary]:
    """Resolve dense candidates in favour of the stronger compatible change."""

    selected: list[_Boundary] = []
    epsilon = 1e-9
    for candidate in candidates:
        if candidate.pts - start_pts + epsilon < min_shot_seconds:
            continue
        if selected and candidate.pts - selected[-1].pts + epsilon < min_shot_seconds:
            previous_start = selected[-2].pts if len(selected) > 1 else start_pts
            replacement_is_compatible = (
                candidate.pts - previous_start + epsilon >= min_shot_seconds
            )
            replacement_preserves_final_shot = (
                end_pts - candidate.pts + epsilon >= min_shot_seconds
                or end_pts - selected[-1].pts + epsilon < min_shot_seconds
            )
            if (
                candidate.score > selected[-1].score
                and replacement_is_compatible
                and replacement_preserves_final_shot
            ):
                selected[-1] = candidate
            continue
        selected.append(candidate)
    if selected and end_pts - selected[-1].pts + epsilon < min_shot_seconds:
        selected.pop()
    return selected


def _build_project(
    start_pts: float,
    end_pts: float,
    boundaries: Sequence[_Boundary],
    engine: str,
    frame_count: int,
) -> ShotProject:
    if not math.isfinite(start_pts) or not math.isfinite(end_pts) or end_pts <= start_pts:
        raise ValueError("Frame timeline must have finite, increasing PTS bounds")
    if frame_count < 1 or any(
        not 0 < boundary.index < frame_count for boundary in boundaries
    ):
        raise ValueError("A shot boundary must refer to a new frame")
    edge_pts = [start_pts, *(boundary.pts for boundary in boundaries), end_pts]
    edge_frames = [0, *(boundary.index for boundary in boundaries), frame_count]
    shots: list[SourceShot] = []
    for ordinal, (left, right) in enumerate(zip(edge_pts, edge_pts[1:]), start=1):
        if right <= left:
            raise ValueError("Detected shot boundaries must have strictly increasing PTS")
        shots.append(
            SourceShot(
                uuid=str(uuid4()),
                ordinal=ordinal,
                start_pts=left,
                end_pts=right,
                start_frame=edge_frames[ordinal - 1],
                end_frame=edge_frames[ordinal],
            )
        )
    transitions = [
        Transition(
            before_uuid=left.uuid,
            after_uuid=right.uuid,
            cut_pts=left.end_pts,
            kind=boundary.kind,
            confidence=float(np.clip(boundary.score, 0.0, 1.0)),
            engine=engine,
        )
        for left, right, boundary in zip(shots, shots[1:], boundaries)
    ]
    return ShotProject(source_shots=shots, transitions=transitions)


class FastDetector:
    """Detect hard cuts and accumulated gradual changes in a frame stream."""

    def __init__(self, config: FastDetectionConfig | None = None) -> None:
        self.config = config or FastDetectionConfig()

    def detect(
        self,
        path: str | Path,
        *,
        cancel: Callable[[], bool] | None = None,
        progress: Callable[[float], None] | None = None,
    ) -> ShotProject:
        frames = decode_frames(
            path,
            size=self.config.analysis_size,
            cancel=cancel,
            progress=progress,
        )
        return self.detect_frames(frames)

    def detect_frames(self, frames: Iterable[DecodedFrame]) -> ShotProject:
        """Run detection without video I/O, suitable for synthetic unit tests."""

        previous_feature: _Feature | None = None
        gradual_anchor: _Feature | None = None
        gradual_frames = 0
        gradual_accumulation = 0.0
        candidates: list[_Boundary] = []
        first_pts: float | None = None
        previous_pts: float | None = None
        last_pts: float | None = None
        last_duration = 0.0

        for index, frame in enumerate(frames):
            pts = float(frame.pts)
            duration = float(frame.duration)
            if not math.isfinite(pts) or not math.isfinite(duration) or duration <= 0:
                raise ValueError("Frame PTS must be finite and duration must be positive")
            if previous_pts is not None and pts <= previous_pts:
                raise ValueError("Frame PTS values must be strictly increasing")
            if first_pts is None:
                first_pts = pts
            feature = _extract_feature(frame.image, self.config.analysis_size)
            if previous_feature is not None:
                score = _feature_change(previous_feature, feature)
                if score >= self.config.threshold:
                    candidates.append(_Boundary(index, pts, score, "cut"))
                    gradual_anchor = None
                    gradual_frames = 0
                    gradual_accumulation = 0.0
                elif score >= self.config.gradual_threshold:
                    if gradual_anchor is None:
                        gradual_anchor = previous_feature
                        gradual_frames = 1
                        gradual_accumulation = 0.0
                    gradual_frames += 1
                    gradual_accumulation += score
                    anchor_change = _feature_change(gradual_anchor, feature)
                    if gradual_frames >= 2 and (
                        gradual_accumulation >= self.config.threshold
                        or anchor_change >= self.config.threshold
                    ):
                        confidence = max(anchor_change, gradual_accumulation)
                        candidates.append(_Boundary(index, pts, confidence, "gradual"))
                        gradual_anchor = None
                        gradual_frames = 0
                        gradual_accumulation = 0.0
                else:
                    gradual_anchor = None
                    gradual_frames = 0
                    gradual_accumulation = 0.0
            previous_feature = feature
            previous_pts = pts
            last_pts = pts
            last_duration = duration

        if first_pts is None or last_pts is None:
            raise ValueError("At least one frame is required for shot detection")
        end_pts = last_pts + last_duration
        boundaries = _select_boundaries(
            candidates, first_pts, end_pts, self.config.min_shot_seconds
        )
        return _build_project(
            first_pts, end_pts, boundaries, "fast", frame_count=index + 1
        )


def detect_frames(
    frames: Iterable[DecodedFrame], config: FastDetectionConfig | None = None
) -> ShotProject:
    """Functional form of ``FastDetector.detect_frames``."""

    return FastDetector(config).detect_frames(frames)
