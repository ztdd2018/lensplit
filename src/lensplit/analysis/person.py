"""Local person counting with a YOLOv8n ONNX detector.

The detector runs entirely on this machine with onnxruntime; no frames leave
the computer. Decoding reuses the PyAV helpers in ``lensplit.video.decoder``.
"""

from __future__ import annotations

import math
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from lensplit.video.decoder import VideoDecoderError, extract_thumbnail

PERSON_CLASS = 0
INPUT_SIZE = 640
_PAD_VALUE = 114.0


class PersonModelError(RuntimeError):
    """Raised when the person detection model is missing or unusable."""


class PersonCancelledError(RuntimeError):
    """Raised when the caller cancels person detection."""


@dataclass(frozen=True)
class PersonModelStatus:
    """Whether a usable detector model is configured on this machine."""

    configured: bool
    available: bool
    path: str = ""
    message: str = ""


@dataclass
class PersonDetectConfig:
    """Tuning for local person counting."""

    model_path: str = ""
    confidence: float = 0.22
    iou: float = 0.5
    samples_per_shot: int = 5
    min_height_ratio: float = 0.03
    max_decode_side: int = 1280

    def __post_init__(self) -> None:
        self.model_path = str(self.model_path or "")
        self.confidence = min(0.99, max(0.01, float(self.confidence)))
        self.iou = min(0.99, max(0.01, float(self.iou)))
        self.samples_per_shot = max(1, min(9, int(self.samples_per_shot)))
        self.min_height_ratio = min(0.9, max(0.0, float(self.min_height_ratio)))
        self.max_decode_side = max(INPUT_SIZE, int(self.max_decode_side))


def _letterbox(image: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Resize keeping aspect ratio and pad to the model input square."""

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("待检测画面尺寸不合法")
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    rows = (np.arange(new_height) * (height / new_height)).astype(np.int32)
    columns = (np.arange(new_width) * (width / new_width)).astype(np.int32)
    resized = image[rows.clip(0, height - 1)][:, columns.clip(0, width - 1)]
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), _PAD_VALUE, dtype=np.uint8)
    top = (INPUT_SIZE - new_height) // 2
    left = (INPUT_SIZE - new_width) // 2
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas, scale, float(left), float(top)


def _downscale(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    rows = (np.arange(new_height) * (height / new_height)).astype(np.int32)
    columns = (np.arange(new_width) * (width / new_width)).astype(np.int32)
    return image[rows.clip(0, height - 1)][:, columns.clip(0, width - 1)]


def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    """Greedy non-maximum suppression over xyxy boxes."""

    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])
        intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[current] + areas[rest] - intersection
        overlap = np.where(union > 0, intersection / np.maximum(union, 1e-9), 0.0)
        order = rest[overlap <= iou]
    return keep


def decode_person_boxes(
    output: np.ndarray,
    scale: float,
    pad_x: float,
    pad_y: float,
    source_size: tuple[int, int],
    confidence: float,
    iou: float,
    min_height_ratio: float,
) -> np.ndarray:
    """Turn a raw ``(1, 84, 8400)`` YOLOv8 output into filtered xyxy boxes."""

    predictions = np.asarray(output, dtype=np.float32)
    if predictions.ndim != 3:
        raise PersonModelError("人物检测模型输出维度不正确")
    matrix = predictions[0]
    if matrix.shape[0] <= 4:
        raise PersonModelError("人物检测模型输出缺少类别分数")
    boxes = matrix[:4].T
    scores = matrix[4 + PERSON_CLASS]
    keep = scores >= confidence
    if not np.any(keep):
        return np.empty((0, 4), dtype=np.float32)
    boxes = boxes[keep]
    scores = scores[keep]
    width, height = source_size
    x1 = (boxes[:, 0] - boxes[:, 2] / 2.0 - pad_x) / max(scale, 1e-6)
    y1 = (boxes[:, 1] - boxes[:, 3] / 2.0 - pad_y) / max(scale, 1e-6)
    x2 = (boxes[:, 0] + boxes[:, 2] / 2.0 - pad_x) / max(scale, 1e-6)
    y2 = (boxes[:, 1] + boxes[:, 3] / 2.0 - pad_y) / max(scale, 1e-6)
    xyxy = np.stack(
        [
            np.clip(x1, 0.0, width),
            np.clip(y1, 0.0, height),
            np.clip(x2, 0.0, width),
            np.clip(y2, 0.0, height),
        ],
        axis=1,
    ).astype(np.float32)
    box_heights = xyxy[:, 3] - xyxy[:, 1]
    big_enough = box_heights >= max(1.0, height * min_height_ratio)
    xyxy = xyxy[big_enough]
    scores = scores[big_enough]
    if len(xyxy) == 0:
        return np.empty((0, 4), dtype=np.float32)
    chosen = _nms(xyxy, scores, iou)
    return xyxy[chosen]


class YoloPersonDetector:
    """Count people with a local YOLOv8n ONNX model."""

    def __init__(self, config: PersonDetectConfig | None = None) -> None:
        self.config = config or PersonDetectConfig()
        self._session = None

    @property
    def status(self) -> PersonModelStatus:
        path = self.config.model_path
        if not path:
            return PersonModelStatus(False, False, "", "未配置人物检测模型")
        model = Path(path).expanduser()
        if not model.is_file():
            return PersonModelStatus(True, False, str(model), f"模型不存在：{model}")
        return PersonModelStatus(True, True, str(model), "人物检测模型就绪")

    def _load(self):
        if self._session is not None:
            return self._session
        status = self.status
        if not status.available:
            raise PersonModelError(status.message)
        try:
            import onnxruntime as ort
        except ImportError as error:  # pragma: no cover - dependency is required
            raise PersonModelError("未安装 onnxruntime，无法运行本地人物检测") from error
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 0
        try:
            self._session = ort.InferenceSession(
                status.path, sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as error:
            raise PersonModelError(f"无法加载人物检测模型：{error}") from error
        return self._session

    def count(self, image: np.ndarray) -> int:
        """Return how many people are visible in one RGB frame."""

        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("人物检测需要 HxWx3 的 RGB 图像")
        session = self._load()
        rgb = np.asarray(image[:, :, :3], dtype=np.uint8)
        frame = _downscale(rgb, self.config.max_decode_side)
        canvas, scale, pad_x, pad_y = _letterbox(frame)
        tensor = canvas.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: np.ascontiguousarray(tensor)})
        boxes = decode_person_boxes(
            outputs[0],
            scale,
            pad_x,
            pad_y,
            (frame.shape[1], frame.shape[0]),
            self.config.confidence,
            self.config.iou,
            self.config.min_height_ratio,
        )
        return int(len(boxes))


def count_persons_per_shot(
    video_path: str,
    intervals: Sequence[tuple[str, float, float]],
    config: PersonDetectConfig,
    *,
    cancel: Callable[[], bool] | None = None,
    progress: Callable[[float], None] | None = None,
    on_shot: Callable[[str, int], None] | None = None,
    detector: YoloPersonDetector | None = None,
) -> dict[str, int]:
    """Sample a few frames per logical shot and return a stable person count."""

    engine = detector or YoloPersonDetector(config)
    results: dict[str, int] = {}
    total = max(1, len(intervals))
    samples = max(1, config.samples_per_shot)
    fractions = tuple((index + 1) / (samples + 1) for index in range(samples))
    for index, (group_uuid, start, end) in enumerate(intervals):
        if cancel is not None and cancel():
            raise PersonCancelledError("人物标记已取消")
        span = max(0.0, float(end) - float(start))
        counts: list[int] = []
        for fraction in fractions:
            if cancel is not None and cancel():
                raise PersonCancelledError("人物标记已取消")
            target = float(start) + span * fraction
            try:
                frame = extract_thumbnail(video_path, max(0.0, target), None)
            except (VideoDecoderError, ValueError):
                continue
            try:
                counts.append(engine.count(frame))
            except PersonModelError:
                raise
            except Exception as error:  # keep other shots usable
                raise PersonModelError(f"人物检测失败：{error}") from error
        # People walk in and out of frame, so the busiest sample is the best
        # estimate of how many people the shot actually needs.
        value = max(counts) if counts else 0
        results[group_uuid] = value
        if on_shot is not None:
            on_shot(group_uuid, value)
        if progress is not None:
            progress((index + 1) / total)
    return results


class PersonDetectWorker:
    """Run person counting on a background thread for the desktop UI."""

    def __init__(
        self,
        video_path: str,
        intervals: Sequence[tuple[str, float, float]],
        config: PersonDetectConfig,
        *,
        on_progress: Callable[[float], None] | None = None,
        on_shot: Callable[[str, int], None] | None = None,
        on_result: Callable[[dict[str, int]], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        self.video_path = str(video_path)
        self.intervals = list(intervals)
        self.config = config
        self._on_progress = on_progress
        self._on_shot = on_shot
        self._on_result = on_result
        self._on_error = on_error
        self._cancel_event = threading.Event()
        self._future: Future[dict[str, int]] = Future()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def future(self) -> Future[dict[str, int]]:
        return self._future

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def start(self) -> PersonDetectWorker:
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("人物标记任务只能启动一次")
            self._thread = threading.Thread(
                target=self._execute, name="lensplit-person", daemon=True
            )
            self._thread.start()
        return self

    def _execute(self) -> None:
        if not self._future.set_running_or_notify_cancel():
            return
        try:
            result = count_persons_per_shot(
                self.video_path,
                self.intervals,
                self.config,
                cancel=self._cancel_event.is_set,
                progress=self._on_progress,
                on_shot=self._on_shot,
            )
        except BaseException as error:
            self._future.set_exception(error)
            if self._on_error is not None:
                self._on_error(error)
            return
        self._future.set_result(result)
        if self._on_result is not None:
            self._on_result(result)

    def cancel(self) -> None:
        self._cancel_event.set()
        self._future.cancel()

    def join(self, timeout: float | None = None) -> PersonDetectWorker:
        thread = self._thread
        if thread is None:
            raise RuntimeError("人物标记任务尚未启动")
        thread.join(timeout)
        return self


__all__ = [
    "PersonCancelledError",
    "PersonDetectConfig",
    "PersonDetectWorker",
    "PersonModelError",
    "PersonModelStatus",
    "YoloPersonDetector",
    "count_persons_per_shot",
    "decode_person_boxes",
]
