"""Local-only speech recognition over detected shot clips."""

from __future__ import annotations

import math
import os
import re
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

import av
import numpy as np

_TARGET_RATE = 16000
_SILENCE_RMS = 0.0016
_PUNCTUATION = re.compile(r"^[\s\.,!?;:'\"\-—…、，。！？；：（）()《》【】]+$")


class AsrError(RuntimeError):
    """Base error for local speech recognition failures."""


class AsrModelError(AsrError):
    """Raised when the local Whisper model cannot be loaded."""


class AsrCancelledError(AsrError):
    """Raised when the user cancels transcription."""


@dataclass(frozen=True)
class LocalAsrStatus:
    configured: bool
    available: bool
    path: str | None = None
    device: str | None = None
    compute_type: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class AsrConfig:
    """Settings for the offline Whisper model."""

    model_path: str | os.PathLike[str] | None = None
    language: str | None = None
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 1
    vad_filter: bool = True
    min_speech_seconds: float = 0.2

    def __post_init__(self) -> None:
        if self.model_path is not None:
            object.__setattr__(self, "model_path", str(self.model_path))
        if self.beam_size < 1:
            raise ValueError("ASR beam size must be at least 1")
        if not math.isfinite(self.min_speech_seconds) or self.min_speech_seconds < 0:
            raise ValueError("Minimum speech duration must be non-negative")


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, object]:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "text": self.text}


@dataclass
class TranscriptResult:
    text: str = ""
    language: str | None = None
    segments: list[TranscriptSegment] = field(default_factory=list)
    speech_seconds: float = 0.0

    @property
    def has_speech(self) -> bool:
        return bool(self.text.strip())


def extract_clip_audio(
    path: str | os.PathLike[str],
    start: float,
    end: float,
    *,
    cancel: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Decode one time span into 16 kHz mono float samples entirely in memory."""

    if end <= start:
        raise ValueError("Clip end must be greater than its start")
    chunks: list[np.ndarray] = []
    first_time: float | None = None
    with av.open(str(path), "r") as container:
        if not container.streams.audio:
            raise AsrError("该视频没有音轨，无法识别台词")
        stream = container.streams.audio[0]
        try:
            container.seek(int(start / stream.time_base), stream=stream, backward=True)
        except (av.AVError, ValueError, OverflowError):
            pass
        resampler = av.AudioResampler(format="flt", layout="mono", rate=_TARGET_RATE)
        for frame in container.decode(stream):
            if cancel is not None and cancel():
                raise AsrCancelledError("台词识别已取消")
            time = frame.time
            if time is None:
                continue
            duration = frame.samples / float(frame.sample_rate or _TARGET_RATE)
            if time + duration <= start:
                continue
            if time >= end:
                break
            if first_time is None:
                first_time = time
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().reshape(-1))
        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    audio = np.concatenate(chunks).astype(np.float32, copy=False)
    offset = max(0.0, (start - (first_time if first_time is not None else start)))
    if offset:
        audio = audio[int(offset * _TARGET_RATE) :]
    limit = int(round((end - start) * _TARGET_RATE))
    if limit > 0:
        audio = audio[:limit]
    return audio


def _load_whisper_model(model_path: str, device: str, compute_type: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:  # pragma: no cover - depends on optional install
        raise AsrModelError(
            "未安装 faster-whisper，无法进行本地台词识别"
        ) from error
    try:
        return WhisperModel(model_path, device=device, compute_type=compute_type)
    except Exception as error:
        raise AsrModelError(f"本地 Whisper 模型加载失败：{error}") from error


class LocalTranscriber:
    """Wrap a local CTranslate2 Whisper model with strict offline behaviour."""

    def __init__(self, config: AsrConfig | None = None) -> None:
        self.config = config or AsrConfig()
        self._prefer_cpu = False
        self.status = self._probe()
        self._model = None
        self._model_lock = threading.Lock()

    def _probe(self) -> LocalAsrStatus:
        path = self.config.model_path
        if not path:
            return LocalAsrStatus(False, False, reason="未配置本地台词模型")
        model_dir = Path(path)
        if not model_dir.is_dir():
            return LocalAsrStatus(False, False, str(model_dir), reason="模型目录不存在")
        if not (model_dir / "model.bin").is_file():
            return LocalAsrStatus(
                False, False, str(model_dir), reason="模型目录缺少 model.bin"
            )
        device, compute_type = self._resolve_device()
        return LocalAsrStatus(
            True,
            True,
            str(model_dir),
            device=device,
            compute_type=compute_type,
        )

    def _resolve_device(self) -> tuple[str, str]:
        if self._prefer_cpu:
            return "cpu", "int8"
        if self.config.device != "auto":
            if self.config.compute_type != "auto":
                return self.config.device, self.config.compute_type
            return self.config.device, "int8" if self.config.device == "cpu" else "float16"
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda", "float16"
        except Exception:
            pass
        return "cpu", "int8"

    @property
    def model(self):
        with self._model_lock:
            if self._model is None:
                device, compute_type = self._resolve_device()
                try:
                    self._model = _load_whisper_model(
                        self.config.model_path, device, compute_type
                    )
                except AsrModelError:
                    if device == "cpu":
                        raise
                    # 驱动或 CUDA 运行库缺失时退回 CPU，不静默使用 GPU。
                    self._model = _load_whisper_model(self.config.model_path, "cpu", "int8")
                    device, compute_type = "cpu", "int8"
                self.status = replace(
                    self.status, device=device, compute_type=compute_type
                )
            return self._model

    def transcribe(
        self,
        path: str | os.PathLike[str],
        start: float,
        end: float,
        *,
        cancel: Callable[[], bool] | None = None,
    ) -> TranscriptResult:
        if not self.status.available:
            raise AsrModelError(self.status.reason or "本地台词模型不可用")
        if end - start < self.config.min_speech_seconds:
            return TranscriptResult()
        audio = extract_clip_audio(path, start, end, cancel=cancel)
        if audio.size == 0:
            return TranscriptResult()
        if float(np.sqrt(np.mean(np.square(audio)))) < _SILENCE_RMS:
            return TranscriptResult()
        try:
            return self._run(self.model, audio, start, cancel)
        except AsrCancelledError:
            raise
        except AsrError:
            if self.status.device == "cpu":
                raise
            # 显存驱动/运行库问题在真正推理时才暴露，这里退回 CPU 后重试一次。
            self._force_cpu()
            return self._run(self.model, audio, start, cancel)

    def _run(
        self,
        model,
        audio: np.ndarray,
        start: float,
        cancel: Callable[[], bool] | None,
    ) -> TranscriptResult:
        try:
            segments, info = model.transcribe(
                audio,
                language=self.config.language,
                beam_size=self.config.beam_size,
                vad_filter=self.config.vad_filter,
                vad_parameters={"min_silence_duration_ms": 400},
                condition_on_previous_text=False,
                word_timestamps=False,
            )
            collected: list[TranscriptSegment] = []
            for segment in segments:
                if cancel is not None and cancel():
                    raise AsrCancelledError("台词识别已取消")
                text = str(segment.text).strip()
                if not text or _PUNCTUATION.match(text):
                    continue
                collected.append(
                    TranscriptSegment(
                        start=start + float(segment.start),
                        end=start + float(segment.end),
                        text=text,
                    )
                )
        except AsrCancelledError:
            raise
        except Exception as error:
            raise AsrError(f"台词识别失败：{error}") from error
        return TranscriptResult(
            text=" ".join(segment.text for segment in collected),
            language=getattr(info, "language", None),
            segments=collected,
            speech_seconds=sum(segment.end - segment.start for segment in collected),
        )

    def _force_cpu(self) -> None:
        self._prefer_cpu = True
        self._model = None
        self.status = replace(self.status, device="cpu", compute_type="int8")


class AsrWorker:
    """Run transcription over many clips on a background thread."""

    def __init__(
        self,
        transcriber: LocalTranscriber,
        path: str | os.PathLike[str],
        clips: Sequence[tuple[str, float, float]],
        *,
        on_progress: Callable[[int, int, str], None] | None = None,
        on_clip_result: Callable[[str, TranscriptResult], None] | None = None,
        on_result: Callable[[dict[str, TranscriptResult]], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        self.transcriber = transcriber
        self.path = str(path)
        self.clips = list(clips)
        self._on_progress = on_progress
        self._on_clip_result = on_clip_result
        self._on_result = on_result
        self._on_error = on_error
        self._cancel = threading.Event()
        self._future: Future[dict[str, TranscriptResult]] = Future()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def future(self) -> Future[dict[str, TranscriptResult]]:
        return self._future

    def start(self) -> AsrWorker:
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("An AsrWorker can only be started once")
            self._thread = threading.Thread(
                target=self._execute, name="lensplit-asr", daemon=True
            )
            self._thread.start()
        return self

    def _execute(self) -> None:
        if not self._future.set_running_or_notify_cancel():
            return
        results: dict[str, TranscriptResult] = {}
        total = len(self.clips)
        try:
            for index, (clip_id, start, end) in enumerate(self.clips, start=1):
                if self._cancel.is_set():
                    raise AsrCancelledError("台词识别已取消")
                if self._on_progress is not None:
                    self._on_progress(index, total, clip_id)
                result = self.transcriber.transcribe(
                    self.path, start, end, cancel=self._cancel.is_set
                )
                results[clip_id] = result
                if self._on_clip_result is not None:
                    self._on_clip_result(clip_id, result)
        except BaseException as error:
            self._future.set_exception(error)
            if self._on_error is not None:
                self._on_error(error)
            return
        self._future.set_result(results)
        if self._on_result is not None:
            self._on_result(results)

    def cancel(self) -> None:
        self._cancel.set()

    def is_running(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def join(self, timeout: float | None = None) -> AsrWorker:
        thread = self._thread
        if thread is None:
            raise RuntimeError("AsrWorker has not been started")
        thread.join(timeout)
        return self


__all__ = [
    "AsrCancelledError",
    "AsrConfig",
    "AsrError",
    "AsrModelError",
    "AsrWorker",
    "LocalAsrStatus",
    "LocalTranscriber",
    "TranscriptResult",
    "TranscriptSegment",
    "extract_clip_audio",
]
