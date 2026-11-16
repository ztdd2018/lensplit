"""Local-only TransNetV2 ONNX inference with bounded sliding windows."""

from __future__ import annotations

import hashlib
import importlib
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from lensplit.analysis.fast import (
    _Boundary,
    _build_project,
    _select_boundaries,
)
from lensplit.core.model import ShotProject
from lensplit.video.decoder import DecodeCancelledError, decode_frames


_PROVIDER_ORDER = (
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
)
_CPU_PROVIDER = "CPUExecutionProvider"


class TransNetError(RuntimeError):
    """Base error for local model and inference failures."""


class TransNetModelError(TransNetError):
    """Raised when no usable local ONNX model can be loaded."""


class TransNetInferenceError(TransNetError):
    """Raised when an ONNX session fails while producing probabilities."""


@dataclass(frozen=True)
class LocalModelStatus:
    configured: bool
    available: bool
    path: str | None = None
    sha256: str | None = None
    providers: tuple[str, ...] = ()
    active_provider: str | None = None
    reason: str | None = None

    @property
    def is_available(self) -> bool:
        return self.available


@dataclass(frozen=True)
class TransNetConfig:
    threshold: float = 0.5
    window: int = 100
    overlap: int = 50
    min_shot_seconds: float = 0.5
    model_path: str | os.PathLike[str] | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("TransNet threshold must be between 0 and 1")
        object.__setattr__(self, "window", int(self.window))
        object.__setattr__(self, "overlap", int(self.overlap))
        if self.window < 2:
            raise ValueError("TransNet window must contain at least two frames")
        if not 0 <= self.overlap < self.window:
            raise ValueError("TransNet overlap must be non-negative and smaller than window")
        if not math.isfinite(self.min_shot_seconds) or self.min_shot_seconds < 0:
            raise ValueError("Minimum shot duration must be finite and non-negative")
        if self.model_path is not None:
            object.__setattr__(self, "model_path", str(self.model_path))


@dataclass(frozen=True)
class _ModelSpec:
    input_name: str
    window: int
    overlap: int
    height: int
    width: int
    dtype: np.dtype

    @property
    def batch_shape(self) -> tuple[int, int, int, int, int]:
        return (1, self.window, self.height, self.width, 3)


class _ModelRunner:
    def __init__(self, session, spec: _ModelSpec, provider: str) -> None:
        self.session = session
        self.spec = spec
        self.provider = provider

    def predict(self, batch: np.ndarray) -> np.ndarray:
        if batch.shape != self.spec.batch_shape:
            raise TransNetInferenceError(
                f"Internal model input shape {batch.shape} does not match {self.spec.batch_shape}"
            )
        try:
            arrays = self.session.run(
                None, {self.spec.input_name: batch.astype(self.spec.dtype, copy=False)}
            )
        except Exception as error:
            raise TransNetInferenceError(
                f"TransNet inference failed on {self.provider}: {error}"
            ) from error
        if not arrays:
            raise TransNetInferenceError("TransNet model returned no outputs")
        output_index = self._single_frame_output_index(arrays)
        probabilities = self._probability_vector(arrays[output_index], self.spec.window)
        if not np.all(np.isfinite(probabilities)):
            raise TransNetInferenceError("TransNet model returned non-finite probabilities")
        return np.clip(probabilities, 0.0, 1.0).astype(np.float32, copy=False)

    def _single_frame_output_index(self, arrays: Sequence[np.ndarray]) -> int:
        try:
            outputs = self.session.get_outputs()
            names = [str(output.name).lower() for output in outputs]
        except (AttributeError, TypeError):
            names = []
        for index, name in enumerate(names[: len(arrays)]):
            compact = name.replace("-", "_")
            if "single" in compact and "frame" in compact:
                return index
        return 0

    @staticmethod
    def _probability_vector(output, window: int) -> np.ndarray:
        array = np.asarray(output)
        if array.ndim == 3 and array.shape[0] == 1 and array.shape[2] == 1:
            array = array[:, :, 0]
        if array.ndim == 2 and array.shape[0] == 1:
            array = array[0]
        elif array.ndim != 1 and array.size == window:
            array = array.reshape(-1)
        if array.ndim != 1 or array.shape[0] != window:
            raise TransNetInferenceError(
                f"Single-frame model output must contain {window} values; got shape {np.asarray(output).shape}"
            )
        return array.astype(np.float32, copy=False)

    def dummy_self_test(self) -> None:
        dummy = np.zeros(self.spec.batch_shape, dtype=self.spec.dtype)
        self.predict(dummy)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _static_dimension(value) -> int | None:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value) if value > 0 else None
    return None


def _input_dtype(input_type: str) -> np.dtype:
    mapping = {
        "tensor(float)": np.dtype(np.float32),
        "tensor(float16)": np.dtype(np.float16),
        "tensor(double)": np.dtype(np.float64),
        "tensor(uint8)": np.dtype(np.uint8),
        "tensor(int8)": np.dtype(np.int8),
    }
    try:
        return mapping[str(input_type)]
    except KeyError as error:
        raise TransNetModelError(
            f"Unsupported TransNet input type '{input_type}'; expected floating-point or 8-bit image data"
        ) from error


def _model_spec(session, config: TransNetConfig) -> _ModelSpec:
    try:
        model_input = session.get_inputs()[0]
    except (IndexError, AttributeError) as error:
        raise TransNetModelError("TransNet ONNX model has no readable input") from error
    shape = list(model_input.shape)
    if len(shape) != 5:
        raise TransNetModelError(
            f"TransNet input must be 5-D channels-last [N,T,H,W,3]; got {shape}"
        )
    batch = _static_dimension(shape[0])
    if batch is not None and batch != 1:
        raise TransNetModelError("Only batch-size 1 or dynamic-batch TransNet models are supported")
    channels = _static_dimension(shape[4])
    if channels is not None and channels != 3:
        raise TransNetModelError("TransNet input must use three RGB channels in the last dimension")

    fixed_window = _static_dimension(shape[1])
    if fixed_window is None:
        window = config.window
        overlap = config.overlap
    else:
        window = fixed_window
        if config.window == fixed_window:
            overlap = config.overlap
        else:
            overlap = int(round(config.overlap * fixed_window / config.window))
            overlap = min(max(0, overlap), window - 1)
    height = _static_dimension(shape[2]) or 27
    width = _static_dimension(shape[3]) or 48
    return _ModelSpec(
        input_name=str(model_input.name),
        window=window,
        overlap=overlap,
        height=height,
        width=width,
        dtype=_input_dtype(model_input.type),
    )


def _model_image(image: np.ndarray, spec: _ModelSpec) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] < 3 or min(array.shape[:2]) < 1:
        raise ValueError("TransNet frames must be non-empty HxWx3 RGB arrays")
    if np.issubdtype(array.dtype, np.floating):
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        if array.size and float(np.max(array)) <= 1.0:
            array = array * 255.0
    array = np.clip(array[:, :, :3], 0, 255).astype(np.uint8, copy=False)
    if array.shape[:2] != (spec.height, spec.width):
        x_indices = np.linspace(0, array.shape[1] - 1, spec.width).astype(np.intp)
        y_indices = np.linspace(0, array.shape[0] - 1, spec.height).astype(np.intp)
        array = array[np.ix_(y_indices, x_indices)]
    return array.astype(spec.dtype, copy=False)


def _session_provider(session, requested: str) -> str:
    try:
        providers = list(session.get_providers())
    except AttributeError:
        return requested
    if requested in providers:
        return requested
    return providers[0] if providers else requested


class TransNetDetector:
    """Run a configured local TransNetV2 model without network model downloads."""

    def __init__(
        self,
        config: TransNetConfig | str | os.PathLike[str] | None = None,
        *,
        model_path: str | os.PathLike[str] | None = None,
        warning: Callable[[str], None] | None = None,
        session_factory: Callable | None = None,
        available_providers: Sequence[str] | None = None,
    ) -> None:
        if isinstance(config, (str, os.PathLike)):
            if model_path is not None:
                raise ValueError("Specify the local model path only once")
            model_path = config
            config = None
        if config is None:
            config = TransNetConfig(model_path=model_path)
        elif model_path is not None:
            config = replace(config, model_path=str(model_path))
        self.config = config
        self._warning = warning
        self._session_factory = session_factory
        self._available_providers = (
            tuple(available_providers) if available_providers is not None else None
        )
        self._runner: _ModelRunner | None = None
        self._cpu_runner: _ModelRunner | None = None
        self.status = self._inspect_local_model()

    @property
    def model_status(self) -> LocalModelStatus:
        return self.status

    def get_model_status(self) -> LocalModelStatus:
        return self.status

    def _provider_candidates(self) -> tuple[str, ...]:
        if self._available_providers is None:
            try:
                ort = importlib.import_module("onnxruntime")
                available = tuple(ort.get_available_providers())
            except Exception as error:
                raise TransNetModelError(
                    f"onnxruntime is unavailable; install a local runtime: {error}"
                ) from error
        else:
            available = self._available_providers
        return tuple(name for name in _PROVIDER_ORDER if name in available)

    def _inspect_local_model(self) -> LocalModelStatus:
        configured_path = self.config.model_path
        if configured_path is None or not str(configured_path).strip():
            return LocalModelStatus(
                configured=False,
                available=False,
                reason="No local TransNetV2 ONNX model is configured",
            )
        if "://" in str(configured_path):
            return LocalModelStatus(
                configured=True,
                available=False,
                path=str(configured_path),
                reason="Only local ONNX files are accepted; remote model URLs are not downloaded",
            )
        path = Path(configured_path).expanduser()
        if not path.is_file():
            return LocalModelStatus(
                configured=True,
                available=False,
                path=str(path),
                reason=f"Local TransNet ONNX file does not exist: '{path}'",
            )
        if path.suffix.lower() != ".onnx":
            return LocalModelStatus(
                configured=True,
                available=False,
                path=str(path),
                reason=f"TransNet model must be a local .onnx file: '{path}'",
            )
        try:
            checksum = _sha256(path)
            providers = self._provider_candidates()
        except (OSError, TransNetModelError) as error:
            return LocalModelStatus(
                configured=True,
                available=False,
                path=str(path),
                reason=str(error),
            )
        if not providers:
            return LocalModelStatus(
                configured=True,
                available=False,
                path=str(path),
                sha256=checksum,
                reason="onnxruntime reports no usable CUDA, DirectML, or CPU execution provider",
            )
        return LocalModelStatus(
            configured=True,
            available=True,
            path=str(path),
            sha256=checksum,
            providers=providers,
        )

    def _new_session(self, providers: Sequence[str]):
        if self._session_factory is not None:
            return self._session_factory(self.status.path, providers=list(providers))
        try:
            ort = importlib.import_module("onnxruntime")
        except Exception as error:
            raise TransNetModelError(f"Cannot import onnxruntime: {error}") from error
        return ort.InferenceSession(self.status.path, providers=list(providers))

    def _load_runner(self, provider: str) -> _ModelRunner:
        providers = [provider]
        available = self.status.providers
        if provider != _CPU_PROVIDER and _CPU_PROVIDER in available:
            providers.append(_CPU_PROVIDER)
        session = self._new_session(providers)
        actual_provider = _session_provider(session, provider)
        spec = _model_spec(session, self.config)
        runner = _ModelRunner(session, spec, actual_provider)
        runner.dummy_self_test()
        return runner

    def _ensure_runner(
        self, warning: Callable[[str], None] | None = None
    ) -> _ModelRunner:
        if self._runner is not None:
            return self._runner
        if not self.status.available or self.status.path is None:
            raise TransNetModelError(self.status.reason or "Local TransNet model is unavailable")
        failures: list[str] = []
        for provider in self.status.providers:
            try:
                runner = self._load_runner(provider)
            except Exception as error:
                failures.append(f"{provider}: {error}")
                if warning is not None and provider != _CPU_PROVIDER:
                    warning(f"TransNet could not initialize {provider}; trying the next provider: {error}")
                continue
            self._runner = runner
            if runner.provider == _CPU_PROVIDER:
                self._cpu_runner = runner
            self.status = replace(self.status, active_provider=runner.provider, reason=None)
            return runner
        self.status = replace(
            self.status,
            available=False,
            reason="Could not load TransNet ONNX model. " + " | ".join(failures),
        )
        raise TransNetModelError(self.status.reason)

    def _ensure_cpu_runner(self) -> _ModelRunner:
        if self._cpu_runner is not None:
            return self._cpu_runner
        if _CPU_PROVIDER not in self.status.providers:
            raise TransNetModelError(
                "GPU inference failed and CPUExecutionProvider is not available for a clean rerun"
            )
        try:
            runner = self._load_runner(_CPU_PROVIDER)
        except Exception as error:
            raise TransNetModelError(
                f"GPU inference failed and the CPU model could not be initialized: {error}"
            ) from error
        self._cpu_runner = runner
        self.status = replace(self.status, active_provider=_CPU_PROVIDER)
        return runner

    def _predict_with_runner(
        self, frames: Iterable[np.ndarray], runner: _ModelRunner
    ) -> np.ndarray:
        spec = runner.spec
        step = spec.window - spec.overlap
        left_pad = spec.overlap // 2
        iterator = iter(frames)
        buffer: list[np.ndarray] = []
        buffer_start = 0
        next_index = 0
        exhausted = False
        block_start = 0
        output_parts: list[np.ndarray] = []
        try:
            while True:
                desired_start = block_start - left_pad
                desired_end = desired_start + spec.window
                keep_from = max(0, desired_start)
                if buffer and buffer_start < keep_from:
                    drop = min(keep_from - buffer_start, len(buffer))
                    del buffer[:drop]
                    buffer_start += drop
                if not buffer:
                    buffer_start = next_index

                while not exhausted and next_index < desired_end:
                    try:
                        image = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    converted = _model_image(image, spec)
                    if next_index >= keep_from:
                        if not buffer:
                            buffer_start = next_index
                        buffer.append(converted)
                    next_index += 1

                valid_count = (
                    min(step, next_index - block_start)
                    if exhausted
                    else step
                )
                if valid_count <= 0:
                    break
                batch = np.zeros(spec.batch_shape, dtype=spec.dtype)
                for offset, image in enumerate(buffer):
                    original_index = buffer_start + offset
                    position = original_index - desired_start
                    if 0 <= position < spec.window:
                        batch[0, position] = image
                probabilities = runner.predict(batch)
                output_parts.append(
                    probabilities[left_pad : left_pad + valid_count].copy()
                )
                block_start += step
                if exhausted and block_start >= next_index:
                    break
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
        if not output_parts:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(output_parts).astype(np.float32, copy=False)

    def predict_frames(self, frames: Iterable[np.ndarray]) -> np.ndarray:
        """Return exactly one probability per input frame using bounded windows."""

        return self._predict_with_runner(frames, self._ensure_runner(self._warning))

    def _probabilities_to_project(
        self,
        probabilities: np.ndarray,
        pts: list[float],
        durations: list[float],
    ) -> ShotProject:
        if len(probabilities) != len(pts) or not pts:
            raise TransNetInferenceError(
                f"TransNet returned {len(probabilities)} probabilities for {len(pts)} decoded frames"
            )
        candidates: list[_Boundary] = []
        index = 0
        while index < len(probabilities):
            if probabilities[index] < self.config.threshold:
                index += 1
                continue
            group_end = index + 1
            while (
                group_end < len(probabilities)
                and probabilities[group_end] >= self.config.threshold
            ):
                group_end += 1
            peak = max(
                range(index, group_end), key=lambda item: float(probabilities[item])
            )
            if peak > 0:
                candidates.append(
                    _Boundary(peak, pts[peak], float(probabilities[peak]), "cut")
                )
            index = group_end
        end_pts = pts[-1] + durations[-1]
        boundaries = _select_boundaries(
            candidates, pts[0], end_pts, self.config.min_shot_seconds
        )
        return _build_project(
            pts[0], end_pts, boundaries, "transnetv2", frame_count=len(pts)
        )

    def _detect_with_runner(
        self,
        path: str | Path,
        runner: _ModelRunner,
        cancel: Callable[[], bool] | None,
        progress: Callable[[float], None] | None,
    ) -> ShotProject:
        pts: list[float] = []
        durations: list[float] = []

        def model_frames():
            for frame in decode_frames(
                path,
                size=(runner.spec.width, runner.spec.height),
                cancel=cancel,
                progress=progress,
            ):
                if pts and frame.pts <= pts[-1]:
                    raise ValueError("Frame PTS values must be strictly increasing")
                pts.append(frame.pts)
                durations.append(frame.duration)
                yield frame.image

        probabilities = self._predict_with_runner(model_frames(), runner)
        return self._probabilities_to_project(probabilities, pts, durations)

    def detect(
        self,
        path: str | Path,
        *,
        cancel: Callable[[], bool] | None = None,
        progress: Callable[[float], None] | None = None,
        warning: Callable[[str], None] | None = None,
    ) -> ShotProject:
        """Detect shots, restarting the entire decode on a non-CPU runtime failure."""

        notify = warning or self._warning
        runner = self._ensure_runner(notify)
        try:
            return self._detect_with_runner(path, runner, cancel, progress)
        except TransNetInferenceError as error:
            if runner.provider == _CPU_PROVIDER:
                raise
            if notify is not None:
                notify(
                    f"TransNet inference failed on {runner.provider}; restarting decoding and inference on CPU: {error}"
                )
            cpu_runner = self._ensure_cpu_runner()
            return self._detect_with_runner(path, cpu_runner, cancel, progress)


__all__ = [
    "LocalModelStatus",
    "TransNetConfig",
    "TransNetDetector",
    "TransNetError",
    "TransNetInferenceError",
    "TransNetModelError",
]
