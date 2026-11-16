"""Thread-based analysis execution without a GUI toolkit dependency."""

from __future__ import annotations

import os
import threading
from concurrent.futures import Future
from typing import Callable

from lensplit.analysis.fast import FastDetectionConfig, FastDetector
from lensplit.analysis.transnet import LocalModelStatus, TransNetConfig, TransNetDetector
from lensplit.core.model import ShotProject


class AnalysisWorker:
    """Own one detector invocation and transfer its result through a Future."""

    def __init__(
        self,
        engine: str,
        path: str | os.PathLike[str],
        *,
        config: FastDetectionConfig | TransNetConfig | None = None,
        model_path: str | os.PathLike[str] | None = None,
        on_progress: Callable[[float], None] | None = None,
        on_result: Callable[[ShotProject], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        on_warning: Callable[[str], None] | None = None,
        on_model_status: Callable[[LocalModelStatus], None] | None = None,
    ) -> None:
        normalized_engine = engine.strip().lower()
        if normalized_engine not in {"fast", "transnet", "transnetv2"}:
            raise ValueError("Analysis engine must be 'fast' or 'transnet'")
        if normalized_engine == "fast" and not isinstance(
            config, (FastDetectionConfig, type(None))
        ):
            raise TypeError("Fast analysis requires FastDetectionConfig")
        if normalized_engine != "fast" and not isinstance(
            config, (TransNetConfig, type(None))
        ):
            raise TypeError("TransNet analysis requires TransNetConfig")
        self.engine = normalized_engine
        self.path = str(path)
        self._config = config
        self._model_path = model_path
        self._on_progress = on_progress
        self._on_result = on_result
        self._on_error = on_error
        self._on_warning = on_warning
        self._on_model_status = on_model_status
        self._cancel_event = threading.Event()
        self._future: Future[ShotProject] = Future()
        self._thread: threading.Thread | None = None
        self._state_lock = threading.Lock()

    @property
    def future(self) -> Future[ShotProject]:
        return self._future

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def start(self) -> AnalysisWorker:
        with self._state_lock:
            if self._thread is not None:
                raise RuntimeError("An AnalysisWorker can only be started once")
            self._thread = threading.Thread(
                target=self._execute,
                name=f"lensplit-{self.engine}",
                daemon=True,
            )
            self._thread.start()
        return self

    def _execute(self) -> None:
        if not self._future.set_running_or_notify_cancel():
            return
        try:
            if self.engine == "fast":
                detector = FastDetector(self._config)
                result = detector.detect(
                    self.path,
                    cancel=self._cancel_event.is_set,
                    progress=self._on_progress,
                )
            else:
                detector = TransNetDetector(
                    self._config,
                    model_path=self._model_path,
                    warning=self._on_warning,
                )
                if self._on_model_status is not None:
                    self._on_model_status(detector.status)
                result = detector.detect(
                    self.path,
                    cancel=self._cancel_event.is_set,
                    progress=self._on_progress,
                    warning=self._on_warning,
                )
                if self._on_model_status is not None:
                    self._on_model_status(detector.status)
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

    def join(self, timeout: float | None = None) -> AnalysisWorker:
        thread = self._thread
        if thread is None:
            raise RuntimeError("AnalysisWorker has not been started")
        thread.join(timeout)
        return self

    def result(self, timeout: float | None = None) -> ShotProject:
        return self._future.result(timeout)

    def is_running(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())


__all__ = ["AnalysisWorker"]
