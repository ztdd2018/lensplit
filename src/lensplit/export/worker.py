"""Thread-based project export without a GUI toolkit dependency."""

from __future__ import annotations

import threading
from concurrent.futures import Future
from typing import Callable

from lensplit.core.model import ShotProject
from lensplit.export.exporter import ExportOptions, ExportReport, FFmpegExporter


class ExportWorker:
    """Own one exporter invocation and transfer its result through a Future."""

    def __init__(
        self,
        exporter: FFmpegExporter,
        project: ShotProject,
        options: ExportOptions,
        *,
        on_progress: Callable[[float], None] | None = None,
        on_result: Callable[[ExportReport], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        on_warning: Callable[[str], None] | None = None,
    ) -> None:
        if not isinstance(exporter, FFmpegExporter):
            raise TypeError("ExportWorker requires an FFmpegExporter")
        self.exporter = exporter
        self.project = project
        self.options = options
        self._on_progress = on_progress
        self._on_result = on_result
        self._on_error = on_error
        self._on_warning = on_warning
        self._cancel_event = threading.Event()
        self._future: Future[ExportReport] = Future()
        self._thread: threading.Thread | None = None
        self._state_lock = threading.Lock()

    @property
    def future(self) -> Future[ExportReport]:
        return self._future

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def start(self) -> ExportWorker:
        with self._state_lock:
            if self._thread is not None:
                raise RuntimeError("An ExportWorker can only be started once")
            self._thread = threading.Thread(
                target=self._execute, name="lensplit-export", daemon=True
            )
            self._thread.start()
        return self

    def _execute(self) -> None:
        if not self._future.set_running_or_notify_cancel():
            return
        try:
            result = self.exporter.export_project(
                self.project,
                self.options,
                progress=self._on_progress,
                cancel=self._cancel_event.is_set,
                warning=self._on_warning,
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

    def join(self, timeout: float | None = None) -> ExportWorker:
        thread = self._thread
        if thread is None:
            raise RuntimeError("ExportWorker has not been started")
        thread.join(timeout)
        return self

    def result(self, timeout: float | None = None) -> ExportReport:
        return self._future.result(timeout)

    def is_running(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())


__all__ = ["ExportWorker"]
