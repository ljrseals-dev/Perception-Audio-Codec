"""Bounded worker pipeline for offline PCD6 jobs.

This Python implementation deliberately uses queue.Queue for safe cross-thread
handoff. It is NOT lock-free and must not be called from a WASAPI/ASIO real-time
callback. A native SPSC ring and in-memory codec API are separate future work.
"""
from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum, auto
import queue
import threading
from typing import Optional

from .pcd6 import PCD6Encoder


class JobKind(Enum):
    ENCODE = auto()
    DECODE = auto()
    SELFTEST = auto()


@dataclass(frozen=True)
class _Job:
    kind: JobKind
    source: Optional[str]
    destination: Optional[str]
    future: Future


class PCD6Worker:
    """Run bounded, serialized PCD6 file operations on a dedicated worker.

    Submit methods return Futures. The queue is bounded so callers receive
    explicit backpressure instead of unbounded memory growth. Closing stops
    accepting work and drains already accepted jobs before joining the thread.
    """

    def __init__(self, target_kbps: float = 96.0, max_pending: int = 8) -> None:
        if isinstance(max_pending, bool) or not isinstance(max_pending, int) or max_pending < 1:
            raise ValueError("max_pending must be a positive integer")
        self._encoder = PCD6Encoder(target_kbps)
        self._jobs: queue.Queue[Optional[_Job]] = queue.Queue(maxsize=max_pending)
        self._lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="pcd6-worker", daemon=False)
        self._thread.start()

    def _submit(self, kind: JobKind, source: Optional[str], destination: Optional[str]) -> Future:
        future: Future = Future()
        with self._lock:
            if self._closed:
                raise RuntimeError("worker is closed")
            try:
                self._jobs.put_nowait(_Job(kind, source, destination, future))
            except queue.Full as exc:
                raise queue.Full("PCD6 worker queue is full; retry when capacity is available") from exc
        return future

    def encode_file(self, input_wav: str, output_bit: str) -> Future:
        return self._submit(JobKind.ENCODE, input_wav, output_bit)

    def decode_file(self, input_bit: str, output_wav: str) -> Future:
        return self._submit(JobKind.DECODE, input_bit, output_wav)

    def selftest(self, output_directory: str = ".") -> Future:
        return self._submit(JobKind.SELFTEST, output_directory, None)

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                if job is None:
                    return
                if not job.future.set_running_or_notify_cancel():
                    continue
                try:
                    if job.kind is JobKind.ENCODE:
                        assert job.source is not None and job.destination is not None
                        self._encoder.encode_file(job.source, job.destination)
                    elif job.kind is JobKind.DECODE:
                        assert job.source is not None and job.destination is not None
                        self._encoder.decode_file(job.source, job.destination)
                    else:
                        assert job.source is not None
                        self._encoder.selftest(job.source)
                except BaseException as exc:
                    job.future.set_exception(exc)
                else:
                    job.future.set_result(None)
            finally:
                self._jobs.task_done()

    def close(self, wait: bool = True) -> None:
        """Stop accepting work; accepted jobs are drained before worker exit."""
        with self._lock:
            if not self._closed:
                self._closed = True
                # Blocking insertion is intentional here: this is a control
                # thread, never an audio callback. Existing work will drain.
                self._jobs.put(None)
        if wait:
            self._thread.join()

    def __enter__(self) -> "PCD6Worker":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
