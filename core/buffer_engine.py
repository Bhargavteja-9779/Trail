"""
core/buffer_engine.py — Shabda AI
=======================================
Rolling buffer for audio chunks.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

import numpy as np

from core.config import CONFIG

log = logging.getLogger(__name__)


class BufferEngine:
    """
    Maintains a continually growing audio buffer until a silence boundary is hit.
    Emits the entire accumulated array repeatedly every `ideal_seconds` to provide "Live Dictation" streaming.
    Because the array is never sliced, context is perfectly retained preventing completely dropped or split words.
    """

    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
    ) -> None:
        self._cfg = CONFIG.buffer
        self._input_q = input_queue
        self._output_q = output_queue

        self._sr = self._cfg.sample_rate
        self._ideal_samples = int(self._cfg.ideal_seconds * self._sr)

        self._buffer: list[np.ndarray] = []
        self._buffer_samples: int = 0
        self._next_emit_samples: int = self._ideal_samples

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="BufferThread",
            daemon=True,
        )
        self._thread.start()
        log.info("BufferEngine started.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        log.info("BufferEngine stopped.")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                chunk: np.ndarray = self._input_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if chunk.size == 0:
                # Silence boundary hit: flush all current data as FINAL
                if self._buffer_samples > 0:
                    self._emit_flush(is_final=True)
                else:
                    self._reset()
                continue

            self._buffer.append(chunk)
            self._buffer_samples += len(chunk)

            # LIVE DICTATION UPDATE TICK
            if self._buffer_samples >= self._next_emit_samples:
                self._emit_flush(is_final=False)
                self._next_emit_samples += self._ideal_samples

    def _emit_flush(self, is_final: bool) -> None:
        """
        Concatenate audio and emit a COPY to the inference queue.
        is_final: True if this is the end of an utterance.
        """
        if not self._buffer:
            return

        audio_data = np.concatenate(self._buffer)

        try:
            # Drain the queue of any pending non-final updates to aggressively minimize latency queue-buildup
            if not is_final:
                while not self._output_q.empty():
                    try:
                        self._output_q.get_nowait()
                    except queue.Empty:
                        break

            self._output_q.put_nowait((audio_data, is_final))
        except queue.Full:
            pass

        if is_final:
            self._reset()

    def _reset(self) -> None:
        self._buffer = []
        self._buffer_samples = 0
        self._next_emit_samples = self._ideal_samples
