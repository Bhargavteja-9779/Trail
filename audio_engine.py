"""
audio_engine.py — Shabda AI Phase 1
=====================================
sounddevice-based streaming audio capture.

Design principles:
  • The sounddevice callback NEVER blocks — it pushes raw frames into a queue.
  • Overflow and mic-disconnect errors are caught and logged; engine keeps running.
  • AudioEngine is a context manager: use `with AudioEngine() as ae:` to guarantee
    the stream is cleaned up on exit.

Thread model: runs entirely on Thread 1 (the OS audio callback thread).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd

from config import CONFIG

log = logging.getLogger(__name__)


class AudioEngine:
    """
    Captures microphone audio and exposes it via a thread-safe Queue.

    Usage
    -----
    engine = AudioEngine()
    engine.start()
    # ... engine.audio_queue contains numpy float32 arrays ...
    engine.stop()

    Or as a context manager:
    with AudioEngine() as engine:
        chunk = engine.audio_queue.get()
    """

    def __init__(self) -> None:
        self._cfg = CONFIG.audio
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(
            maxsize=self._cfg.queue_maxsize
        )
        self._stream: Optional[sd.InputStream] = None
        self._running = threading.Event()
        self._overflow_count: int = 0
        self._total_frames: int = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open the audio stream and begin capturing. Non-blocking."""
        if self._running.is_set():
            log.warning("AudioEngine already running.")
            return
        self._stream = sd.InputStream(
            samplerate=self._cfg.sample_rate,
            channels=self._cfg.channels,
            blocksize=self._cfg.block_size,
            latency=self._cfg.latency,
            dtype=self._cfg.dtype,
            callback=self._audio_callback,
        )
        self._running.set()
        self._stream.start()
        log.info(
            "AudioEngine started — %d Hz, mono, block=%d, latency=%s",
            self._cfg.sample_rate,
            self._cfg.block_size,
            self._cfg.latency,
        )

    def stop(self) -> None:
        """Gracefully stop the audio stream."""
        if not self._running.is_set():
            return
        self._running.clear()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                log.warning("Error closing audio stream: %s", exc)
            finally:
                self._stream = None
        log.info(
            "AudioEngine stopped — captured %d frames, %d overflows",
            self._total_frames,
            self._overflow_count,
        )

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    def get_stats(self) -> dict:
        return {
            "total_frames": self._total_frames,
            "overflow_count": self._overflow_count,
            "queue_size": self.audio_queue.qsize(),
        }

    # ── Context manager support ──────────────────────────────────────────────

    def __enter__(self) -> "AudioEngine":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ── Internal callback (runs on OS audio thread) ──────────────────────────

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """
        Called by sounddevice on every audio block.

        MUST be non-blocking. We only copy data and enqueue it.
        Any exception here is silently swallowed by sounddevice, so we
        wrap everything and log explicitly.
        """
        try:
            if status:
                if status.input_overflow:
                    self._overflow_count += 1
                    if CONFIG.logging.log_dropped_frames:
                        log.debug("Audio overflow #%d", self._overflow_count)
                elif status.input_underflow:
                    log.debug("Audio underflow (mic signal gap)")

            # Flatten to 1-D float32, make a writable copy
            audio_chunk = indata[:, 0].copy()
            self._total_frames += frames

            try:
                self.audio_queue.put_nowait(audio_chunk)
            except queue.Full:
                # Drop oldest frame rather than blocking the audio thread
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    pass
                self.audio_queue.put_nowait(audio_chunk)
                self._overflow_count += 1
                if CONFIG.logging.log_dropped_frames:
                    log.debug("Audio queue full — oldest frame dropped")

        except Exception as exc:
            log.error("Audio callback error: %s", exc)


def wait_for_mic(timeout_seconds: float = 10.0) -> bool:
    """
    Block until a microphone is available. Returns True on success, False on timeout.
    Useful for recovery after mic disconnect.
    """
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            devices = sd.query_devices()
            for d in devices:
                if d["max_input_channels"] > 0:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False
