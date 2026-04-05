"""
core/vad_engine.py — Shabda AI
====================================
Voice Activity Detection using Silero VAD.

Thread model: runs on Thread 2.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Optional, Tuple

import numpy as np
import torch

from core.config import CONFIG

log = logging.getLogger(__name__)

_VAD_MODEL: Optional[object] = None
_VAD_UTILS: Optional[Tuple] = None
_VAD_LOCK = threading.Lock()


def _load_silero_vad() -> Tuple[object, Tuple]:
    """Load Silero VAD model from torch.hub (cached)."""
    global _VAD_MODEL, _VAD_UTILS
    with _VAD_LOCK:
        if _VAD_MODEL is None:
            log.info("Loading Silero VAD model...")
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )
            _VAD_MODEL = model
            _VAD_UTILS = utils
            log.info("Silero VAD loaded.")
    return _VAD_MODEL, _VAD_UTILS


class VADEngine:
    """
    Applies Silero VAD to incoming audio chunks.
    Emits active speech chunks continuously to `output_queue`.
    Emits an empty numpy array to signal the END of a speech utterance (i.e. silence reached).
    """

    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
    ) -> None:
        self._cfg_vad = CONFIG.vad
        self._cfg_audio = CONFIG.audio
        self._input_q = input_queue
        self._output_q = output_queue

        self._model, self._utils = _load_silero_vad()

        self._in_speech: bool = False
        self._silence_samples: int = 0
        self._speech_samples: int = 0

        self._vad_triggers: int = 0
        self._frames_processed: int = 0

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Pre-buffer to keep unverified speech until it exceeds `min_speech_duration_ms`
        self._temp_buffer: list[np.ndarray] = []

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="VADThread",
            daemon=True,
        )
        self._thread.start()
        log.info("VADEngine started.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        log.info(
            "VADEngine stopped — %d triggers, %d frames processed",
            self._vad_triggers,
            self._frames_processed,
        )

    def _run(self) -> None:
        sr = self._cfg_audio.sample_rate
        min_speech_samples = int(self._cfg_vad.min_speech_duration_ms * sr / 1000)
        min_silence_samples = int(self._cfg_vad.min_silence_duration_ms * sr / 1000)

        vad_chunk_size = 512
        incoming_buffer = np.array([], dtype=np.float32)

        while not self._stop_event.is_set():
            try:
                chunk: np.ndarray = self._input_q.get(timeout=0.1)
                incoming_buffer = np.concatenate((incoming_buffer, chunk))
            except queue.Empty:
                pass

            while len(incoming_buffer) >= vad_chunk_size:
                process_chunk = incoming_buffer[:vad_chunk_size]
                incoming_buffer = incoming_buffer[vad_chunk_size:]

                self._frames_processed += len(process_chunk)
                tensor = torch.from_numpy(process_chunk).float()

                try:
                    speech_prob: float = self._model(tensor, sr).item()
                except Exception as exc:
                    log.error("VAD inference error: %s", exc)
                    continue

                is_speech = speech_prob >= self._cfg_vad.speech_threshold

                if is_speech:
                    self._silence_samples = 0

                    if not self._in_speech:
                        self._temp_buffer.append(process_chunk)
                        self._speech_samples += len(process_chunk)

                        if self._speech_samples >= min_speech_samples:
                            self._in_speech = True
                            self._vad_triggers += 1
                            if CONFIG.logging.log_vad_triggers:
                                log.debug("VAD: speech start (trigger #%d)", self._vad_triggers)

                            for c in self._temp_buffer:
                                self._emit(c)
                            self._temp_buffer = []
                    else:
                        self._speech_samples += len(process_chunk)
                        self._emit(process_chunk)

                else:
                    if self._in_speech:
                        self._silence_samples += len(process_chunk)
                        self._emit(process_chunk)

                        if self._silence_samples >= min_silence_samples:
                            self._emit(np.array([], dtype=np.float32))
                            self._reset_state()
                    else:
                        self._temp_buffer = []
                        self._speech_samples = 0

    def _emit(self, chunk: np.ndarray) -> None:
        try:
            self._output_q.put_nowait(chunk)
        except queue.Full:
            log.warning("VAD output queue full — chunk dropped")

    def _reset_state(self) -> None:
        self._in_speech = False
        self._speech_samples = 0
        self._silence_samples = 0
        self._temp_buffer = []
