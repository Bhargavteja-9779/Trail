"""
product/app_controller.py — Shabda AI Product Layer
=====================================================
Central coordinator. Wires the core engine pipeline to the product UI layer.

State machine:
  IDLE → LISTENING → PROCESSING → PAUSED → IDLE

Thread model:
  Main thread   — Qt event loop, UI, settings, transcript drain timer
  AudioThread   — AudioEngine sounddevice stream
  VADThread     — VADEngine inference loop
  BufferThread  — BufferEngine accumulator
  InferenceThread — InferenceEngine Whisper loop
  DictationThread — DictationEngine keyboard typing (separate thread pool)

Key design rules:
  • Model loaded ONCE at startup — never reloaded
  • Model stays in memory across pause/resume cycles
  • All queue draining happens on the main thread via QTimer
  • No direct cross-thread Qt signal emits from worker threads
    (worker threads put to queue → QTimer drains on main thread)
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from core.config import CONFIG
from core.audio_engine import AudioEngine
from core.vad_engine import VADEngine
from core.buffer_engine import BufferEngine
from core.inference_engine import InferenceEngine
from core.keyword_engine import KeywordEngine
from core.post_engine import PostEngine
from core.device_manager import print_device_banner

from product.transcript_manager import TranscriptManager
from product.dictation_engine import DictationEngine
from product.settings_ui import load_settings

log = logging.getLogger(__name__)

# ── State constants ────────────────────────────────────────────────────────────
STATE_IDLE       = "idle"
STATE_LISTENING  = "listening"
STATE_PROCESSING = "processing"
STATE_PAUSED     = "paused"


class AppController(QObject):
    """
    Orchestrates all engine threads and signals the UI layer.
    Runs on the Qt main thread; all heavy work is on daemon threads.
    """

    # ── Signals ────────────────────────────────────────────────────────────────
    state_changed       = pyqtSignal(str)   # "idle" | "listening" | "processing"
    transcript_updated  = pyqtSignal(str)   # Full transcript text
    error_occurred      = pyqtSignal(str)   # Human-readable error message

    def __init__(self) -> None:
        super().__init__()

        self._settings = load_settings()
        self._state = STATE_IDLE
        self._model_loaded = False

        # ── Queues ─────────────────────────────────────────────────────────
        self._audio_queue     = queue.Queue(maxsize=CONFIG.audio.queue_maxsize)
        self._speech_queue    = queue.Queue(maxsize=50)
        self._inference_queue = queue.Queue(maxsize=2)
        self._transcript_queue = queue.Queue(maxsize=50)

        # ── Core engines (created once, model loaded once) ─────────────────
        self._keyword_engine  = KeywordEngine()
        self._post_engine     = PostEngine(self._keyword_engine)
        self._audio_engine:   Optional[AudioEngine]    = None
        self._vad_engine:     Optional[VADEngine]      = None
        self._buffer_engine:  Optional[BufferEngine]   = None
        self._inference_engine: Optional[InferenceEngine] = None

        # ── Product layer ──────────────────────────────────────────────────
        self._transcript_manager = TranscriptManager()
        self._dictation_engine   = DictationEngine()

        # ── QTimer to drain transcript queue on the main thread ────────────
        self._drain_timer = QTimer(self)
        self._drain_timer.setInterval(50)   # 50ms polling
        self._drain_timer.timeout.connect(self._drain_transcript_queue)

        # Apply dictation state from settings
        if self._settings.get("dictation", False):
            self._dictation_engine.enable()

    # ── Startup ────────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """
        One-time setup: load model (blocking, called before Qt event loop).
        Kicks off dictation thread.
        """
        log.info("AppController: initializing engines...")
        print_device_banner()

        # Inject model from settings
        model_size = self._settings.get("model", CONFIG.model.model_size)
        object.__setattr__(CONFIG.model, 'model_size', model_size)

        self._build_pipeline()
        self._inference_engine.start()             # Loads model HERE (blocking)
        self._model_loaded = True

        self._dictation_engine.start()
        self._drain_timer.start()

        log.info("AppController: initialized. Model loaded. Ready.")

    def _build_pipeline(self, mic_device: Optional[int] = None) -> None:
        """Instantiate all core engines."""
        self._audio_engine = AudioEngine(device=mic_device)
        self._audio_engine.audio_queue = self._audio_queue

        self._vad_engine = VADEngine(
            input_queue=self._audio_queue,
            output_queue=self._speech_queue,
        )
        self._buffer_engine = BufferEngine(
            input_queue=self._speech_queue,
            output_queue=self._inference_queue,
        )
        self._inference_engine = InferenceEngine(
            input_queue=self._inference_queue,
            output_queue=self._transcript_queue,
            keyword_engine=self._keyword_engine,
            post_engine=self._post_engine,
        )

    # ── State transitions ──────────────────────────────────────────────────────

    def start_listening(self) -> None:
        """Start audio capture + processing."""
        if self._state == STATE_LISTENING:
            return
        if not self._model_loaded:
            log.warning("AppController: model not loaded yet.")
            return

        log.info("AppController: starting listening.")
        self._vad_engine.start()
        self._buffer_engine.start()
        self._audio_engine.start()

        self._set_state(STATE_LISTENING)
        self._dictation_engine.reset()

    def stop_listening(self) -> None:
        """Pause audio capture. Model stays loaded. Transcript preserved."""
        if self._state not in (STATE_LISTENING, STATE_PROCESSING):
            return

        log.info("AppController: stopping listening (model kept in memory).")
        self._audio_engine.stop()
        self._buffer_engine.stop()
        self._vad_engine.stop()

        self._set_state(STATE_IDLE)

    def toggle_listening(self) -> None:
        """Toggle between listening and idle."""
        if self._state == STATE_LISTENING:
            self.stop_listening()
        else:
            self.start_listening()

    def toggle_dictation(self) -> None:
        """Toggle keyboard typing mode."""
        if self._dictation_engine.is_enabled:
            self._dictation_engine.disable()
            log.info("Dictation: OFF")
        else:
            self._dictation_engine.enable()
            self._dictation_engine.reset()   # Reset index when turning on
            log.info("Dictation: ON")

    @property
    def dictation_active(self) -> bool:
        return self._dictation_engine.is_enabled

    def clear_transcript(self) -> None:
        self._transcript_manager.clear()
        self._dictation_engine.reset()
        self.transcript_updated.emit("")

    # ── Settings handling ──────────────────────────────────────────────────────

    def apply_settings(self, settings: dict) -> None:
        """Called when settings window Apply/Save is clicked."""
        log.info("AppController: applying settings: %s", settings)
        self._settings = settings

        # Dictation toggle
        if settings.get("dictation", False):
            self._dictation_engine.enable()
        else:
            self._dictation_engine.disable()

        # Debug mode
        if settings.get("debug", False):
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)

        # Mic change — requires pipeline restart if actively listening
        new_mic = settings.get("mic", "default")
        mic_device = None if new_mic == "default" else int(new_mic)

        if self._audio_engine and mic_device != self._audio_engine._device:
            was_listening = self._state == STATE_LISTENING
            if was_listening:
                self.stop_listening()
            # Rebuild audio engine with new device
            self._audio_engine = AudioEngine(device=mic_device)
            self._audio_engine.audio_queue = self._audio_queue
            self._vad_engine = VADEngine(
                input_queue=self._audio_queue,
                output_queue=self._speech_queue,
            )
            self._buffer_engine = BufferEngine(
                input_queue=self._speech_queue,
                output_queue=self._inference_queue,
            )
            if was_listening:
                self.start_listening()

    def reload_keywords(self, keywords: list) -> None:
        """Hot-reload keywords without restart."""
        log.info("AppController: hot-reloading %d keywords.", len(keywords))
        # Update in-memory keyword list via KeywordEngine
        import json, os
        os.makedirs("config", exist_ok=True)
        with open("config/keywords.json", "w", encoding="utf-8") as f:
            json.dump({"keywords": keywords}, f, indent=2)
        self._keyword_engine.reload()
        self._post_engine.reload_keywords()
        log.info("Keywords reloaded — no restart required.")

    # ── Transcript queue drain (runs on main thread) ───────────────────────────

    def _drain_transcript_queue(self) -> None:
        """
        Called every 50ms by QTimer on the main thread.
        Drains the inference output queue and updates transcript + dictation.
        """
        processed = 0
        while processed < 10:  # Process up to 10 items per tick to stay responsive
            try:
                result = self._transcript_queue.get_nowait()
            except queue.Empty:
                break

            text = result.get("text", "")
            is_final = result.get("is_final", False)

            if not text:
                continue

            # Update state indicator
            if is_final:
                self._set_state(STATE_LISTENING)
            else:
                if self._state == STATE_LISTENING:
                    self._set_state(STATE_PROCESSING)

            # Persist transcript
            full_text = self._transcript_manager.update(text, is_final)

            # Feed dictation engine (only NEW characters typed)
            self._dictation_engine.update_transcript(full_text)

            # Emit to UI
            self.transcript_updated.emit(full_text)

            processed += 1

    # ── Internal ──────────────────────────────────────────────────────────────

    def _set_state(self, state: str) -> None:
        if self._state != state:
            self._state = state
            self.state_changed.emit(state)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Clean stop of all engines."""
        log.info("AppController: shutting down...")
        self._drain_timer.stop()

        if self._audio_engine and self._audio_engine.is_running:
            self._audio_engine.stop()
        if self._buffer_engine:
            self._buffer_engine.stop()
        if self._vad_engine:
            self._vad_engine.stop()
        if self._inference_engine:
            self._inference_engine.stop()
        if self._dictation_engine:
            self._dictation_engine.stop()

        log.info("AppController: shutdown complete.")

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "state": self._state,
            "word_count": self._transcript_manager.get_word_count(),
            "segment_count": self._transcript_manager.get_segment_count(),
            "dictation": self._dictation_engine.is_enabled,
        }
