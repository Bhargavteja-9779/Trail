"""
product/dictation_engine.py — Shabda AI Product Layer
=======================================================
Types new transcript text into the last focused window via pynput.

Key design decisions:
  - Only types NEW characters (delta from _last_char_index forward)
  - 5–15ms random delay between chars to avoid OS buffer flooding
  - Word boundary pauses (optional 20ms) for naturalness
  - Runs on its own dedicated thread, communicates via Queue
  - Never modifies existing typed text — append-only
  - Works identically on macOS and Windows (pynput abstraction)
"""

from __future__ import annotations

import logging
import queue
import random
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)


class DictationEngine:
    """
    Receives full transcript strings and types any new suffix into the
    currently focused application via keyboard simulation.
    """

    def __init__(self) -> None:
        self._enabled: bool = False
        self._last_char_index: int = 0
        self._keyboard = None  # Lazy-loaded to avoid import errors at startup
        self._queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=100)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background typing thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="DictationThread",
            daemon=True,
        )
        self._thread.start()
        log.info("DictationEngine started.")

    def stop(self) -> None:
        self._stop_event.set()
        self._queue.put(None)  # Unblock thread
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        log.info("DictationEngine stopped.")

    def enable(self) -> None:
        with self._lock:
            self._enabled = True
        log.info("Dictation mode ENABLED.")

    def disable(self) -> None:
        with self._lock:
            self._enabled = False
        log.info("Dictation mode DISABLED.")

    @property
    def is_enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def update_transcript(self, full_text: str) -> None:
        """
        Called whenever the TranscriptManager produces new text.
        Only the new suffix (beyond _last_char_index) is queued for typing.
        """
        with self._lock:
            if not self._enabled:
                return
            new_text = full_text[self._last_char_index:]
            if not new_text:
                return
            self._last_char_index = len(full_text)

        # Push new text to the typing thread
        try:
            self._queue.put_nowait(new_text)
        except queue.Full:
            log.warning("DictationEngine queue full — skipping chunk")

    def reset(self) -> None:
        """Reset index. Call when starting a new dictation session."""
        with self._lock:
            self._last_char_index = 0
        # Drain queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        log.info("DictationEngine: session index reset.")

    # ── Typing thread ──────────────────────────────────────────────────────────

    def _run(self) -> None:
        """Background worker that drains the queue and types text."""
        while not self._stop_event.is_set():
            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if text is None:
                break  # Shutdown sentinel

            self._type_text(text)

    def _type_text(self, text: str) -> None:
        """Type text immediately without artificial delays to minimize latency."""
        import platform
        if platform.system() == "Darwin":
            import subprocess
            try:
                safe_text = text.replace('\\', '\\\\').replace('"', '\\"')
                script = f'tell application "System Events" to keystroke "{safe_text}"'
                subprocess.run(['osascript', '-e', script], check=False)
            except Exception as e:
                log.error("DictationEngine: osascript error: %s", e)
            return

        kb = self._get_keyboard()
        if kb is None:
            log.error("DictationEngine: keyboard controller unavailable.")
            return

        try:
            # For Windows/Linux, type via pynput instantly but with a microscopic 
            # delay between characters. If it types too fast (kb.type(text)), 
            # Windows apps like Notepad will drop/swallow characters.
            for char in text:
                if self._stop_event.is_set():
                    break
                try:
                    kb.press(char)
                    kb.release(char)
                except Exception as char_err:
                    log.debug("DictationEngine: char type error '%s': %s", char, char_err)
                    continue
                
                # 1ms to 2ms microscopic delay to avoid overwhelming the OS buffer
                time.sleep(0.002)
        except Exception as exc:
            log.error("DictationEngine: typing error: %s", exc)

    def _get_keyboard(self):
        """Lazy-load pynput keyboard controller."""
        if self._keyboard is None:
            try:
                from pynput.keyboard import Controller
                self._keyboard = Controller()
            except Exception as exc:
                log.error("DictationEngine: pynput unavailable: %s", exc)
                return None
        return self._keyboard
