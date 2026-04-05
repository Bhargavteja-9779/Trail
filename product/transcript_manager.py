"""
product/transcript_manager.py — Shabda AI Product Layer
=========================================================
Maintains a persistent, never-erasing session transcript.

Architecture:
  - segments[]     → committed stable text (never removed)
  - _partial       → current unstable/live tail (updated each partial result)
  - _lock          → thread-safe access

Rules:
  - is_final=True  → append to segments, clear partial
  - is_final=False → update _partial only
  - Full transcript = " ".join(segments) + " " + partial (trimmed)
"""

from __future__ import annotations

import logging
import re
import threading
from typing import List

log = logging.getLogger(__name__)


class TranscriptManager:
    """
    Thread-safe transcript store.
    Segments are never removed — only the unstable tail (_partial) is updated.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._segments: List[str] = []   # Stable committed segments
        self._partial: str = ""           # Current unstable tail

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, text: str, is_final: bool) -> str:
        """
        Feed a new inference result.
        Returns the full current transcript (stable + partial).
        """
        if not text or not text.strip():
            return self.get_full_transcript()

        with self._lock:
            if is_final:
                # Dedup: avoid appending if identical to last segment
                if not self._segments or self._segments[-1] != text:
                    self._segments.append(text)
                    log.debug("TranscriptManager: committed segment [%d]: %s", len(self._segments), text[:60])
                self._partial = ""
            else:
                self._partial = text

        return self._build_transcript()

    def get_full_transcript(self) -> str:
        """Return the full session transcript (thread-safe read)."""
        with self._lock:
            return self._build_transcript()

    def get_stable_transcript(self) -> str:
        """Return only committed (final) segments."""
        with self._lock:
            return self._normalize(" ".join(self._segments))

    def get_word_count(self) -> int:
        with self._lock:
            full = self._build_transcript()
        return len(full.split()) if full else 0

    def clear(self) -> None:
        """Reset the session. Called on new session start."""
        with self._lock:
            self._segments = []
            self._partial = ""
        log.info("TranscriptManager: session cleared.")

    def get_segment_count(self) -> int:
        with self._lock:
            return len(self._segments)

    # ── Internal ────────────────────────────────────────────────────────────────

    def _build_transcript(self) -> str:
        """Builds the full displayed transcript. NOT locked — caller must hold lock."""
        parts = list(self._segments)
        if self._partial:
            parts.append(self._partial)
        return self._normalize(" ".join(parts))

    @staticmethod
    def _normalize(text: str) -> str:
        """Collapse multiple spaces, fix sentence starts."""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text).strip()
        # Capitalize after sentence-ending punctuation
        text = re.sub(r'([.!?])\s+([a-z])', lambda m: m.group(1) + " " + m.group(2).upper(), text)
        # Ensure first letter capitalized
        if text:
            text = text[0].upper() + text[1:]
        return text
