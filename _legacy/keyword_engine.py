"""
keyword_engine.py — Shabda AI Phase 1
=======================================
Loads keywords from keywords.json, provides context prompts for Faster-Whisper,
and outputs a dictionary mapping phonetic matches to keywords.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List

from config import CONFIG

log = logging.getLogger(__name__)

# Default keywords if file doesn't exist
DEFAULT_KEYWORDS = [
    "Bhargav",
    "Teja",
    "Xibotix",
    "Karatsuba",
    "FastAPI",
    "PyTorch",
    "ByteTrack",
    "CrowdSense",
    "OpenCV",
]

class KeywordEngine:
    """Manages vocabulary biasing and keyword correction data."""

    def __init__(self) -> None:
        self._cfg = CONFIG
        self.keywords: List[str] = []
        self.correction_map: Dict[str, str] = {}
        
        self._load_keywords()

    def _load_keywords(self) -> None:
        path = self._cfg.keywords_path
        if not os.path.exists(path):
            log.warning("Keyword file `%s` not found. Creating default.", path)
            self._save_default()

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.keywords = data.get("keywords", [])
        except BaseException as e:
            log.error("Failed to load keywords from %s: %s", path, e)
            self.keywords = DEFAULT_KEYWORDS

        # Mapping lowercase + spaced combos to proper keywords for RapidFuzz
        self.correction_map = {}
        for kw in self.keywords:
            self.correction_map[kw] = kw
            self.correction_map[kw.lower()] = kw
            # Add simple spaced variations (e.g. "py torch" -> "PyTorch")
            # RapidFuzz handles fuzzy matching against keys natively but mapping helps
            if k_spaced := kw.replace(" ", "") != kw:
                self.correction_map[kw.replace(" ", "")] = kw

        log.info("KeywordEngine initialized with %d keywords.", len(self.keywords))

    def _save_default(self) -> None:
        path = self._cfg.keywords_path
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"keywords": DEFAULT_KEYWORDS}, f, indent=4)

    def get_initial_prompt(self) -> str:
        """
        Dynamically append user-defined keywords to the initial context prompt.
        """
        base = self._cfg.model.initial_prompt
        # Adding explicitly just to ensure everything is covered
        added = ", ".join(self.keywords)
        return f"{base} Include exactly these terms when spoken: {added}."
