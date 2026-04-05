"""
core/post_engine.py — Shabda AI
====================================
Post-processing using RapidFuzz.
No LLMs, no rewrites, strictly formatting & keyword correction.
"""

from __future__ import annotations

import logging
import re
from typing import Dict

from rapidfuzz import fuzz, process

from core.config import CONFIG
from core.keyword_engine import KeywordEngine

log = logging.getLogger(__name__)

class PostEngine:
    """Applies phonetic keyword correction and spacing text normalisations."""

    def __init__(self, keyword_engine: KeywordEngine) -> None:
        self._cfg = CONFIG.post
        self._keyword_engine = keyword_engine

        # We fuzzy match against lowercased keys for better distance calculation.
        self._kw_targets = {k.lower(): v for k, v in self._keyword_engine.correction_map.items()}
        self._kw_keys = list(self._kw_targets.keys())

    def reload_keywords(self) -> None:
        """Sync correction map after keyword_engine.reload() is called."""
        self._kw_targets = {k.lower(): v for k, v in self._keyword_engine.correction_map.items()}
        self._kw_keys = list(self._kw_targets.keys())

    def process(self, text: str) -> str:
        """Pipeline for correcting the text."""
        if not text.strip():
            return text

        text = self._correct_keywords(text)
        text = self._fix_spacing(text)

        # Regex clean trailing VAD-static artifacts ("and", "with us", "thanks")
        text = re.sub(r'(?i)(?:\b(?:and|with us|thanks)\b\s*[\.,:]*\s*)+$', '', text)

        # Kill common Whisper silence hallucinations explicitly
        lower_t = text.lower().strip()
        hallucinations = [
            "thank you for watching", "thank you for your time",
            "thank you for your attention", "i look forward to seeing you in the next video",
            "bye.", "bye", "hello.", "hello"
        ]
        for bad_phrase in hallucinations:
            if bad_phrase in lower_t:
                return ""

        return text

    def _correct_keywords(self, text: str) -> str:
        """
        Naive fuzzy replacement: split into words and phrases,
        and replace matches above the threshold.
        """
        words = text.split()
        if not words:
            return text

        corrected_words = []

        skip_next = False
        limit = len(words)

        for i in range(limit):
            if skip_next:
                skip_next = False
                continue

            word = words[i]

            # Check 2-word phrase
            if i + 1 < limit:
                phrase = f"{word} {words[i+1]}"
                best_match = process.extractOne(
                    phrase.lower(),
                    self._kw_keys,
                    scorer=fuzz.ratio,
                    score_cutoff=self._cfg.similarity_threshold
                )
                if best_match:
                    match_key, score, _ = best_match
                    corrected_words.append(self._kw_targets[match_key])
                    skip_next = True
                    continue

            # Check 1-word
            best_match = process.extractOne(
                word.lower(),
                self._kw_keys,
                scorer=fuzz.ratio,
                score_cutoff=self._cfg.similarity_threshold
            )
            if best_match:
                match_key, score, _ = best_match
                clean_word = "".join(c for c in word if c.isalnum())
                if clean_word:
                    corrected_words.append(
                        f"{self._kw_targets[match_key]}{word[-1] if not word[-1].isalnum() else ''}"
                    )
                else:
                    corrected_words.append(self._kw_targets[match_key])
            else:
                corrected_words.append(word)

        return " ".join(corrected_words)

    def _fix_spacing(self, text: str) -> str:
        """Resolves double spaces and weird capitalization."""
        text = re.sub(r'\s+', ' ', text)
        text = text.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")

        # Ensure capitalization at start
        if len(text) > 0:
            text = text[0].upper() + text[1:]
        return text.strip()
