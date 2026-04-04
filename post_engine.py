"""
post_engine.py — Shabda AI Phase 1
====================================
Post-processing using RapidFuzz.
No LLMs, no rewrites, strictly formatting & keyword correction.
"""

from __future__ import annotations

import logging
from typing import Dict
import re

from rapidfuzz import fuzz, process

from config import CONFIG
from keyword_engine import KeywordEngine

log = logging.getLogger(__name__)

class PostEngine:
    """Applies phonetic keyword correction and spacing text normalisations."""

    def __init__(self, keyword_engine: KeywordEngine) -> None:
        self._cfg = CONFIG.post
        self._keyword_engine = keyword_engine
        
        # We fuzzy match against lowercased keys for better distance calculation.
        self._kw_targets = {k.lower(): v for k, v in self._keyword_engine.correction_map.items()}
        self._kw_keys = list(self._kw_targets.keys())

    def process(self, text: str) -> str:
        """Pipeline for correcting the text."""
        if not text.strip():
            return text

        text = self._correct_keywords(text)
        text = self._fix_spacing(text)
        
        # Regex clean trailing VAD-static artifacts ("and", "with us", "thanks")
        text = re.sub(r'(?i)(?:\b(?:and|with us|thanks)\b\s*[\.,]*\s*)+$', '', text)
        
        # Kill common Whisper silence hallucinations explicitly
        lower_t = text.lower().strip()
        hallucinations = ["thank you for watching", "thank you for your time", "thank you for your attention", "i look forward to seeing you in the next video", "bye.", "bye", "hello.", "hello"]
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
        if not words: return text

        corrected_words = []
        
        # We can implement a simple sliding window of sizes 1 and 2 to match
        # multi-word concepts (e.g. "fast api")
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
                # We need to preserve original punctuation attached to word!
                # Simple strip for now, we attach trailing punctuation randomly matching.
                # A robust way:
                clean_word = "".join(c for c in word if c.isalnum())
                if clean_word:
                    punct_suffix = word[len(clean_word):] if word.endswith(tuple(".,?!;")) else ""
                    punct_prefix = word[:len(word)-len(clean_word)-len(punct_suffix)] if not clean_word.isalnum() else ""
                    # Actually let's keep it simple:
                    corrected_words.append(f"{self._kw_targets[match_key]}{word[-1] if not word[-1].isalnum() else ''}")
                else:
                    corrected_words.append(self._kw_targets[match_key])
            else:
                corrected_words.append(word)

        return " ".join(corrected_words)

    def _fix_spacing(self, text: str) -> str:
        """Resolves double spaces and weird capitalization."""
        # Simple spacing cleanup
        import re
        text = re.sub(r'\s+', ' ', text)
        text = text.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
        
        # Ensure capitalization at start
        if len(text) > 0:
            text = text[0].upper() + text[1:]
        return text.strip()
