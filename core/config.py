"""
core/config.py — Shabda AI
============================
Central configuration dataclass. All runtime constants live here.
No magic numbers are scattered across the codebase.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class AudioConfig:
    """sounddevice audio capture parameters."""
    sample_rate: int = 16_000          # Hz — Whisper native rate
    channels: int = 1                  # Mono
    block_size: int = 512              # Samples per callback block (Silero VAD requires 512 at 16kHz)
    latency: str = "low"               # sounddevice latency hint
    dtype: str = "float32"             # PCM format
    queue_maxsize: int = 100           # Max queued audio blocks


@dataclass(frozen=True)
class VADConfig:
    """Silero VAD configuration."""
    speech_threshold: float = 0.3      # Aggressively sensitive to detect quiet/fast speech
    min_speech_duration_ms: int = 100  # Trigger on split-second syllables
    min_silence_duration_ms: int = 2000 # Wait 2.0 full seconds of pure silence before dropping the 'Final' sentence cut
    window_size_samples: int = 512     # VAD frame size (16kHz → 32ms)


@dataclass(frozen=True)
class BufferConfig:
    """Rolling audio buffer configuration."""
    min_seconds: float = 0.1           # Unused, dropping barrier
    ideal_seconds: float = 0.4        # Super rapid tick rate for Live Dictation UI update pacing
    max_seconds: float = 10.0         # Hard maximum — flush if exceeded
    sample_rate: int = 16_000         # Must match AudioConfig


@dataclass(frozen=True)
class ModelConfig:
    """Faster-Whisper inference parameters."""
    model_size: str = "medium.en"            # 769M parameters. Maximum English dictation accuracy.
    beam_size: int = 1
    best_of: int = 1
    temperature: float = 0.0
    temperature_fallback: List[float] = field(default_factory=list) # Disable "creative" guessing on static entirely
    patience: float = 1.0
    length_penalty: float = 1.0
    repetition_penalty: float = 1.3    # Actively prohibit grammar loop hallucination
    no_repeat_ngram_size: int = 3
    compression_ratio_threshold: float = 1.9 # Tightly snap repetitive output probabilities
    log_prob_threshold: float = -0.5
    no_speech_threshold: float = 0.4
    condition_on_previous_text: bool = False # Kills the Whisper "youtube-caption hallucination" feedback loop
    word_timestamps: bool = True
    language: Optional[str] = "en"     # Force english natively

    # Context prompt injected into every transcription call
    initial_prompt: str = (
        "This conversation may include Indian English accents and technical terms "
        "such as Bhargav, Teja, Karatsuba, PyTorch, FastAPI, ByteTrack, OpenCV, "
        "machine learning, data structures, neural networks, algorithms."
    )


@dataclass(frozen=True)
class PostProcessingConfig:
    """RapidFuzz post-processing parameters."""
    similarity_threshold: float = 85.0    # Minimum score for keyword correction


@dataclass(frozen=True)
class LoggingConfig:
    """Runtime logging and diagnostics."""
    debug_mode: bool = False              # Set True for verbose per-segment logs
    log_inference_time: bool = True
    log_rtf: bool = True
    log_memory: bool = True
    log_vad_triggers: bool = True
    log_dropped_frames: bool = True


@dataclass(frozen=True)
class ShabdaConfig:
    """Top-level configuration aggregate. Import this everywhere."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    post: PostProcessingConfig = field(default_factory=PostProcessingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    keywords_path: str = "config/keywords.json"


# Module-level singleton — import and use directly.
CONFIG = ShabdaConfig()
