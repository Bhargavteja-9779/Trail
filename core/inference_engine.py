"""
core/inference_engine.py — Shabda AI
=========================================
Wraps the Faster-Whisper model into a persistent, async inference worker.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import os
import requests
from typing import Optional

from faster_whisper import WhisperModel
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
)
from rich.console import Console

from core.config import CONFIG
from core.device_manager import get_faster_whisper_device
from core.keyword_engine import KeywordEngine
from core.post_engine import PostEngine

log = logging.getLogger(__name__)
console = Console()

class InferenceEngine:
    """ASR Inference loop using Faster-Whisper and persistent model."""

    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        keyword_engine: KeywordEngine,
        post_engine: PostEngine,
    ) -> None:
        self._cfg_model = CONFIG.model
        self._input_q = input_queue
        self._output_q = output_queue
        self._kw_engine = keyword_engine
        self._post_engine = post_engine

        self._model: Optional[WhisperModel] = None

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._global_time_start = time.monotonic()
        self._locked_language: Optional[str] = None

    def _ensure_model_downloaded(self, hf_repo: str, dest_dir: str) -> None:
        """Beautiful custom downloader to guarantee terminal UI rendering."""
        files = ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]
        base_url = f"https://huggingface.co/{hf_repo}/resolve/main/"

        os.makedirs(dest_dir, exist_ok=True)

        needs_download = False
        for f in files:
            p = os.path.join(dest_dir, f)
            if not os.path.exists(p) or os.path.getsize(p) < 100:
                needs_download = True
                break

        if not needs_download:
            return

        console.print(f"\n[bold magenta]↓ Downloading Model ({self._cfg_model.model_size}) into Local Directory ↓[/bold magenta]")

        with Progress(
            SpinnerColumn(spinner_name="dots2"),
            TextColumn("[bold blue]{task.fields[filename]:<15}"),
            BarColumn(bar_width=40),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            DownloadColumn(),
            "•",
            TransferSpeedColumn(),
            "•",
            TimeRemainingColumn(),
            console=console,
            transient=False
        ) as progress:
            for filename in files:
                filepath = os.path.join(dest_dir, filename)
                url = base_url + filename

                try:
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    console.print(f"[bold red]Fetch Failed for {filename}! Check your internet connection.[/bold red]")
                    raise e

                total_size = int(response.headers.get("content-length", 0))
                task_id = progress.add_task("download", filename=filename, total=total_size)

                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            progress.update(task_id, advance=len(chunk))

        console.print("[bold green]✔ Model completely downloaded and verified![/bold green]\n")

    def _load_model(self) -> None:
        if self._model is not None:
            return

        device, compute_type = get_faster_whisper_device()

        hf_repo = f"Systran/faster-whisper-{self._cfg_model.model_size}"
        local_dir = os.path.abspath(os.path.join("models", self._cfg_model.model_size))

        self._ensure_model_downloaded(hf_repo, local_dir)

        self._model = WhisperModel(
            model_size_or_path=local_dir,
            device=device,
            compute_type=compute_type,
            local_files_only=True,
        )

        console.print("[dim]Warming up neural network matrices...[/dim]")
        import numpy as np
        dummy_audio = np.zeros(16000, dtype=np.float32)
        self._model.transcribe(dummy_audio, beam_size=1)

    def start(self) -> None:
        self._load_model()
        self._global_time_start = time.monotonic()

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="InferenceThread",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        log.info("Inference engine stopped.")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                audio_data, is_final = self._input_q.get(timeout=0.1)
            except queue.Empty:
                continue

            self._process_chunk(audio_data, is_final)

    def _process_chunk(self, audio_data, is_final: bool) -> None:
        start_t = time.perf_counter()
        duration_s = len(audio_data) / CONFIG.buffer.sample_rate

        prompt = self._kw_engine.get_initial_prompt()

        try:
            segments, info = self._model.transcribe(
                audio_data,
                beam_size=self._cfg_model.beam_size,
                best_of=self._cfg_model.best_of,
                temperature=self._cfg_model.temperature,
                patience=self._cfg_model.patience,
                length_penalty=self._cfg_model.length_penalty,
                repetition_penalty=self._cfg_model.repetition_penalty,
                no_repeat_ngram_size=self._cfg_model.no_repeat_ngram_size,
                compression_ratio_threshold=self._cfg_model.compression_ratio_threshold,
                log_prob_threshold=self._cfg_model.log_prob_threshold,
                no_speech_threshold=self._cfg_model.no_speech_threshold,
                condition_on_previous_text=self._cfg_model.condition_on_previous_text,
                word_timestamps=self._cfg_model.word_timestamps,
                initial_prompt=prompt,
                language=self._locked_language or self._cfg_model.language
            )

            if getattr(self, '_locked_language', None) is None:
                self._locked_language = info.language
                console.print(f"[bold cyan]🔍 Language automatically locked to: '{info.language}' to prevent hallucinations.[/bold cyan]")

            text_segments = []
            segment_probs = []

            for seg in segments:
                text_segments.append(seg.text.strip())
                segment_probs.append(seg.no_speech_prob)

            raw_text = " ".join(text_segments)

            end_t = time.perf_counter()
            inference_ms = (end_t - start_t) * 1000.0

            final_text = self._post_engine.process(raw_text)
            conf = 1.0 - (sum(segment_probs) / len(segment_probs)) if segment_probs else 0.0

            if not final_text:
                return

            curr_time = time.monotonic() - self._global_time_start

            self._output_q.put_nowait({
                "text": final_text,
                "latency_ms": int(inference_ms),
                "confidence": conf,
                "start_time": curr_time - duration_s,
                "end_time": curr_time,
                "is_final": is_final
            })

        except Exception as e:
            log.error("FasterWhisper transcribing error: %s", e)
