"""
main.py — Shabda AI Phase 1
=============================
Main CLI application. Wires up threads and queues. Provides terminal output.
Uses `sys.stdout` dynamic line clearing to give an MS-Word 'Live Dictation' rendering visual.
"""

from __future__ import annotations

import logging
import queue
import sys
import argparse
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from config import CONFIG
from device_manager import print_device_banner
from audio_engine import AudioEngine
from vad_engine import VADEngine
from buffer_engine import BufferEngine
from inference_engine import InferenceEngine
from keyword_engine import KeywordEngine
from post_engine import PostEngine

console = Console()

logging.basicConfig(
    level=logging.DEBUG if CONFIG.logging.debug_mode else logging.INFO,
    format="[dim]%(asctime)s[/dim] | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
for spam_logger in ["httpx", "huggingface_hub", "faster_whisper", "urllib3", "filelock", "huggingface_hub.utils._http"]:
    logging.getLogger(spam_logger).setLevel(logging.ERROR)
log = logging.getLogger("shabda")


def main():
    parser = argparse.ArgumentParser(description="Shabda AI Neural Engine - Live Speech to Text")
    parser.add_argument("-m", "--model", type=str, default=None, 
                        help="Specify the Whisper model size natively from the prompt.")
    args = parser.parse_args()

    model_to_use = args.model
    if not model_to_use:
        from rich.prompt import Prompt
        console.print("\n[bold cyan]🧠 Select Neural Inference Model:[/bold cyan]")
        console.print("  [1] [green]tiny.en[/green]   [dim](~100MB  | Fastest, basic accuracy)[/dim]")
        console.print("  [2] [green]base.en[/green]   [dim](~145MB  | Fast, standard baseline)[/dim]")
        console.print("  [3] [yellow]small.en[/yellow]  [dim](~480MB  | 'The Goldilocks' Best for CPU real-time)[/dim]")
        console.print("  [4] [red]medium.en[/red] [dim](~1.5GB  | High accuracy, slower on CPU)[/dim]")
        console.print("  [5] [magenta]large-v3[/magenta]  [dim](~3.0GB  | Max accuracy, requires discrete NVIDIA GPU)[/dim]")
        
        choice = Prompt.ask("\n[bold white]Enter choice[/bold white]", choices=["1", "2", "3", "4", "5"], default="4")
        
        mapping = {
            "1": "tiny.en",
            "2": "base.en",
            "3": "small.en",
            "4": "medium.en",
            "5": "large-v3"
        }
        model_to_use = mapping[choice]

    # Dynamically inject the requested model variant directly into the frozen config
    object.__setattr__(CONFIG.model, 'model_size', model_to_use)
    
    print_device_banner()
    
    keyword_engine = KeywordEngine()
    post_engine = PostEngine(keyword_engine)

    audio_queue = queue.Queue(maxsize=CONFIG.audio.queue_maxsize)
    speech_queue = queue.Queue(maxsize=50)
    inference_queue = queue.Queue(maxsize=2) # Tightly bound inference queue to ensure Live updates drop frames if lagging
    transcript_queue = queue.Queue(maxsize=50)

    audio_engine = AudioEngine()
    audio_engine.audio_queue = audio_queue
    vad_engine = VADEngine(input_queue=audio_queue, output_queue=speech_queue)
    buffer_engine = BufferEngine(input_queue=speech_queue, output_queue=inference_queue)
    inference_engine = InferenceEngine(
        input_queue=inference_queue,
        output_queue=transcript_queue,
        keyword_engine=keyword_engine,
        post_engine=post_engine
    )

    console.print("\n[bold green]Starting Shabda AI engines...[/bold green]")
    
    inference_engine.start()
    vad_engine.start()
    buffer_engine.start()
    audio_engine.start()
    
    console.print("\n[bold cyan]🎙️  Live Dictation Active...[/bold cyan] [dim](Press CTRL+C to exit)[/dim]\n")

    committed_paragraph = ""

    try:
        with Live(console=console, auto_refresh=False, vertical_overflow="visible") as live:
            # Initial setup of the empty box
            live.update(Panel("", title="[cyan]Live Audio Transcript[/cyan]", border_style="dim"), refresh=True)
            
            while True:
                try:
                    res = transcript_queue.get(timeout=0.1)
                    text = res['text']
                    if not text:
                        continue
                    
                    if res['is_final']:
                        # Append continuously to form a massive paragraph instead of line breaks
                        if committed_paragraph:
                            committed_paragraph += " " + text
                        else:
                            committed_paragraph = text
                            
                        # Refresh rendering with finalized green text
                        live.update(
                            Panel(
                                f"[bold green]{committed_paragraph}[/bold green]",
                                title="[cyan]Live Audio Transcript[/cyan]",
                                border_style="dim"
                            ), 
                            refresh=True
                        )
                    else:
                        # Display finalized context plus the current blue live-sliding text in the same seamless paragraph
                        display_text = f"[bold green]{committed_paragraph}[/bold green] [cyan]••• {text}[/cyan]" if committed_paragraph else f"[cyan]••• {text}[/cyan]"
                        live.update(
                            Panel(
                                display_text,
                                title="[cyan]Live Audio Transcript[/cyan]",
                                border_style="dim"
                            ), 
                            refresh=True
                        )
                        
                except queue.Empty:
                    pass
                
    except KeyboardInterrupt:
        console.print("\n\n[bold red]CTRL+C detected. Shutting down gracefully...[/bold red]")
    finally:
        audio_engine.stop()
        buffer_engine.stop()
        vad_engine.stop()
        inference_engine.stop()
        console.print("[bold green]Shutdown complete.[/bold green]")

    sys.exit(0)

if __name__ == "__main__":
    main()
