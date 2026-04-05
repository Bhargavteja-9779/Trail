"""
main.py — Shabda AI Product Entry Point
=========================================
Launches the PyQt6 application with:
  • Floating overlay microphone window
  • Hidden settings window (shown on right-click)
  • AppController wiring all signals

Run:
    python main.py

Optional CLI flags:
    --model tiny.en     Override model from command line
    --debug             Enable verbose logging
    --cli               Fallback to original terminal-only mode
"""

from __future__ import annotations

import argparse
import logging
import os
import sys


def _configure_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-22s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in [
        "httpx", "huggingface_hub", "faster_whisper", "urllib3",
        "filelock", "huggingface_hub.utils._http",
    ]:
        logging.getLogger(noisy).setLevel(logging.ERROR)


log = logging.getLogger("shabda.main")


# ── CLI fallback (legacy terminal mode) ───────────────────────────────────────

def run_cli(model_override: str | None = None) -> None:
    """Original terminal-mode dictation (Phase 1 compatibility)."""
    import queue
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.prompt import Prompt

    from core.config import CONFIG
    from core.device_manager import print_device_banner
    from core.audio_engine import AudioEngine
    from core.vad_engine import VADEngine
    from core.buffer_engine import BufferEngine
    from core.inference_engine import InferenceEngine
    from core.keyword_engine import KeywordEngine
    from core.post_engine import PostEngine

    console = Console()

    if model_override:
        model_to_use = model_override
    else:
        console.print("\n[bold cyan]🧠 Select Neural Inference Model:[/bold cyan]")
        console.print("  [1] [green]tiny.en[/green]   [dim](~100MB  | Fastest)[/dim]")
        console.print("  [2] [green]base.en[/green]   [dim](~145MB  | Fast)[/dim]")
        console.print("  [3] [yellow]small.en[/yellow]  [dim](~480MB  | Balanced)[/dim]")
        console.print("  [4] [red]medium.en[/red] [dim](~1.5GB  | High accuracy)[/dim]")
        console.print("  [5] [magenta]large-v3[/magenta]  [dim](~3.0GB  | Max accuracy)[/dim]")
        choice = Prompt.ask("\n[bold white]Enter choice[/bold white]",
                            choices=["1", "2", "3", "4", "5"], default="4")
        model_to_use = {"1": "tiny.en", "2": "base.en", "3": "small.en",
                        "4": "medium.en", "5": "large-v3"}[choice]

    object.__setattr__(CONFIG.model, 'model_size', model_to_use)
    print_device_banner()

    audio_q = queue.Queue(maxsize=CONFIG.audio.queue_maxsize)
    speech_q = queue.Queue(maxsize=50)
    inf_q = queue.Queue(maxsize=2)
    tx_q = queue.Queue(maxsize=50)

    kw = KeywordEngine()
    pe = PostEngine(kw)
    ae = AudioEngine()
    ae.audio_queue = audio_q
    vad = VADEngine(input_queue=audio_q, output_queue=speech_q)
    buf = BufferEngine(input_queue=speech_q, output_queue=inf_q)
    inf = InferenceEngine(input_queue=inf_q, output_queue=tx_q,
                          keyword_engine=kw, post_engine=pe)

    console.print("\n[bold green]Starting Shabda AI engines...[/bold green]")
    inf.start(); vad.start(); buf.start(); ae.start()
    console.print("\n[bold cyan]🎙️  Live Dictation Active...[/bold cyan] [dim](Press CTRL+C to exit)[/dim]\n")

    committed = ""
    try:
        with Live(console=console, auto_refresh=False, vertical_overflow="visible") as live:
            live.update(Panel("", title="[cyan]Live Audio Transcript[/cyan]", border_style="dim"), refresh=True)
            while True:
                try:
                    res = tx_q.get(timeout=0.1)
                    text = res.get("text", "")
                    if not text:
                        continue
                    if res["is_final"]:
                        committed = (committed + " " + text).strip() if committed else text
                        live.update(Panel(f"[bold green]{committed}[/bold green]",
                                         title="[cyan]Live Audio Transcript[/cyan]",
                                         border_style="dim"), refresh=True)
                    else:
                        display = (f"[bold green]{committed}[/bold green] [cyan]••• {text}[/cyan]"
                                   if committed else f"[cyan]••• {text}[/cyan]")
                        live.update(Panel(display, title="[cyan]Live Audio Transcript[/cyan]",
                                         border_style="dim"), refresh=True)
                except queue.Empty:
                    pass
    except KeyboardInterrupt:
        console.print("\n\n[bold red]CTRL+C detected. Shutting down...[/bold red]")
    finally:
        ae.stop(); buf.stop(); vad.stop(); inf.stop()
        console.print("[bold green]Shutdown complete.[/bold green]")
    sys.exit(0)


# ── GUI mode (main product experience) ────────────────────────────────────────

def run_gui(model_override: str | None = None) -> None:
    """Launch the overlay + settings UI product experience."""

    # Ensure we're running from project root so relative paths resolve
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt
    except ImportError:
        print("\n[ERROR] PyQt6 is not installed.\n"
              "Run:  pip install PyQt6\n"
              "Or use --cli flag for terminal-only mode.\n")
        sys.exit(1)

    from product.settings_ui import load_settings
    from product.app_controller import AppController
    from product.overlay_ui import OverlayWindow
    from product.settings_ui import SettingsWindow
    from core.config import CONFIG

    # Suppress Qt platform warnings
    os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")

    app = QApplication(sys.argv)
    app.setApplicationName("Shabda AI")
    app.setApplicationVersion("2.0")
    app.setQuitOnLastWindowClosed(False)   # Don't quit when settings window closes

    # ── Apply model override ────────────────────────────────────────────────
    settings = load_settings()
    if model_override:
        settings["model"] = model_override

    # Inject model size into frozen config
    object.__setattr__(CONFIG.model, 'model_size', settings.get("model", CONFIG.model.model_size))

    # ── Create components ──────────────────────────────────────────────────
    controller = AppController()
    overlay    = OverlayWindow()
    settings_win = SettingsWindow()

    # ── Wire signals ────────────────────────────────────────────────────────

    # Overlay → Controller
    overlay.toggle_listening.connect(controller.toggle_listening)
    overlay.open_settings.connect(settings_win.show)
    overlay.toggle_dictation.connect(_on_toggle_dictation(controller, overlay))

    # Controller → Overlay
    controller.state_changed.connect(overlay.set_state)
    controller.state_changed.connect(
        lambda s: overlay.set_dictation_active(controller.dictation_active)
    )
    controller.transcript_updated.connect(overlay.show_transcript_preview)
    controller.transcript_updated.connect(settings_win.update_transcript)

    # Settings → Controller
    settings_win.settings_changed.connect(controller.apply_settings)
    settings_win.keywords_changed.connect(controller.reload_keywords)
    settings_win.keywords_changed.connect(
        lambda kws: overlay.set_dictation_active(controller.dictation_active)
    )

    # ── Initialize controller (blocking model load) ─────────────────────────
    log.info("Loading model — please wait...")
    controller.initialize()

    # ── Show overlay ────────────────────────────────────────────────────────
    overlay.show()
    log.info("Shabda AI overlay ready. Click mic to start.")

    # ── Graceful shutdown on app quit ────────────────────────────────────────
    app.aboutToQuit.connect(controller.shutdown)

    sys.exit(app.exec())


def _on_toggle_dictation(controller, overlay):
    """Return a slot that toggles dictation and updates overlay badge."""
    def _slot():
        controller.toggle_dictation()
        overlay.set_dictation_active(controller.dictation_active)
    return _slot


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shabda AI — Local Dictation Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py                    # GUI overlay mode\n"
            "  python main.py --model small.en   # Override model\n"
            "  python main.py --cli              # Terminal only mode\n"
            "  python main.py --debug            # Verbose logging\n"
        )
    )
    parser.add_argument("-m", "--model", type=str, default=None,
                        help="Whisper model size (tiny.en / base.en / small.en / medium.en / large-v3)")
    parser.add_argument("--cli", action="store_true",
                        help="Run in terminal-only CLI mode (no GUI)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose debug logging")
    args = parser.parse_args()

    _configure_logging(debug=args.debug)

    if args.cli:
        run_cli(model_override=args.model)
    else:
        run_gui(model_override=args.model)


if __name__ == "__main__":
    main()
