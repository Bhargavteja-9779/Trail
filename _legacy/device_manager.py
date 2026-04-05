"""
device_manager.py — Shabda AI Phase 1
=======================================
Automatic hardware detection and compute backend selection.
Supports Apple Silicon (MPS), NVIDIA GPU (CUDA), and CPU fallback.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass
from typing import Tuple

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceProfile:
    device: str
    compute_type: str
    platform: str
    description: str


def _detect() -> DeviceProfile:
    system = platform.system()
    machine = platform.machine()

    try:
        import torch
        if torch.backends.mps.is_available():
            torch.set_float32_matmul_precision("high")
            return DeviceProfile("mps", "float16", f"macOS {platform.mac_ver()[0]} ({machine})", "Apple Silicon (MPS)")
    except Exception as exc:
        log.debug("MPS probe failed: %s", exc)

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return DeviceProfile("cuda", "float16", f"Windows/Linux {gpu_name}", f"CUDA GPU ({gpu_name})")
    except Exception as exc:
        log.debug("CUDA probe failed: %s", exc)

    compute_type = "int8_float16"
    try:
        import ctranslate2
    except Exception:
        compute_type = "int8"

    return DeviceProfile("cpu", compute_type, f"{system} ({machine})", f"CPU ({compute_type})")


DEVICE = _detect()


def print_device_banner() -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.align import Align
    
    console = Console()
    text = (
        f"[bold cyan]Hardware Device:[/bold cyan] {DEVICE.device.upper()}\n"
        f"[bold cyan]Host Platform:[/bold cyan]   {DEVICE.platform}\n"
        f"[bold cyan]Compute Engine:[/bold cyan]  {DEVICE.compute_type}"
    )
    
    panel = Panel(
        Align.center(text),
        title="[bold magenta] Shabda AI Core Engine [/bold magenta]",
        border_style="cyan",
        expand=False
    )
    
    console.print("")
    console.print(Align.center(panel))
    console.print("")


def get_faster_whisper_device() -> Tuple[str, str]:
    if DEVICE.device == "mps":
        # CTranslate2 does not natively support MPS.
        # "int8" massively accelerates inference on Apple's CPU architecture 
        # compared to standard auto/float32.
        return "cpu", "int8"
    return DEVICE.device, DEVICE.compute_type
