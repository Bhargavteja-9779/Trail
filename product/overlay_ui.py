"""
product/overlay_ui.py — Shabda AI Product Layer
=================================================
Floating always-on-top microphone overlay built with PyQt6.

Design:
  • 60×60 px circular mic button — frameless, transparent, draggable
  • States: IDLE (gray), LISTENING (red pulse), PROCESSING (green)
  • Dictation badge shown as small indicator dot when active
  • Left click  → toggle listening
  • Right click → open settings
  • Double click → toggle dictation mode
  • Drag         → move overlay anywhere on screen
  • Never steals keyboard focus (Qt.Tool)
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from PyQt6.QtCore import (
    Qt, QPoint, QTimer, QPropertyAnimation, QEasingCurve,
    pyqtSignal, QSize
)
from PyQt6.QtGui import (
    QPainter, QColor, QRadialGradient, QPen, QFont,
    QLinearGradient, QIcon, QPixmap
)
from PyQt6.QtWidgets import QWidget, QApplication, QToolTip

log = logging.getLogger(__name__)

# ── Colour palette ─────────────────────────────────────────────────────────────
COLOUR_IDLE       = QColor(80, 80, 90)
COLOUR_IDLE_RING  = QColor(110, 110, 120)
COLOUR_LISTENING  = QColor(220, 50, 50)
COLOUR_LISTEN_OUT = QColor(180, 30, 30)
COLOUR_PROCESSING = QColor(40, 190, 100)
COLOUR_PROC_OUT   = QColor(20, 150, 70)
COLOUR_DICTATION  = QColor(60, 140, 255)
COLOUR_BG         = QColor(20, 20, 28, 200)


class OverlayWindow(QWidget):
    """Floating circular microphone indicator overlay."""

    # ── Signals emitted to AppController ──────────────────────────────────────
    toggle_listening = pyqtSignal()
    open_settings    = pyqtSignal()
    toggle_dictation = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()

        self._state: str = "idle"          # "idle" | "listening" | "processing"
        self._dictation_active: bool = False
        self._pulse_phase: float = 0.0     # 0..1 for breathing animation

        self._drag_pos: Optional[QPoint] = None
        self._double_click_guard: bool = False

        self._setup_window()
        self._setup_pulse_timer()

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _setup_window(self) -> None:
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow, True)
        self.setFixedSize(76, 76)           # 60px button + 8px glow margin each side

        # Position bottom-right of primary screen by default
        screen = QApplication.primaryScreen().geometry()
        self.move(screen.width() - 120, screen.height() - 140)

        self.setToolTip("Shabda AI  |  Click: mic  |  Dbl-click: dictation  |  Right-click: settings")

    def _setup_pulse_timer(self) -> None:
        """50ms tick for breathing animation when listening / processing."""
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(50)
        self._pulse_timer.timeout.connect(self._on_pulse_tick)
        self._pulse_timer.start()

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_state(self, state: str) -> None:
        """state ∈ {"idle", "listening", "processing"}"""
        if state not in ("idle", "listening", "processing"):
            log.warning("OverlayWindow.set_state: unknown state '%s'", state)
            return
        self._state = state
        self.update()

    def set_dictation_active(self, active: bool) -> None:
        self._dictation_active = active
        self.update()

    def show_transcript_preview(self, text: str) -> None:
        """Show last ~80 chars as a tooltip near the overlay."""
        preview = ("…" + text[-77:]) if len(text) > 80 else text
        QToolTip.showText(self.mapToGlobal(QPoint(38, 0)), preview, self)

    # ── Paint ──────────────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx, cy, r = 38, 38, 26          # centre x/y, button radius

        # ── Outer glow ring (animated) ─────────────────────────────────────
        if self._state in ("listening", "processing"):
            pulse = 0.5 + 0.5 * math.sin(self._pulse_phase * math.pi * 2)
            glow_r = int(r + 6 + pulse * 6)
            glow_alpha = int(60 + pulse * 80)

            glow_colour = COLOUR_LISTENING if self._state == "listening" else COLOUR_PROCESSING
            glow_c = QColor(glow_colour)
            glow_c.setAlpha(glow_alpha)

            pen = QPen(glow_c, 2)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(cx - glow_r, cy - glow_r, glow_r * 2, glow_r * 2)

        # ── Background pill ────────────────────────────────────────────────
        bg = QColor(COLOUR_BG)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(bg)
        p.drawEllipse(cx - r - 4, cy - r - 4, (r + 4) * 2, (r + 4) * 2)

        # ── Main circle gradient ────────────────────────────────────────────
        if self._state == "idle":
            inner, outer = COLOUR_IDLE, COLOUR_IDLE_RING
        elif self._state == "listening":
            inner = COLOUR_LISTENING
            outer = COLOUR_LISTEN_OUT
        else:
            inner = COLOUR_PROCESSING
            outer = COLOUR_PROC_OUT

        grad = QRadialGradient(cx - r // 3, cy - r // 3, r * 1.4)
        grad.setColorAt(0, inner.lighter(130))
        grad.setColorAt(1, outer)
        p.setBrush(grad)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(cx - r, cy - r, r * 2, r * 2)

        # ── Mic icon (simple geometric) ────────────────────────────────────
        p.setPen(QPen(QColor(240, 240, 240, 230), 2, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.setBrush(QColor(240, 240, 240, 200))

        # Mic capsule body
        p.drawRoundedRect(cx - 5, cy - 12, 10, 16, 5, 5)

        # Stand / U-shape
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawArc(cx - 9, cy - 2, 18, 14, 0, -180 * 16)

        # Vertical stand
        p.drawLine(cx, cy + 12, cx, cy + 16)

        # Base line
        p.drawLine(cx - 5, cy + 16, cx + 5, cy + 16)

        # ── Dictation dot ──────────────────────────────────────────────────
        if self._dictation_active:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(COLOUR_DICTATION)
            p.drawEllipse(cx + 14, cy - 18, 10, 10)

        p.end()

    # ── Pulse animation ────────────────────────────────────────────────────────

    def _on_pulse_tick(self) -> None:
        if self._state in ("listening", "processing"):
            self._pulse_phase = (self._pulse_phase + 0.03) % 1.0
            self.update()

    # ── Mouse events ──────────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint()
            self._double_click_guard = False
        elif event.button() == Qt.MouseButton.RightButton:
            self.open_settings.emit()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self._drag_pos is not None:
                delta = event.globalPosition().toPoint() - self._drag_pos
                # Only treat as click if didn't move more than 5px
                if delta.manhattanLength() < 5 and not self._double_click_guard:
                    self.toggle_listening.emit()
            self._drag_pos = None

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._double_click_guard = True
            self.toggle_dictation.emit()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_pos is not None and \
                event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.globalPosition().toPoint() - self._drag_pos
            if delta.manhattanLength() >= 5:
                self.move(self.pos() + delta)
                self._drag_pos = event.globalPosition().toPoint()
