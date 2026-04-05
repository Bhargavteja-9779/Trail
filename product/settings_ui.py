"""
product/settings_ui.py — Shabda AI Product Layer
==================================================
Full settings window — non-blocking, non-modal.

Tabs:
  [General]    — dictation toggle, latency mode, model selector, debug
  [Microphone] — input device dropdown
  [Keywords]   — add / remove / edit keywords list (hot-reload on Apply)
  [Transcript] — live read-only full session transcript view

Save  → writes config/settings.json + config/keywords.json
Apply → emits settings_changed / keywords_changed signals immediately
Cancel → discards unsaved changes
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QPalette
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QLabel, QComboBox, QCheckBox, QPushButton, QListWidget,
    QListWidgetItem, QLineEdit, QTextEdit, QGroupBox,
    QFormLayout, QSplitter, QSizePolicy, QMessageBox,
    QFrame, QScrollArea
)

log = logging.getLogger(__name__)

SETTINGS_PATH = "config/settings.json"
KEYWORDS_PATH = "config/keywords.json"

DEFAULT_SETTINGS = {
    "dictation": True,
    "model": "base.en",
    "mic": "default",
    "latency_mode": "balanced",
    "debug": False,
}

MODEL_OPTIONS = ["tiny.en", "base.en", "small.en", "medium.en", "large-v3"]
LATENCY_OPTIONS = ["fast", "balanced", "accurate"]


def load_settings() -> dict:
    os.makedirs("config", exist_ok=True)
    if not os.path.exists(SETTINGS_PATH):
        _save_settings(DEFAULT_SETTINGS)
        return dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Merge with defaults to handle missing keys
        merged = dict(DEFAULT_SETTINGS)
        merged.update(data)
        return merged
    except Exception as e:
        log.error("Failed to load settings: %s", e)
        return dict(DEFAULT_SETTINGS)


def _save_settings(settings: dict) -> None:
    os.makedirs("config", exist_ok=True)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


def load_keywords() -> List[str]:
    os.makedirs("config", exist_ok=True)
    if not os.path.exists(KEYWORDS_PATH):
        return []
    try:
        with open(KEYWORDS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("keywords", [])
    except Exception as e:
        log.error("Failed to load keywords: %s", e)
        return []


def save_keywords(keywords: List[str]) -> None:
    os.makedirs("config", exist_ok=True)
    with open(KEYWORDS_PATH, "w", encoding="utf-8") as f:
        json.dump({"keywords": keywords}, f, indent=2)


class SettingsWindow(QDialog):
    """Non-blocking settings + keyword editor window."""

    settings_changed = pyqtSignal(dict)
    keywords_changed = pyqtSignal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._current_settings: dict = load_settings()
        self._current_keywords: List[str] = load_keywords()
        self._transcript_text: str = ""

        self._setup_ui()
        self._apply_dark_theme()

    # ── UI Construction ────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        self.setWindowTitle("Shabda AI — Settings")
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowCloseButtonHint |
            Qt.WindowType.WindowMinimizeButtonHint
        )
        self.resize(600, 520)
        self.setMinimumSize(500, 400)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # ── Title bar ──────────────────────────────────────────────────────
        title = QLabel("⚙️  Shabda AI Settings")
        title.setFont(QFont("Inter", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #c9d1d9; padding-bottom: 4px;")
        layout.addWidget(title)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: #30363d;")
        layout.addWidget(divider)

        # ── Tab widget ─────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_general_tab(), "  General  ")
        self._tabs.addTab(self._build_microphone_tab(), "  Microphone  ")
        self._tabs.addTab(self._build_keywords_tab(), "  Keywords  ")
        self._tabs.addTab(self._build_transcript_tab(), "  Transcript  ")
        layout.addWidget(self._tabs)

        # ── Button row ─────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setFixedWidth(90)
        self._btn_cancel.clicked.connect(self.reject)

        self._btn_apply = QPushButton("Apply")
        self._btn_apply.setFixedWidth(90)
        self._btn_apply.setDefault(False)
        self._btn_apply.clicked.connect(self._on_apply)

        self._btn_save = QPushButton("Save & Close")
        self._btn_save.setFixedWidth(120)
        self._btn_save.setDefault(True)
        self._btn_save.clicked.connect(self._on_save)

        btn_row.addWidget(self._btn_cancel)
        btn_row.addWidget(self._btn_apply)
        btn_row.addWidget(self._btn_save)

        layout.addLayout(btn_row)

    # ── Tab builders ───────────────────────────────────────────────────────────

    def _build_general_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Dictation
        group_dictation = QGroupBox("Dictation")
        g_layout = QFormLayout(group_dictation)
        self._cb_dictation = QCheckBox("Enable global typing (type into any app)")
        self._cb_dictation.setChecked(self._current_settings.get("dictation", False))
        g_layout.addRow(self._cb_dictation)
        layout.addWidget(group_dictation)

        # Model
        group_model = QGroupBox("Speech Model")
        m_layout = QFormLayout(group_model)
        self._combo_model = QComboBox()
        self._combo_model.addItems(MODEL_OPTIONS)
        current_model = self._current_settings.get("model", "medium.en")
        idx = MODEL_OPTIONS.index(current_model) if current_model in MODEL_OPTIONS else 3
        self._combo_model.setCurrentIndex(idx)
        m_layout.addRow("Model Size:", self._combo_model)

        note = QLabel("⚠️  Model changes take effect on next launch (model stays loaded).")
        note.setStyleSheet("color: #888; font-size: 11px;")
        m_layout.addRow(note)
        layout.addWidget(group_model)

        # Latency
        group_latency = QGroupBox("Latency Mode")
        l_layout = QFormLayout(group_latency)
        self._combo_latency = QComboBox()
        self._combo_latency.addItems(LATENCY_OPTIONS)
        current_lat = self._current_settings.get("latency_mode", "balanced")
        idx2 = LATENCY_OPTIONS.index(current_lat) if current_lat in LATENCY_OPTIONS else 1
        self._combo_latency.setCurrentIndex(idx2)
        l_layout.addRow("Mode:", self._combo_latency)
        layout.addWidget(group_latency)

        # Debug
        self._cb_debug = QCheckBox("Debug mode (verbose logging)")
        self._cb_debug.setChecked(self._current_settings.get("debug", False))
        layout.addWidget(self._cb_debug)

        layout.addStretch()
        return w

    def _build_microphone_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        group = QGroupBox("Input Device")
        g_layout = QFormLayout(group)

        self._combo_mic = QComboBox()
        self._combo_mic.addItem("System Default", "default")

        # Populate microphone list (lazy import to avoid slow init at module load)
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                if d["max_input_channels"] > 0:
                    self._combo_mic.addItem(f"[{i}] {d['name']}", str(i))
        except Exception as e:
            log.warning("Could not enumerate audio devices: %s", e)

        saved_mic = self._current_settings.get("mic", "default")
        for i in range(self._combo_mic.count()):
            if self._combo_mic.itemData(i) == saved_mic:
                self._combo_mic.setCurrentIndex(i)
                break

        g_layout.addRow("Microphone:", self._combo_mic)
        layout.addWidget(group)
        layout.addStretch()
        return w

    def _build_keywords_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        label = QLabel(
            "Keywords are injected as hints into each transcription call.\n"
            "Add technical terms, names, or product names for better accuracy."
        )
        label.setStyleSheet("color: #8b949e; font-size: 12px;")
        label.setWordWrap(True)
        layout.addWidget(label)

        self._kw_list = QListWidget()
        self._kw_list.setAlternatingRowColors(True)
        self._kw_list.setSortingEnabled(False)
        for kw in self._current_keywords:
            self._kw_list.addItem(kw)
        layout.addWidget(self._kw_list)

        # Input row
        input_row = QHBoxLayout()
        self._kw_input = QLineEdit()
        self._kw_input.setPlaceholderText("New keyword (e.g. PyTorch, Bhargav)…")
        self._kw_input.returnPressed.connect(self._add_keyword)
        input_row.addWidget(self._kw_input)

        btn_add = QPushButton("Add")
        btn_add.setFixedWidth(70)
        btn_add.clicked.connect(self._add_keyword)
        input_row.addWidget(btn_add)

        btn_remove = QPushButton("Remove")
        btn_remove.setFixedWidth(80)
        btn_remove.clicked.connect(self._remove_keyword)
        input_row.addWidget(btn_remove)

        layout.addLayout(input_row)

        count_label = QLabel(f"{len(self._current_keywords)} keywords loaded")
        self._kw_count_label = count_label
        count_label.setStyleSheet("color: #8b949e; font-size: 11px;")
        layout.addWidget(count_label)

        return w

    def _build_transcript_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        label = QLabel("Full session transcript (live, read-only):")
        label.setStyleSheet("color: #8b949e; font-size: 12px;")
        layout.addWidget(label)

        self._transcript_view = QTextEdit()
        self._transcript_view.setReadOnly(True)
        self._transcript_view.setFont(QFont("Menlo", 12) if os.name != "nt" else QFont("Consolas", 12))
        self._transcript_view.setStyleSheet(
            "background-color: #0d1117; color: #c9d1d9; "
            "border: 1px solid #30363d; border-radius: 6px; padding: 8px;"
        )
        self._transcript_view.setPlaceholderText("Transcript will appear here as you speak…")
        layout.addWidget(self._transcript_view)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_copy = QPushButton("Copy All")
        btn_copy.clicked.connect(self._copy_transcript)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._clear_transcript)
        btn_row.addWidget(btn_copy)
        btn_row.addWidget(btn_clear)
        layout.addLayout(btn_row)

        return w

    # ── Keyword actions ────────────────────────────────────────────────────────

    def _add_keyword(self) -> None:
        text = self._kw_input.text().strip()
        if not text:
            return
        # Avoid duplicates
        existing = [self._kw_list.item(i).text() for i in range(self._kw_list.count())]
        if text in existing:
            QMessageBox.information(self, "Duplicate", f"'{text}' is already in the list.")
            return
        self._kw_list.addItem(text)
        self._kw_input.clear()
        self._update_kw_count()

    def _remove_keyword(self) -> None:
        selected = self._kw_list.selectedItems()
        if not selected:
            return
        for item in selected:
            self._kw_list.takeItem(self._kw_list.row(item))
        self._update_kw_count()

    def _update_kw_count(self) -> None:
        count = self._kw_list.count()
        self._kw_count_label.setText(f"{count} keywords loaded")

    def _get_current_keywords_from_ui(self) -> List[str]:
        return [self._kw_list.item(i).text() for i in range(self._kw_list.count())]

    def _get_current_settings_from_ui(self) -> dict:
        return {
            "dictation": self._cb_dictation.isChecked(),
            "model": self._combo_model.currentText(),
            "mic": self._combo_mic.currentData(),
            "latency_mode": self._combo_latency.currentText(),
            "debug": self._cb_debug.isChecked(),
        }

    # ── Transcript actions ─────────────────────────────────────────────────────

    def update_transcript(self, text: str) -> None:
        """Called from AppController to keep the transcript tab live."""
        self._transcript_text = text
        if self._tabs.currentIndex() == 3:  # Only update if tab is visible
            self._transcript_view.setPlainText(text)
            # Auto-scroll to bottom
            cursor = self._transcript_view.textCursor()
            from PyQt6.QtGui import QTextCursor
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self._transcript_view.setTextCursor(cursor)

    def _copy_transcript(self) -> None:
        from PyQt6.QtWidgets import QApplication
        QApplication.clipboard().setText(self._transcript_text)

    def _clear_transcript(self) -> None:
        self._transcript_view.clear()
        self._transcript_text = ""

    # ── Button handlers ────────────────────────────────────────────────────────

    def _on_apply(self) -> None:
        settings = self._get_current_settings_from_ui()
        keywords = self._get_current_keywords_from_ui()
        self.settings_changed.emit(settings)
        self.keywords_changed.emit(keywords)
        log.info("Settings applied (not saved to disk).")

    def _on_save(self) -> None:
        settings = self._get_current_settings_from_ui()
        keywords = self._get_current_keywords_from_ui()

        _save_settings(settings)
        save_keywords(keywords)

        self.settings_changed.emit(settings)
        self.keywords_changed.emit(keywords)
        log.info("Settings saved to disk.")
        self.accept()

    # ── Dark theme ─────────────────────────────────────────────────────────────

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet("""
            QDialog, QWidget {
                background-color: #0d1117;
                color: #c9d1d9;
            }
            QTabWidget::pane {
                border: 1px solid #30363d;
                border-radius: 6px;
                background-color: #161b22;
            }
            QTabBar::tab {
                background-color: #161b22;
                color: #8b949e;
                border: 1px solid #30363d;
                border-bottom: none;
                padding: 6px 14px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: #21262d;
                color: #58a6ff;
                border-bottom: 1px solid #21262d;
            }
            QGroupBox {
                border: 1px solid #30363d;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
                color: #8b949e;
                font-size: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QComboBox, QLineEdit {
                background-color: #21262d;
                border: 1px solid #30363d;
                border-radius: 5px;
                color: #c9d1d9;
                padding: 5px 8px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #21262d;
                color: #c9d1d9;
                selection-background-color: #388bfd33;
            }
            QPushButton {
                background-color: #21262d;
                border: 1px solid #30363d;
                border-radius: 5px;
                color: #c9d1d9;
                padding: 6px 14px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #30363d;
                border-color: #58a6ff;
                color: #58a6ff;
            }
            QPushButton[default="true"] {
                background-color: #1f6feb;
                border-color: #388bfd;
                color: white;
            }
            QPushButton[default="true"]:hover {
                background-color: #388bfd;
            }
            QListWidget {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 5px;
                color: #c9d1d9;
                alternate-background-color: #1c2128;
            }
            QListWidget::item:selected {
                background-color: #388bfd33;
                color: #58a6ff;
            }
            QCheckBox {
                color: #c9d1d9;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 1px solid #30363d;
                background-color: #21262d;
            }
            QCheckBox::indicator:checked {
                background-color: #1f6feb;
                border-color: #388bfd;
            }
            QLabel {
                color: #c9d1d9;
            }
            QScrollBar:vertical {
                background: #0d1117;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #30363d;
                border-radius: 4px;
                min-height: 20px;
            }
        """)
