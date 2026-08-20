"""Shared visual theme for the annotator's Qt chrome.

Colors deliberately echo the ones already used in the 3D scene (plotter
background, gizmo axis colors, obstacle/selection accents) so the window
chrome and the 3D content read as one coherent product instead of a default
Qt palette wrapped around a dark viewport.
"""

from __future__ import annotations

BG_APP = "#14161a"
BG_PANEL = "#1a1d22"
BG_CARD = "#20242b"
BG_FIELD = "#262b33"
BG_FIELD_HOVER = "#2c323c"
BORDER = "#2f3540"
BORDER_STRONG = "#3d4553"
TEXT_PRIMARY = "#e6e9ef"
TEXT_SECONDARY = "#97a1b3"
TEXT_MUTED = "#6b7484"
ACCENT = "#4fd8c4"
ACCENT_HOVER = "#6fe3d3"
ACCENT_PRESSED = "#3bbfac"
ACCENT_TEXT = "#0b1f1c"
DANGER = "#ef476f"
DANGER_HOVER = "#f47189"
DANGER_PRESSED = "#d13a5c"
DANGER_TEXT = "#2a0810"

FONT_STACK = (
    '"Inter", "Ubuntu", "Segoe UI", "PingFang SC", "Microsoft YaHei UI", '
    '"Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC", sans-serif'
)

STYLESHEET = f"""
QMainWindow, QWidget {{
    background: {BG_APP};
    color: {TEXT_PRIMARY};
    font-family: {FONT_STACK};
    font-size: 13px;
}}

QToolTip {{
    background: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_STRONG};
    border-radius: 6px;
    padding: 4px 8px;
}}

QScrollArea {{
    background: {BG_PANEL};
    border: 0;
}}
QScrollArea > QWidget > QWidget {{
    background: {BG_PANEL};
}}

QGroupBox {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    margin-top: 16px;
    padding: 16px 12px 12px 12px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    color: {TEXT_PRIMARY};
    background: transparent;
    font-size: 13px;
}}
QGroupBox::indicator {{
    width: 0;
    height: 0;
}}

QLabel {{
    color: {TEXT_SECONDARY};
    background: transparent;
}}

QPushButton {{
    background: {BG_FIELD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_STRONG};
    border-radius: 8px;
    padding: 7px 16px;
}}
QPushButton:hover {{
    background: {BG_FIELD_HOVER};
    border-color: {ACCENT};
}}
QPushButton:pressed {{
    background: {BG_CARD};
}}
QPushButton:disabled {{
    color: {TEXT_MUTED};
    border-color: {BORDER};
    background: {BG_CARD};
}}
QPushButton:checked {{
    border-color: {ACCENT};
    color: {ACCENT};
}}
QPushButton[cssClass="primary"] {{
    background: {ACCENT};
    color: {ACCENT_TEXT};
    border: 1px solid {ACCENT};
    font-weight: 600;
}}
QPushButton[cssClass="primary"]:hover {{
    background: {ACCENT_HOVER};
    border-color: {ACCENT_HOVER};
}}
QPushButton[cssClass="primary"]:pressed {{
    background: {ACCENT_PRESSED};
    border-color: {ACCENT_PRESSED};
}}
QPushButton[cssClass="primary"]:disabled {{
    background: {BG_CARD};
    color: {TEXT_MUTED};
    border-color: {BORDER};
}}
QPushButton[cssClass="danger"] {{
    background: transparent;
    color: {DANGER};
    border: 1px solid {DANGER};
}}
QPushButton[cssClass="danger"]:hover {{
    background: {DANGER};
    color: {DANGER_TEXT};
    border-color: {DANGER_HOVER};
}}
QPushButton[cssClass="danger"]:pressed {{
    background: {DANGER_PRESSED};
    color: {DANGER_TEXT};
    border-color: {DANGER_PRESSED};
}}
QPushButton[cssClass="danger"]:disabled {{
    background: {BG_CARD};
    color: {TEXT_MUTED};
    border-color: {BORDER};
}}

QLineEdit, QAbstractSpinBox, QComboBox {{
    background: {BG_FIELD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_STRONG};
    border-radius: 7px;
    padding: 4px 8px;
    selection-background-color: {ACCENT};
    selection-color: {ACCENT_TEXT};
}}
QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT};
}}
QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled {{
    color: {TEXT_MUTED};
    background: {BG_CARD};
}}
QComboBox::drop-down {{
    border: 0;
    width: 18px;
}}
QComboBox QAbstractItemView {{
    background: {BG_FIELD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_STRONG};
    selection-background-color: {ACCENT};
    selection-color: {ACCENT_TEXT};
    outline: 0;
}}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
    width: 14px;
    border: 0;
    background: transparent;
}}

QCheckBox {{
    color: {TEXT_SECONDARY};
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 15px;
    height: 15px;
    border: 1px solid {BORDER_STRONG};
    border-radius: 4px;
    background: {BG_FIELD};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}

QListWidget {{
    background: {BG_FIELD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 3px;
    outline: 0;
}}
QListWidget::item {{
    padding: 5px 8px;
    border-radius: 5px;
}}
QListWidget::item:hover {{
    background: {BG_FIELD_HOVER};
}}
QListWidget::item:selected {{
    background: {ACCENT};
    color: {ACCENT_TEXT};
}}

QSlider::groove:horizontal {{
    height: 5px;
    background: {BG_FIELD};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 2px;
}}
QSlider::add-page:horizontal {{
    background: {BORDER};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 13px;
    margin: -4px 0;
    background: {TEXT_PRIMARY};
    border: 2px solid {ACCENT};
    border-radius: 7px;
}}

QSplitter::handle {{
    background: {BORDER};
}}
QSplitter::handle:hover {{
    background: {ACCENT};
}}
QSplitter::handle:horizontal {{
    width: 3px;
}}

QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_STRONG};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {TEXT_MUTED};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER_STRONG};
    border-radius: 4px;
    min-width: 24px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

QStatusBar {{
    background: {BG_APP};
    color: {TEXT_MUTED};
    border-top: 1px solid {BORDER};
}}

QMenuBar {{
    background: {BG_APP};
    color: {TEXT_SECONDARY};
    border-bottom: 1px solid {BORDER};
}}
QMenuBar::item:selected {{
    background: {BG_FIELD};
    color: {TEXT_PRIMARY};
}}
QMenu {{
    background: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_STRONG};
}}
QMenu::item:selected {{
    background: {ACCENT};
    color: {ACCENT_TEXT};
}}

QFrame#jointStateCard {{
    background: {BG_CARD};
    border: 0;
}}
QLabel#jointName {{
    color: {TEXT_PRIMARY};
    font-size: 11px;
}}
QLabel#jointLimit {{
    color: {TEXT_MUTED};
    font-size: 10px;
}}
QLabel#jointValue {{
    color: {TEXT_PRIMARY};
    background: {BG_FIELD};
    border: 1px solid {BORDER_STRONG};
    border-radius: 4px;
    padding: 2px 5px;
    font-weight: 600;
}}

QLabel#sectionHint {{
    color: {TEXT_MUTED};
    font-size: 11px;
}}
"""
