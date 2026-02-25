import os
import re

from PySide6.QtGui import QColor

from runner import ServiceState


# ---- Project paths ----

_UI_DIR      = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_UI_DIR)))
_MODELS_DIR  = os.path.join(_PROJECT_ROOT, "models")
_CONFIGS_DIR = os.path.join(_PROJECT_ROOT, "configs")
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "runner_config.json")
_AGENTS_DIR  = os.path.join(_PROJECT_ROOT, "agents")
_MODULES_DIR = os.path.join(_PROJECT_ROOT, "modules")


# ---- Status badge colors ----

_STATE_COLORS = {
    ServiceState.IDLE:       "#888888",
    ServiceState.LOADING:    "#d4a017",
    ServiceState.READY:      "#4caf50",
    ServiceState.GENERATING: "#d4a017",
    ServiceState.STOPPING:   "#888888",
    ServiceState.ERROR:      "#e53935",
}

_AGENT_COLORS = [
    QColor("#89b4fa"),  # blue
    QColor("#a6e3a1"),  # green
    QColor("#f38ba8"),  # red
    QColor("#f9e2af"),  # yellow
    QColor("#cba6f7"),  # mauve
    QColor("#89dceb"),  # sky
    QColor("#94e2d5"),  # teal
    QColor("#eba0ac"),  # maroon
]


# ---- Log segment parsing ----

_COMBINED_RE   = re.compile(
    r'(<basic_log>.*?</basic_log>|<llama_cpp\b[^>]*>.*?</llama_cpp>)', re.DOTALL)
_BASIC_COLOR   = QColor("#89dceb")   # Catppuccin sky — light blue
_VERBOSE_COLOR = QColor("#fab387")   # Catppuccin peach — warm light orange
_DEFAULT_COLOR = QColor("#cdd6f4")   # Catppuccin lavender — matches existing stylesheet


def _parse_segment(part: str) -> tuple[str, str]:
    """Return (kind, content) for a segment from _COMBINED_RE.split().

    kind is 'plain' | 'basic' | 'verbose'.
    Content has XML wrapper tags stripped for basic; kept as-is for verbose.
    """
    if part.startswith('<basic_log>'):
        return 'basic', part[len('<basic_log>'):-len('</basic_log>')]
    if part.startswith('<llama_cpp'):
        return 'verbose', part
    return 'plain', part


# ---- Dark-theme stylesheet ----

DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-size: 10pt;
}
QLabel {
    color: #cdd6f4;
}
QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px 6px;
    selection-background-color: #585b70;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background-color: #313244;
    color: #cdd6f4;
    selection-background-color: #585b70;
}
QPushButton {
    background-color: #45475a;
    color: #cdd6f4;
    border: 1px solid #585b70;
    border-radius: 4px;
    padding: 5px 14px;
    min-height: 22px;
}
QPushButton:hover {
    background-color: #585b70;
}
QPushButton:pressed {
    background-color: #6c7086;
}
QPushButton:disabled {
    background-color: #313244;
    color: #6c7086;
}
QListWidget {
    background-color: #181825;
    border: 1px solid #45475a;
    border-radius: 4px;
    outline: none;
}
QListWidget::item {
    padding: 6px 8px;
    border-bottom: 1px solid #313244;
}
QListWidget::item:selected {
    background-color: #45475a;
}
QCheckBox {
    spacing: 6px;
    color: #cdd6f4;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #585b70;
    border-radius: 3px;
    background-color: #313244;
}
QCheckBox::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 14px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #89b4fa;
}
QScrollArea {
    border: none;
}
QSplitter::handle {
    background-color: #585b70;
    width: 5px;
    height: 5px;
}
QSplitter::handle:hover {
    background-color: #89b4fa;
}
QPushButton#agentDelete {
    background-color: transparent;
    color: #6c7086;
    border: 1px solid transparent;
    border-radius: 3px;
    padding: 0;
    min-height: 0;
}
QPushButton#agentDelete:hover {
    color: #f38ba8;
    border-color: #f38ba8;
}
"""
