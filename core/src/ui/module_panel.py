"""Collapsible right-side panel for module command interception requests."""

import logging
from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QSizePolicy,
)

logger = logging.getLogger(__name__)

_MAX_PENDING = 50

# ---- Styles (Catppuccin Mocha) ----

_CARD_STYLE = (
    "QFrame { background-color: #181825; border: 1px solid #45475a; "
    "border-radius: 6px; padding: 8px; margin-bottom: 4px; }"
)

_ALLOW_BTN_STYLE = (
    "QPushButton { background-color: #a6e3a1; color: #1e1e2e; "
    "font-weight: bold; border-radius: 3px; padding: 3px 10px; min-height: 20px; }"
    "QPushButton:hover { background-color: #b5eeb3; }"
)

_DENY_BTN_STYLE = (
    "QPushButton { background-color: #f38ba8; color: #1e1e2e; "
    "font-weight: bold; border-radius: 3px; padding: 3px 10px; min-height: 20px; }"
    "QPushButton:hover { background-color: #f5a0b8; }"
)

_LABEL_STYLE = "color: #cdd6f4; font-size: 9pt;"
_DIM_LABEL_STYLE = "color: #6c7086; font-size: 9pt;"
_MODULE_LABEL_STYLE = "color: #89b4fa; font-weight: bold; font-size: 9pt;"
_AGENT_LABEL_STYLE = "color: #f9e2af; font-size: 9pt;"
_HEADER_STYLE = "color: #cdd6f4; font-size: 10pt; font-weight: bold;"
_NOTE_STYLE = "color: #6c7086; font-size: 8pt; font-style: italic;"


@dataclass
class PendingRequest:
    id: int
    module_name: str
    command_name: str
    agent_name: str
    description: str
    action: Callable
    on_deny: Callable | None
    widget: QWidget


class ModulePanel(QWidget):
    """Collapsible panel showing pending module interception requests."""

    badge_changed = Signal(int)
    expanded_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._requests: dict[int, PendingRequest] = {}
        self._next_id: int = 0
        self._agent_filter: str | None = None
        self._collapsed = True

        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Content area (hidden when collapsed)
        self._content = QWidget()
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(4)

        # Header + info note
        self._header_label = QLabel("Pending Actions")
        self._header_label.setStyleSheet(_HEADER_STYLE)
        content_layout.addWidget(self._header_label)

        self._note_label = QLabel("")
        self._note_label.setStyleSheet(_NOTE_STYLE)
        self._note_label.setWordWrap(True)
        self._note_label.hide()
        content_layout.addWidget(self._note_label)

        # Scroll area for request cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        self._card_container = QWidget()
        self._card_layout = QVBoxLayout(self._card_container)
        self._card_layout.setContentsMargins(0, 0, 0, 0)
        self._card_layout.setSpacing(4)
        self._card_layout.addStretch()
        scroll.setWidget(self._card_container)
        content_layout.addWidget(scroll, stretch=1)

        root.addWidget(self._content, stretch=1)

        # Start collapsed
        self._content.hide()
        self.setMaximumWidth(0)

    # ---- Public API ----

    def add_request(self, module_name: str, command_name: str, agent_name: str,
                    description: str, action: Callable, on_deny: Callable | None = None):
        """Add a pending request card.  Auto-expand if collapsed."""
        # Cap at max pending
        if len(self._requests) >= _MAX_PENDING:
            oldest_id = min(self._requests)
            logger.warning("Module panel: max pending (%d) reached — auto-denying oldest request", _MAX_PENDING)
            self._on_deny(oldest_id)

        req_id = self._next_id
        self._next_id += 1
        card = self._create_card(req_id, module_name, command_name,
                                 agent_name, description)
        entry = PendingRequest(req_id, module_name, command_name,
                               agent_name, description, action, on_deny, card)
        self._requests[req_id] = entry

        # Insert before the stretch at the end
        self._card_layout.insertWidget(self._card_layout.count() - 1, card)
        self._apply_filter(card, agent_name)
        self._update_badge()

        # Auto-expand if collapsed
        if self._collapsed:
            self.set_expanded(True)

    def remove_requests_for_agent(self, agent_name: str):
        """Remove all pending requests for an agent (no callbacks).  For cleanup."""
        to_remove = [rid for rid, r in self._requests.items()
                     if r.agent_name == agent_name]
        for rid in to_remove:
            entry = self._requests.pop(rid)
            entry.widget.deleteLater()
        self._update_badge()

    def clear_requests(self):
        """Remove all pending requests silently (no callbacks).  For shutdown."""
        for entry in self._requests.values():
            entry.widget.deleteLater()
        self._requests.clear()
        self._update_badge()

    def set_agent_filter(self, agent_name: str | None):
        """Show only requests for the given agent (or all if None)."""
        self._agent_filter = agent_name
        for entry in self._requests.values():
            self._apply_filter(entry.widget, entry.agent_name)
        self._update_badge()

    def pending_count(self) -> int:
        return len(self._requests)

    # ---- Internal ----

    def set_expanded(self, expanded: bool):
        self._collapsed = not expanded
        self._content.setVisible(expanded)
        if expanded:
            self.setMinimumWidth(200)
            self.setMaximumWidth(16777215)  # QWIDGETSIZE_MAX
            self.resize(260, self.height())
        else:
            self.setMinimumWidth(0)
            self.setMaximumWidth(0)
        self.expanded_changed.emit(expanded)

    def _create_card(self, req_id: int, module_name: str, command_name: str,
                     agent_name: str, description: str) -> QFrame:
        card = QFrame()
        card.setStyleSheet(_CARD_STYLE)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(3)

        # Module + command
        mod_lbl = QLabel(module_name)
        mod_lbl.setStyleSheet(_MODULE_LABEL_STYLE)
        layout.addWidget(mod_lbl)

        cmd_lbl = QLabel(command_name)
        cmd_lbl.setStyleSheet(_LABEL_STYLE)
        layout.addWidget(cmd_lbl)

        # Agent
        agent_lbl = QLabel(f"from: {agent_name}")
        agent_lbl.setStyleSheet(_AGENT_LABEL_STYLE)
        layout.addWidget(agent_lbl)

        # Description (truncated)
        desc_text = description if len(description) <= 120 else description[:117] + "..."
        desc_lbl = QLabel(desc_text)
        desc_lbl.setStyleSheet(_DIM_LABEL_STYLE)
        desc_lbl.setWordWrap(True)
        layout.addWidget(desc_lbl)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        allow_btn = QPushButton("Allow")
        allow_btn.setStyleSheet(_ALLOW_BTN_STYLE)
        allow_btn.clicked.connect(lambda checked=False, rid=req_id: self._on_allow(rid))
        btn_row.addWidget(allow_btn)

        deny_btn = QPushButton("Deny")
        deny_btn.setStyleSheet(_DENY_BTN_STYLE)
        deny_btn.clicked.connect(lambda checked=False, rid=req_id: self._on_deny(rid))
        btn_row.addWidget(deny_btn)

        layout.addLayout(btn_row)
        return card

    def _resolve_request(self, req_id: int) -> PendingRequest | None:
        entry = self._requests.pop(req_id, None)
        if entry:
            entry.widget.deleteLater()
            self._update_badge()
        return entry

    def _on_allow(self, req_id: int):
        entry = self._resolve_request(req_id)
        if entry:
            entry.action()

    def _on_deny(self, req_id: int):
        entry = self._resolve_request(req_id)
        if entry and entry.on_deny:
            entry.on_deny()

    def _apply_filter(self, widget: QWidget, agent_name: str):
        """Show/hide widget based on current agent filter."""
        widget.setVisible(self._agent_filter is None or agent_name == self._agent_filter)

    def _update_badge(self):
        """Emit badge count and update note about hidden requests."""
        total = len(self._requests)
        self.badge_changed.emit(total)
        if self._agent_filter is not None and total > 0:
            visible = sum(1 for r in self._requests.values()
                          if r.agent_name == self._agent_filter)
            hidden = total - visible
            if hidden > 0:
                self._note_label.setText(f"(+{hidden} from other agents)")
                self._note_label.show()
                return
        self._note_label.hide()
