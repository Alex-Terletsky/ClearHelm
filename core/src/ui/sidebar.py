from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem, QSizePolicy,
)

from runner import ServiceState

from .constants import _STATE_COLORS


# ---- Agent row widget ----

class _AgentRow(QWidget):
    """Custom widget for a single agent entry in the sidebar list."""

    delete_clicked = Signal()

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        self._name_label = QLabel(name)
        self._name_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self._state_badge = QLabel("[IDLE]")
        self._state_badge.setStyleSheet(f"color: {_STATE_COLORS[ServiceState.IDLE]}; font-size: 9pt;")

        self._del_btn = QPushButton("×")
        self._del_btn.setObjectName("agentDelete")
        self._del_btn.setFixedSize(20, 20)
        self._del_btn.clicked.connect(self.delete_clicked.emit)

        layout.addWidget(self._name_label)
        layout.addWidget(self._state_badge)
        layout.addWidget(self._del_btn)

    def update_state(self, state: ServiceState):
        label = state.value.upper()
        color = _STATE_COLORS.get(state, "#888888")
        self._state_badge.setText(f"[{label}]")
        self._state_badge.setStyleSheet(f"color: {color}; font-size: 9pt;")


# ---- Model sidebar ----

class ModelSidebar(QWidget):
    """Left panel: agent list with add/load/unload controls."""

    load_requested    = Signal(str)   # agent name
    unload_requested  = Signal(str)   # agent name
    add_requested     = Signal()
    delete_requested  = Signal(str)   # agent name
    selection_changed = Signal(str)   # agent name (or "" if deselected)
    save_requested    = Signal(str)   # agent name

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: dict[str, tuple[QListWidgetItem, _AgentRow]] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        header = QLabel("Agents")
        header.setStyleSheet("font-size: 15px; font-weight: bold; color: #89b4fa;")
        layout.addWidget(header)

        self.list = QListWidget()
        self.list.currentItemChanged.connect(self._on_selection_changed)
        layout.addWidget(self.list, stretch=1)

        btn_row = QHBoxLayout()
        self.btn_add    = QPushButton("Add")
        self.btn_load   = QPushButton("Load")
        self.btn_unload = QPushButton("Unload")
        self.btn_save   = QPushButton("Save")
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_load)
        btn_row.addWidget(self.btn_unload)
        btn_row.addWidget(self.btn_save)
        layout.addLayout(btn_row)

        self.btn_add.clicked.connect(self.add_requested.emit)
        self.btn_load.clicked.connect(self._on_load)
        self.btn_unload.clicked.connect(self._on_unload)
        self.btn_save.clicked.connect(self._on_save)

    def populate(self, names: list[str]):
        self.list.clear()
        self._items.clear()
        for name in names:
            self.add_agent(name)

    def add_agent(self, name: str):
        row = _AgentRow(name)
        row.delete_clicked.connect(lambda n=name: self.delete_requested.emit(n))

        item = QListWidgetItem()
        item.setData(Qt.UserRole, name)
        item.setSizeHint(QSize(0, 32))
        self.list.addItem(item)
        self.list.setItemWidget(item, row)
        self._items[name] = (item, row)

    def remove_agent(self, name: str):
        entry = self._items.pop(name, None)
        if entry is None:
            return
        item, _ = entry
        row_idx = self.list.row(item)
        self.list.takeItem(row_idx)

    def update_status(self, name: str, state: ServiceState):
        entry = self._items.get(name)
        if entry is None:
            return
        _, row = entry
        row.update_state(state)

    def selected_model(self) -> str | None:
        item = self.list.currentItem()
        if item is None:
            return None
        return item.data(Qt.UserRole)

    def _on_selection_changed(self, current, _previous):
        name = current.data(Qt.UserRole) if current else ""
        self.selection_changed.emit(name)

    def _on_load(self):
        name = self.selected_model()
        if name:
            self.load_requested.emit(name)

    def _on_unload(self):
        name = self.selected_model()
        if name:
            self.unload_requested.emit(name)

    def _on_save(self):
        name = self.selected_model()
        if name:
            self.save_requested.emit(name)
