from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem,
    QSizePolicy, QCheckBox, QFrame,
)

from runner import ServiceState

from .constants import _STATE_COLORS, ECHO_NAME


# ---- Agent row widget ----

class _AgentRow(QWidget):
    """Custom widget for a single agent entry in the sidebar list."""

    multi_toggled = Signal(bool)

    def __init__(self, name: str, permanent: bool = False, parent=None):
        super().__init__(parent)
        self._permanent = permanent

        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)

        self._multi_cb = QCheckBox()
        self._multi_cb.setVisible(False)
        self._multi_cb.setStyleSheet("QCheckBox { spacing: 0px; background: transparent; }")
        self._multi_cb.toggled.connect(self.multi_toggled.emit)

        self._name_label = QLabel(name)
        self._name_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self._state_badge = None
        if not permanent:
            self._state_badge = QLabel("[IDLE]")
            self._state_badge.setStyleSheet(
                f"color: {_STATE_COLORS[ServiceState.IDLE]}; font-size: 9pt;")

        self._color_strip = QFrame()
        self._color_strip.setFixedWidth(4)
        self._color_strip.setVisible(False)
        self._color_strip.setStyleSheet("background-color: transparent; border-radius: 2px;")

        layout.addWidget(self._multi_cb)
        layout.addWidget(self._name_label)
        if self._state_badge is not None:
            layout.addWidget(self._state_badge)
        layout.addWidget(self._color_strip)

    def update_state(self, state: ServiceState):
        if self._permanent or self._state_badge is None:
            return
        label = state.value.upper()
        color = _STATE_COLORS.get(state, "#888888")
        self._state_badge.setText(f"[{label}]")
        self._state_badge.setStyleSheet(f"color: {color}; font-size: 9pt;")

    def set_multi_visible(self, visible: bool):
        self._multi_cb.setVisible(visible)
        self._color_strip.setVisible(visible)

    def set_agent_color(self, color: str) -> None:
        self._color_strip.setStyleSheet(
            f"background-color: {color}; border-radius: 2px;")

    def is_checked(self) -> bool:
        return self._multi_cb.isChecked()

    def set_checked(self, checked: bool):
        self._multi_cb.setChecked(checked)


# ---- Model sidebar ----

class ModelSidebar(QWidget):
    """Left panel: agent list with add/load/unload/delete controls and Multi View."""

    load_requested          = Signal(str)   # agent name
    unload_requested        = Signal(str)   # agent name
    add_requested           = Signal()
    delete_requested        = Signal(str)   # agent name
    selection_changed       = Signal(str)   # agent name (or "" if deselected)
    multi_view_changed      = Signal(bool)  # multi view active
    multi_selection_changed = Signal(list)  # list of checked agent names

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: dict[str, tuple[QListWidgetItem, _AgentRow]] = {}
        self._divider_item: QListWidgetItem | None = None
        self._echo_item: QListWidgetItem | None = None
        self._echo_row: _AgentRow | None = None
        self._current_state: ServiceState | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        header = QLabel("Agents")
        header.setStyleSheet("font-size: 15px; font-weight: bold; color: #89b4fa;")
        layout.addWidget(header)

        self.list = QListWidget()
        self.list.currentItemChanged.connect(self._on_selection_changed)
        layout.addWidget(self.list, stretch=1)

        # Multi View toggle
        self.btn_multi = QPushButton("Multi View")
        self.btn_multi.setCheckable(True)
        self.btn_multi.setChecked(False)
        self.btn_multi.setStyleSheet(
            "QPushButton { background-color: #45475a; color: #cdd6f4; "
            "border: 1px solid #585b70; border-radius: 4px; padding: 5px 14px; }"
            "QPushButton:checked { background-color: #89b4fa; color: #1e1e2e; "
            "border-color: #89b4fa; font-weight: bold; }"
            "QPushButton:hover { background-color: #585b70; }"
            "QPushButton:checked:hover { background-color: #b4d0fb; }"
        )
        self.btn_multi.clicked.connect(self._on_multi_toggled)
        layout.addWidget(self.btn_multi)

        # Select All / Select None (hidden by default)
        self._select_row = QWidget()
        select_layout = QHBoxLayout(self._select_row)
        select_layout.setContentsMargins(0, 0, 0, 0)
        self._btn_select_all = QPushButton("Select All")
        self._btn_select_none = QPushButton("Select None")
        select_layout.addWidget(self._btn_select_all)
        select_layout.addWidget(self._btn_select_none)
        self._select_row.setVisible(False)
        layout.addWidget(self._select_row)

        self._btn_select_all.clicked.connect(self._select_all)
        self._btn_select_none.clicked.connect(self._select_none)

        # Action buttons
        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("Add")
        self.btn_load_unload = QPushButton("Load")
        self.btn_delete = QPushButton("Delete")
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_load_unload)
        btn_row.addWidget(self.btn_delete)
        layout.addLayout(btn_row)

        self.btn_add.clicked.connect(self.add_requested.emit)
        self.btn_load_unload.clicked.connect(self._on_load_unload)
        self.btn_delete.clicked.connect(self._on_delete)

    # ---- Echo entry ----

    def add_echo(self):
        """Add Echo as a permanent entry below all user agents with a divider."""
        # Divider
        divider_widget = QFrame()
        divider_widget.setFrameShape(QFrame.HLine)
        divider_widget.setStyleSheet("color: #45475a;")
        divider_widget.setFixedHeight(2)

        self._divider_item = QListWidgetItem()
        self._divider_item.setFlags(Qt.NoItemFlags)
        self._divider_item.setSizeHint(QSize(0, 6))
        self.list.addItem(self._divider_item)
        self.list.setItemWidget(self._divider_item, divider_widget)

        # Echo row
        self._echo_row = _AgentRow(ECHO_NAME, permanent=True)
        self._echo_row.multi_toggled.connect(lambda _: self._emit_multi_selection())

        self._echo_item = QListWidgetItem()
        self._echo_item.setData(Qt.UserRole, ECHO_NAME)
        self._echo_item.setSizeHint(QSize(0, 32))
        self.list.addItem(self._echo_item)
        self.list.setItemWidget(self._echo_item, self._echo_row)

    # ---- Agent management ----

    def populate(self, names: list[str]):
        self.list.clear()
        self._items.clear()
        for name in names:
            self.add_agent(name)

    def add_agent(self, name: str):
        row = _AgentRow(name)
        row.multi_toggled.connect(lambda _: self._emit_multi_selection())

        # Show checkbox if multi view is active
        if self.btn_multi.isChecked():
            row.set_multi_visible(True)

        item = QListWidgetItem()
        item.setData(Qt.UserRole, name)
        item.setSizeHint(QSize(0, 32))

        # Insert above divider if it exists
        if self._divider_item is not None:
            idx = self.list.row(self._divider_item)
            self.list.insertItem(idx, item)
        else:
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

    # ---- Load/Unload button state ----

    def update_load_button(self, state: ServiceState | None):
        """Update the Load/Unload button text and enabled state for the selected agent."""
        self._current_state = state
        name = self.selected_model()
        is_echo = name == ECHO_NAME

        if is_echo or state is None:
            self.btn_load_unload.setText("Load")
            self.btn_load_unload.setEnabled(False)
            self.btn_delete.setEnabled(not is_echo)
            return

        self.btn_load_unload.setEnabled(True)
        self.btn_delete.setEnabled(True)

        if state in (ServiceState.IDLE, ServiceState.ERROR):
            self.btn_load_unload.setText("Load")
        else:
            self.btn_load_unload.setText("Unload")

    # ---- Multi View ----

    def _on_multi_toggled(self):
        active = self.btn_multi.isChecked()
        self._select_row.setVisible(active)
        for _item, row in self._items.values():
            row.set_multi_visible(active)
        if self._echo_row is not None:
            self._echo_row.set_multi_visible(active)
        if not active:
            # Uncheck all when leaving multi view
            for _item, row in self._items.values():
                row.set_checked(False)
            if self._echo_row is not None:
                self._echo_row.set_checked(False)
        self.multi_view_changed.emit(active)
        self._emit_multi_selection()

    def _set_all_checked(self, checked: bool):
        for _item, row in self._items.values():
            row.set_checked(checked)
        if self._echo_row is not None:
            self._echo_row.set_checked(checked)
        self._emit_multi_selection()

    def _select_all(self):
        self._set_all_checked(True)

    def _select_none(self):
        self._set_all_checked(False)

    def _emit_multi_selection(self):
        checked = [name for name, (_item, row) in self._items.items()
                   if row.is_checked()]
        if self._echo_row is not None and self._echo_row.is_checked():
            checked.append(ECHO_NAME)
        self.multi_selection_changed.emit(checked)

    # ---- Agent color ----

    def set_agent_color(self, name: str, color: str) -> None:
        entry = self._items.get(name)
        if entry:
            _, row = entry
            row.set_agent_color(color)
        elif name == ECHO_NAME and self._echo_row:
            self._echo_row.set_agent_color(color)

    # ---- Selection / action handlers ----

    def _on_selection_changed(self, current, _previous):
        if current is None:
            self.selection_changed.emit("")
            return
        name = current.data(Qt.UserRole)
        if name is None:
            return
        self.selection_changed.emit(name)

    def _on_load_unload(self):
        name = self.selected_model()
        if not name or name == ECHO_NAME:
            return
        if self.btn_load_unload.text() == "Load":
            self.load_requested.emit(name)
        else:
            self.unload_requested.emit(name)

    def _on_delete(self):
        name = self.selected_model()
        if name and name != ECHO_NAME:
            self.delete_requested.emit(name)
