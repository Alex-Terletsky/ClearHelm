import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QDialogButtonBox,
)

from .widgets import NoScrollComboBox


class AddAgentDialog(QDialog):
    """Dialog for selecting a name, model, and preset when adding a new agent."""

    def __init__(self, available: list[dict], preset_paths: list[tuple[str, str | None]],
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Agent")
        self.setMinimumWidth(360)

        self._selected_model: dict | None = None
        self._selected_preset: str | None = None
        self._selected_name: str = ""

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Name:"))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Agent name")
        layout.addWidget(self._name_edit)

        layout.addWidget(QLabel("Model:"))
        self._model_list = QListWidget()
        for m in available:
            item = QListWidgetItem(m["name"])
            item.setData(Qt.UserRole, m)
            self._model_list.addItem(item)
        self._model_list.currentItemChanged.connect(self._on_model_selection_changed)
        layout.addWidget(self._model_list)

        if not available:
            no_models = QLabel("No models found in models/")
            no_models.setStyleSheet("color: #6c7086; font-style: italic;")
            layout.addWidget(no_models)

        layout.addWidget(QLabel("Preset:"))
        self._preset_combo = NoScrollComboBox()
        for label, path in preset_paths:
            self._preset_combo.addItem(label, userData=path)
        default_idx = self._preset_combo.findText("default")
        if default_idx >= 0:
            self._preset_combo.setCurrentIndex(default_idx)
        layout.addWidget(self._preset_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        buttons.button(QDialogButtonBox.Ok).setText("Add")
        if not available:
            buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_model_selection_changed(self, current, _previous):
        if current is None:
            return
        m = current.data(Qt.UserRole)
        # Only auto-fill if the user hasn't typed anything yet
        if not self._name_edit.text():
            stem = os.path.splitext(m["name"])[0]
            self._name_edit.setText(stem)

    def _on_accept(self):
        name = self._name_edit.text().strip()
        if not name:
            return
        item = self._model_list.currentItem()
        if item is None:
            return
        self._selected_name = name
        self._selected_model = item.data(Qt.UserRole)
        self._selected_preset = self._preset_combo.currentData()
        self.accept()

    def agent_name(self) -> str:
        return self._selected_name

    def selected_model(self) -> dict | None:
        return self._selected_model

    def selected_preset(self) -> str | None:
        return self._selected_preset
