import json

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QHBoxLayout,
    QLabel, QCheckBox, QGroupBox, QSpinBox, QDoubleSpinBox, QLineEdit,
)

from params import PARAMETER_GROUPS, RunnerConfig
from runner import ServiceState

from .widgets import NoScrollSpinBox, NoScrollDoubleSpinBox


class ParameterPanel(QWidget):
    """Group toggles + editable parameter detail for the active model."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_config: RunnerConfig | None = None
        self._param_widgets: dict[str, QWidget] = {}
        self._warn_labels: dict[str, QLabel] = {}
        self._checkboxes: dict[str, QCheckBox] = {}
        self._model_state: ServiceState | None = None
        self._original_loading: dict[str, object] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)

        header = QLabel("Parameters")
        header.setStyleSheet("font-size: 15px; font-weight: bold; color: #89b4fa;")
        outer.addWidget(header)

        # Group toggle grid
        toggle_group = QGroupBox("Visibility Groups")
        toggle_layout = QHBoxLayout(toggle_group)

        left_col = QVBoxLayout()
        right_col = QVBoxLayout()

        groups = ["essential"] + sorted(g for g in PARAMETER_GROUPS if g != "essential")
        mid = (len(groups) + 1) // 2
        for i, gname in enumerate(groups):
            cb = QCheckBox(gname)
            if gname == "essential":
                cb.setChecked(True)
                cb.setEnabled(False)
            cb.stateChanged.connect(self._on_group_toggled)
            self._checkboxes[gname] = cb
            if i < mid:
                left_col.addWidget(cb)
            else:
                right_col.addWidget(cb)

        toggle_layout.addLayout(left_col)
        toggle_layout.addLayout(right_col)
        outer.addWidget(toggle_group)

        # Detail area (outer scroll handles overflow)
        self._detail_container = QWidget()
        self._detail_layout = QFormLayout(self._detail_container)
        self._detail_layout.setContentsMargins(6, 6, 6, 6)
        outer.addWidget(self._detail_container)

    def set_config(self, config: RunnerConfig | None):
        self._current_config = config
        if config is None:
            self._clear_detail()
            return
        for gname, cb in self._checkboxes.items():
            if gname == "essential":
                continue
            cb.blockSignals(True)
            cb.setChecked(gname in config.active_groups)
            cb.blockSignals(False)
        self._rebuild_detail()

    def _on_group_toggled(self):
        if self._current_config is None:
            return
        self._current_config.active_groups = self.active_groups()
        self._rebuild_detail()

    def active_groups(self) -> list[str]:
        groups = ["essential"]
        for gname, cb in self._checkboxes.items():
            if gname != "essential" and cb.isChecked():
                groups.append(gname)
        return groups

    def set_state(self, state: "ServiceState | None"):
        def _is_active(s):
            return s in (ServiceState.LOADING, ServiceState.READY,
                         ServiceState.GENERATING, ServiceState.STOPPING)

        prev = self._model_state
        self._model_state = state

        if state == ServiceState.LOADING and not _is_active(prev):
            self._snapshot_original()
        elif not _is_active(state) and _is_active(prev):
            self._original_loading.clear()

        self._update_warn_labels()

    def _snapshot_original(self):
        if self._current_config is None:
            return
        mc = self._current_config.model_config
        self._original_loading = {}
        for gdef in PARAMETER_GROUPS.values():
            for pname in gdef.get("loading", []):
                if hasattr(mc, pname):
                    self._original_loading[pname] = getattr(mc, pname)

    def _update_warn_label(self, pname: str):
        label = self._warn_labels.get(pname)
        if label is None:
            return
        cfg = self._current_config
        if not self._original_loading or cfg is None:
            label.setText("")
            return
        current = getattr(cfg.model_config, pname, None)
        label.setText("⟳ restart" if current != self._original_loading.get(pname) else "")

    def _update_warn_labels(self):
        for pname in self._warn_labels:
            self._update_warn_label(pname)

    def _clear_detail(self):
        self._param_widgets.clear()
        self._warn_labels.clear()
        while self._detail_layout.count():
            child = self._detail_layout.takeAt(0)
            w = child.widget()
            if w:
                w.deleteLater()

    def _rebuild_detail(self):
        self._clear_detail()
        cfg = self._current_config
        if cfg is None:
            return

        active = set(cfg.active_groups) | {"essential"}
        ordered = ["essential"] + sorted(g for g in PARAMETER_GROUPS if g != "essential" and g in active)
        params_to_show: list[tuple[str, str, object, bool]] = []

        for gname in ordered:
            gdef = PARAMETER_GROUPS.get(gname)
            if gdef is None:
                continue
            for pname in gdef.get("loading", []):
                if hasattr(cfg.model_config, pname):
                    params_to_show.append(
                        (gname, pname, getattr(cfg.model_config, pname), True)
                    )
            for pname in gdef.get("generation", []):
                if hasattr(cfg.generation_config, pname):
                    params_to_show.append(
                        (gname, pname, getattr(cfg.generation_config, pname), False)
                    )

        current_group = None
        for group, pname, value, is_loading in params_to_show:
            if group != current_group:
                current_group = group
                desc = PARAMETER_GROUPS[group]["description"]
                sep = QLabel(f"[{group}] {desc}")
                sep.setStyleSheet(
                    "font-weight: bold; color: #89b4fa; "
                    "margin-top: 8px; margin-bottom: 2px;"
                )
                self._detail_layout.addRow(sep)
            widget = self._make_editor(pname, value)
            self._param_widgets[pname] = widget
            if is_loading:
                warn_row = QHBoxLayout()
                warn_row.setContentsMargins(0, 0, 0, 0)
                warn_row.addWidget(widget, stretch=1)
                tag = QLabel("")
                tag.setStyleSheet("color: #fab387; font-size: 9pt;")
                tag.setFixedWidth(54)
                self._warn_labels[pname] = tag
                warn_row.addWidget(tag)
                wrap = QWidget()
                wrap.setLayout(warn_row)
                lbl = QLabel(f'{pname}<span style="color: #fab387;">*</span>:')
                lbl.setTextFormat(Qt.RichText)
                self._detail_layout.addRow(lbl, wrap)
                self._update_warn_label(pname)
            else:
                self._detail_layout.addRow(f"{pname}:", widget)

    def _make_editor(self, pname: str, value) -> QWidget:
        if isinstance(value, bool):
            w = QCheckBox()
            w.setChecked(value)
            w.stateChanged.connect(lambda state, n=pname: self._on_param_changed(n))
        elif isinstance(value, int):
            w = NoScrollSpinBox()
            w.setRange(-1, 999999)
            w.setValue(value)
            w.valueChanged.connect(lambda v, n=pname: self._on_param_changed(n))
        elif isinstance(value, float):
            w = NoScrollDoubleSpinBox()
            w.setRange(-1.0, 999999.0)
            w.setDecimals(4)
            w.setSingleStep(0.01)
            w.setValue(value)
            w.valueChanged.connect(lambda v, n=pname: self._on_param_changed(n))
        else:
            w = QLineEdit(str(value) if value is not None else "")
            w.editingFinished.connect(lambda n=pname: self._on_param_changed(n))
        return w

    def _on_param_changed(self, pname: str):
        cfg = self._current_config
        if cfg is None:
            return
        widget = self._param_widgets.get(pname)
        if widget is None:
            return

        target = None
        if hasattr(cfg.model_config, pname):
            target = cfg.model_config
        elif hasattr(cfg.generation_config, pname):
            target = cfg.generation_config
        if target is None:
            return

        old = getattr(target, pname)

        if isinstance(widget, QCheckBox):
            setattr(target, pname, widget.isChecked())
        elif isinstance(widget, QSpinBox):
            setattr(target, pname, widget.value())
        elif isinstance(widget, QDoubleSpinBox):
            setattr(target, pname, widget.value())
        elif isinstance(widget, QLineEdit):
            text = widget.text()
            if isinstance(old, str) or old is None:
                setattr(target, pname, text if text else None)
            elif isinstance(old, list):
                try:
                    setattr(target, pname, json.loads(text))
                except Exception:
                    pass
            else:
                setattr(target, pname, text)

        self._update_warn_label(pname)
