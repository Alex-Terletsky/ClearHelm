import json

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QHBoxLayout,
    QLabel, QCheckBox, QGroupBox, QSpinBox, QDoubleSpinBox, QLineEdit,
    QTextEdit, QComboBox,
)

from params import PARAMETER_GROUPS, ModuleParam, InterceptableCommand, RunnerConfig
from runner import ServiceState
from chat_format import discover_templates

from .constants import _TEMPLATES_DIR
from .widgets import NoScrollSpinBox, NoScrollDoubleSpinBox, NoScrollComboBox

_MODULE_HEADER_STYLE = (
    "font-weight: bold; color: #a6e3a1; "
    "margin-top: 8px; margin-bottom: 2px;"
)


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
        self._module_schemas: dict[str, list[ModuleParam]] = {}
        self._module_param_widgets: dict[str, dict[str, QWidget]] = {}
        self._module_access_widgets: dict[str, QCheckBox] = {}
        self._intercept_schemas: dict[str, list[InterceptableCommand]] = {}
        self._intercept_widgets: dict[str, dict[str, QCheckBox]] = {}

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
        self._module_param_widgets.clear()
        self._module_access_widgets.clear()
        self._intercept_widgets.clear()
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

        # Module access + config + intercept sections
        self._rebuild_module_access(cfg)
        self._rebuild_module_config(cfg)
        self._rebuild_module_intercept(cfg)

    def _create_widget(self, pname: str, value) -> QWidget:
        """Build a widget for *pname*/*value* without connecting signals."""
        if pname == "chat_template":
            w = NoScrollComboBox()
            w.addItem("none")
            for t in discover_templates(_TEMPLATES_DIR):
                w.addItem(t["name"])
            w.setCurrentText(str(value))
            return w
        if pname == "system_prompt":
            w = QTextEdit()
            w.setPlainText(str(value) if value else "")
            w.setFixedHeight(80)
            return w
        if isinstance(value, bool):
            w = QCheckBox()
            w.setChecked(value)
        elif isinstance(value, int):
            w = NoScrollSpinBox()
            w.setRange(-1, 999999)
            w.setValue(value)
        elif isinstance(value, float):
            w = NoScrollDoubleSpinBox()
            w.setRange(-1.0, 999999.0)
            w.setDecimals(4)
            w.setSingleStep(0.01)
            w.setValue(value)
        else:
            w = QLineEdit(str(value) if value is not None else "")
        return w

    @staticmethod
    def _connect_widget_signal(w: QWidget, callback):
        """Connect the appropriate change signal on *w* to *callback*."""
        if isinstance(w, QComboBox):
            w.currentTextChanged.connect(lambda _v: callback())
        elif isinstance(w, QTextEdit):
            w.textChanged.connect(callback)
        elif isinstance(w, QCheckBox):
            w.stateChanged.connect(lambda _s: callback())
        elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
            w.valueChanged.connect(lambda _v: callback())
        elif isinstance(w, QLineEdit):
            w.editingFinished.connect(callback)

    def _make_editor(self, pname: str, value) -> QWidget:
        w = self._create_widget(pname, value)
        self._connect_widget_signal(w, lambda n=pname: self._on_param_changed(n))
        return w

    def _make_module_editor(self, mod_name: str, param_name: str, value) -> QWidget:
        w = self._create_widget(param_name, value)
        self._connect_widget_signal(
            w, lambda m=mod_name, p=param_name: self._on_module_param_changed(m, p))
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

        if isinstance(widget, QComboBox):
            setattr(target, pname, widget.currentText())
        elif isinstance(widget, QTextEdit):
            setattr(target, pname, widget.toPlainText())
        elif isinstance(widget, QCheckBox):
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

    # ---- Module schemas / access / config ----

    def set_module_schemas(self, schemas: dict[str, list]):
        """Called once after module load to register available schemas."""
        self._module_schemas = schemas

    def _read_widget_value(self, widget: QWidget):
        """Extract the current value from a widget."""
        if isinstance(widget, QComboBox):
            return widget.currentText()
        if isinstance(widget, QTextEdit):
            return widget.toPlainText()
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.value()
        if isinstance(widget, QLineEdit):
            return widget.text()
        return None

    def _add_module_header(self, text: str):
        sep = QLabel(text)
        sep.setStyleSheet(_MODULE_HEADER_STYLE)
        self._detail_layout.addRow(sep)

    def _rebuild_module_access(self, cfg: RunnerConfig):
        """Render a 'Module Access' section with a checkbox per installed module."""
        if not self._module_schemas:
            return
        self._add_module_header("[module] Access")
        for mod_name in self._module_schemas:
            cb = QCheckBox()
            enabled = cfg.module_access.get(mod_name, False)
            cb.setChecked(enabled)
            cb.stateChanged.connect(
                lambda state, m=mod_name: self._on_module_access_changed(m, state))
            self._module_access_widgets[mod_name] = cb
            self._detail_layout.addRow(f"{mod_name}:", cb)

    def _rebuild_module_config(self, cfg: RunnerConfig):
        """Render config fields for each installed module that has a schema."""
        for mod_name, params in self._module_schemas.items():
            if not params:
                continue
            if not cfg.module_access.get(mod_name, False):
                continue
            self._add_module_header(f"[module] {mod_name}")
            mod_cfg = cfg.module_config.get(mod_name, {})
            widgets: dict[str, QWidget] = {}
            for p in params:
                value = mod_cfg.get(p.name, p.default)
                w = self._make_module_editor(mod_name, p.name, value)
                widgets[p.name] = w
                self._detail_layout.addRow(f"{p.name}:", w)
            self._module_param_widgets[mod_name] = widgets

    def _on_module_access_changed(self, module_name: str, state: int):
        cfg = self._current_config
        if cfg is None:
            return
        cfg.module_access[module_name] = bool(state)
        self._rebuild_detail()

    def _on_module_param_changed(self, mod_name: str, param_name: str):
        cfg = self._current_config
        if cfg is None:
            return
        widgets = self._module_param_widgets.get(mod_name, {})
        widget = widgets.get(param_name)
        if widget is None:
            return
        value = self._read_widget_value(widget)
        cfg.module_config.setdefault(mod_name, {})[param_name] = value

    # ---- Intercept schemas / toggles ----

    def set_intercept_schemas(self, schemas: dict[str, list]):
        """Called once after module load to register intercept schemas."""
        self._intercept_schemas = schemas

    def _rebuild_module_intercept(self, cfg: RunnerConfig):
        """Render intercept toggles for each enabled module with an intercept schema."""
        for mod_name, commands in self._intercept_schemas.items():
            if not commands:
                continue
            if not cfg.module_access.get(mod_name, False):
                continue
            self._add_module_header(f"[module] {mod_name} — intercepts")
            mod_intercept = cfg.module_intercept.get(mod_name, {})
            widgets: dict[str, QCheckBox] = {}
            for cmd in commands:
                cb = QCheckBox()
                checked = mod_intercept.get(cmd.name, cmd.intercept)
                cb.setChecked(checked)
                cb.stateChanged.connect(
                    lambda state, m=mod_name, c=cmd.name:
                        self._on_intercept_changed(m, c, state))
                widgets[cmd.name] = cb
                label = cmd.name
                if cmd.description:
                    label = f"{cmd.name} ({cmd.description})"
                self._detail_layout.addRow(f"{label}:", cb)
            self._intercept_widgets[mod_name] = widgets

    def _on_intercept_changed(self, module_name: str, command_name: str, state: int):
        cfg = self._current_config
        if cfg is None:
            return
        cfg.module_intercept.setdefault(module_name, {})[command_name] = bool(state)
