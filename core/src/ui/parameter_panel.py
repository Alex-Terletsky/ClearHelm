import copy
import json

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QCheckBox, QGroupBox, QSpinBox, QDoubleSpinBox, QLineEdit,
    QTextEdit, QComboBox, QScrollArea, QFrame, QMenu, QFileDialog,
    QMessageBox, QPushButton,
)

from dataclasses import fields as dc_fields

from params import PARAMETER_GROUPS, PARAM_META, ModuleParam, InterceptableCommand, RunnerConfig
from runner import ServiceState
from chat_format import discover_templates
from manager import discover_configs

from .constants import _TEMPLATES_DIR, _CONFIGS_DIR
from .widgets import NoScrollSpinBox, NoScrollDoubleSpinBox, NoScrollComboBox

_MODULE_HEADER_STYLE = (
    "font-weight: bold; color: #a6e3a1; "
    "margin-top: 8px; margin-bottom: 2px;"
)


class ParameterPanel(QWidget):
    """Sticky header with config controls + scrollable parameter detail."""

    apply_requested = Signal()
    save_config_requested = Signal()
    save_agent_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(380)
        self._live_config: RunnerConfig | None = None
        self._staged_config: RunnerConfig | None = None
        self._current_config: RunnerConfig | None = None  # always points to _staged_config
        self._agent_name: str = ""
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
        outer.setSpacing(4)

        # ---- STICKY HEADER (not scrollable) ----
        self._agent_label = QLabel("Parameters")
        self._agent_label.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #89b4fa;")
        outer.addWidget(self._agent_label)

        # Config action buttons
        _param_btn_style = (
            "QPushButton { padding: 2px 8px; min-height: 18px; font-size: 9pt; }"
        )
        config_btn_row = QHBoxLayout()
        config_btn_row.setSpacing(3)
        self._btn_apply = QPushButton("Apply")
        self._btn_undo = QPushButton("Undo")
        self._btn_load_config = QPushButton("Load Config")
        self._btn_save_config = QPushButton("Save Config")
        self._btn_save_agent = QPushButton("Save to Agent")
        for btn in (self._btn_apply, self._btn_undo, self._btn_load_config,
                    self._btn_save_config, self._btn_save_agent):
            btn.setStyleSheet(_param_btn_style)
            config_btn_row.addWidget(btn)
        outer.addLayout(config_btn_row)

        self._btn_apply.clicked.connect(self._on_apply)
        self._btn_undo.clicked.connect(self._on_undo)
        self._btn_load_config.clicked.connect(self._show_load_menu)
        self._btn_save_config.clicked.connect(self.save_config_requested.emit)
        self._btn_save_agent.clicked.connect(self.save_agent_requested.emit)

        # Thin divider
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet("color: #45475a;")
        divider.setFixedHeight(2)
        outer.addWidget(divider)

        # ---- SCROLLABLE CONTENT ----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 4, 0, 4)
        scroll_layout.setSpacing(6)

        # Group toggle grid
        toggle_group = QGroupBox("Visibility Groups")
        toggle_layout = QHBoxLayout(toggle_group)

        left_col = QVBoxLayout()
        right_col = QVBoxLayout()

        groups = ["essential"] + sorted(
            g for g in PARAMETER_GROUPS if g != "essential")
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
        scroll_layout.addWidget(toggle_group)

        # Detail area
        self._detail_container = QWidget()
        self._detail_layout = QVBoxLayout(self._detail_container)
        self._detail_layout.setContentsMargins(6, 6, 6, 6)
        self._detail_layout.setSpacing(4)
        scroll_layout.addWidget(self._detail_container)
        scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        outer.addWidget(scroll, stretch=1)

        # Initial button state
        self._update_header_buttons()

    # ---- Public API ----

    def set_config(self, config: RunnerConfig | None):
        self._live_config = config
        if config is None:
            self._staged_config = None
            self._current_config = None
            self._clear_detail()
            self._update_header_buttons()
            return
        self._staged_config = copy.deepcopy(config)
        self._current_config = self._staged_config
        self._sync_group_checkboxes()
        self._rebuild_detail()
        self._update_header_buttons()
        self._mark_applied()

    def set_agent_name(self, name: str):
        self._agent_name = name
        if name:
            self._agent_label.setText(f"{name}'s Parameters")
        else:
            self._agent_label.setText("Parameters")

    def staged_config(self) -> RunnerConfig | None:
        return self._staged_config

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

    def active_groups(self) -> list[str]:
        groups = ["essential"]
        for gname, cb in self._checkboxes.items():
            if gname != "essential" and cb.isChecked():
                groups.append(gname)
        return groups

    # ---- Pending-change highlight ----

    _APPLY_NORMAL = (
        "QPushButton { padding: 2px 8px; min-height: 18px; font-size: 9pt; }"
    )
    _APPLY_PENDING = (
        "QPushButton { padding: 2px 8px; min-height: 18px; font-size: 9pt; "
        "background-color: #89b4fa; color: #1e1e2e; font-weight: bold; }"
    )

    def _mark_pending(self):
        self._btn_apply.setStyleSheet(self._APPLY_PENDING)

    def _mark_applied(self):
        self._btn_apply.setStyleSheet(self._APPLY_NORMAL)

    # ---- Apply / Undo ----

    def _on_apply(self):
        if self._live_config is None or self._staged_config is None:
            return
        live = self._live_config
        staged = self._staged_config

        # Copy model_config fields
        for f in dc_fields(staged.model_config):
            setattr(live.model_config, f.name,
                    getattr(staged.model_config, f.name))

        # Copy generation_config fields
        for f in dc_fields(staged.generation_config):
            setattr(live.generation_config, f.name,
                    getattr(staged.generation_config, f.name))

        # Copy top-level RunnerConfig fields
        live.active_groups = list(staged.active_groups)
        live.module_config = copy.deepcopy(staged.module_config)
        live.module_access = copy.deepcopy(staged.module_access)
        live.module_intercept = copy.deepcopy(staged.module_intercept)

        self._mark_applied()
        self.apply_requested.emit()

    def _on_undo(self):
        if self._live_config is None:
            return
        self._staged_config = copy.deepcopy(self._live_config)
        self._current_config = self._staged_config
        self._sync_group_checkboxes()
        self._rebuild_detail()
        self._mark_applied()

    # ---- Load Config (popup menu) ----

    def _show_load_menu(self):
        menu = QMenu(self)
        configs = discover_configs(_CONFIGS_DIR)
        for c in configs:
            action = menu.addAction(c["name"])
            action.setData(c["path"])
        menu.addSeparator()
        menu.addAction("Load from file...")

        chosen = menu.exec(self._btn_load_config.mapToGlobal(
            self._btn_load_config.rect().bottomLeft()))
        if chosen is None:
            return

        if chosen.text() == "Load from file...":
            path, _ = QFileDialog.getOpenFileName(
                self, "Load Config", _CONFIGS_DIR, "JSON Files (*.json)")
            if not path:
                return
        else:
            path = chosen.data()

        self._load_config_into_staging(path)

    def _load_config_into_staging(self, path: str):
        if self._staged_config is None:
            return
        try:
            saved_path = self._staged_config.model_config.model_path
            saved_name = self._staged_config.model_name
            new_cfg = RunnerConfig.from_file(path, model_path=saved_path)

            self._staged_config.active_groups = new_cfg.active_groups
            self._staged_config.model_config = new_cfg.model_config
            self._staged_config.generation_config = new_cfg.generation_config
            self._staged_config.module_config = new_cfg.module_config
            self._staged_config.module_access = new_cfg.module_access
            self._staged_config.module_intercept = new_cfg.module_intercept
            self._staged_config.model_name = saved_name
            self._current_config = self._staged_config

            self._sync_group_checkboxes()
            self._rebuild_detail()
            if self._staged_config == self._live_config:
                self._mark_applied()
            else:
                self._mark_pending()
        except Exception as e:
            QMessageBox.warning(self, "Config Error", str(e))

    # ---- Header buttons state ----

    def _update_header_buttons(self):
        enabled = self._live_config is not None
        self._btn_apply.setEnabled(enabled)
        self._btn_undo.setEnabled(enabled)
        self._btn_load_config.setEnabled(enabled)
        self._btn_save_config.setEnabled(enabled)
        self._btn_save_agent.setEnabled(enabled)

    # ---- Group toggles ----

    def _sync_group_checkboxes(self):
        """Sync group checkboxes to match staged config without triggering signals."""
        for gname, cb in self._checkboxes.items():
            if gname == "essential":
                continue
            cb.blockSignals(True)
            cb.setChecked(gname in self._staged_config.active_groups)
            cb.blockSignals(False)

    def _on_group_toggled(self):
        if self._current_config is None:
            return
        self._current_config.active_groups = self.active_groups()
        self._rebuild_detail()
        self._mark_pending()

    # ---- Warn labels (loading param restart detection) ----

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
        widget = self._warn_labels.get(pname)
        if widget is None:
            return
        cfg = self._current_config
        changed = (
            self._original_loading and cfg is not None
            and getattr(cfg.model_config, pname, None)
            != self._original_loading.get(pname))
        desc = PARAM_META.get(pname, {}).get("description", "")
        reload_tip = "Requires model reload to take effect"
        if isinstance(widget, QCheckBox):
            if changed:
                widget.setText(f"{pname}* \u27f3")
                widget.setStyleSheet("color: #fab387;")
                tip = f"{desc}\n\n{reload_tip}" if desc else reload_tip
            else:
                widget.setText(f"{pname}*")
                widget.setStyleSheet("")
                tip = desc
            widget.setToolTip(tip)
        else:
            if changed:
                widget.setText(
                    f'<span style="color: #fab387;">'
                    f'{pname}* \u27f3:</span>')
                tip = f"{desc}\n\n{reload_tip}" if desc else reload_tip
            else:
                widget.setText(
                    f'{pname}<span style="color: #fab387;">*</span>:')
                tip = desc
            widget.setToolTip(tip)

    def _update_warn_labels(self):
        for pname in self._warn_labels:
            self._update_warn_label(pname)

    # ---- Detail area ----

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
        ordered = ["essential"] + sorted(
            g for g in PARAMETER_GROUPS
            if g != "essential" and g in active)

        # Collect parameters grouped
        grouped_params: list[tuple[str, list[tuple[str, object, bool]]]] = []
        for gname in ordered:
            gdef = PARAMETER_GROUPS.get(gname)
            if gdef is None:
                continue
            group_items: list[tuple[str, object, bool]] = []
            for pname in gdef.get("loading", []):
                if hasattr(cfg.model_config, pname):
                    group_items.append(
                        (pname, getattr(cfg.model_config, pname), True))
            for pname in gdef.get("generation", []):
                if hasattr(cfg.generation_config, pname):
                    group_items.append(
                        (pname, getattr(cfg.generation_config, pname), False))
            if group_items:
                grouped_params.append((gname, group_items))

        for group, items in grouped_params:
            desc = PARAMETER_GROUPS[group]["description"]
            sep = QLabel(f"[{group}] {desc}")
            sep.setStyleSheet(
                "font-weight: bold; color: #89b4fa; "
                "margin-top: 8px; margin-bottom: 2px;")
            self._detail_layout.addWidget(sep)

            # Buffer non-loading booleans and numerics for side-by-side layout
            bool_buffer: list[tuple[str, object]] = []
            numeric_buffer: list[tuple[str, object]] = []

            for pname, value, is_loading in items:
                # Booleans: buffer up to 4 per row
                if isinstance(value, bool):
                    if numeric_buffer:
                        self._flush_numeric_buffer(numeric_buffer)
                        numeric_buffer.clear()
                    bool_buffer.append((pname, value, is_loading))
                    if len(bool_buffer) == 4:
                        self._add_bool_row(bool_buffer)
                        bool_buffer.clear()
                    continue

                # Numerics: buffer in pairs
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if bool_buffer:
                        self._add_bool_row(bool_buffer)
                        bool_buffer.clear()
                    numeric_buffer.append((pname, value, is_loading))
                    if len(numeric_buffer) == 2:
                        self._add_numeric_pair(numeric_buffer[0], numeric_buffer[1])
                        numeric_buffer.clear()
                    continue

                # Flush any pending buffers before a full-width param
                if bool_buffer:
                    self._add_bool_row(bool_buffer)
                    bool_buffer.clear()
                if numeric_buffer:
                    self._flush_numeric_buffer(numeric_buffer)
                    numeric_buffer.clear()

                widget = self._make_editor(pname, value)
                self._param_widgets[pname] = widget
                if is_loading:
                    lbl = QLabel(
                        f'{pname}<span style="color: #fab387;">*</span>:')
                    lbl.setTextFormat(Qt.RichText)
                    self._warn_labels[pname] = lbl
                else:
                    lbl = QLabel(f"{pname}:")
                row = QHBoxLayout()
                row.setContentsMargins(0, 0, 0, 0)
                row.addWidget(lbl)
                row.addWidget(widget, stretch=1)
                wrap = QWidget()
                wrap.setLayout(row)
                self._detail_layout.addWidget(wrap)
                if is_loading:
                    self._update_warn_label(pname)
                tip = PARAM_META.get(pname, {}).get("description", "")
                if tip:
                    lbl.setToolTip(tip)

            # Flush trailing buffers
            if bool_buffer:
                self._add_bool_row(bool_buffer)
            if numeric_buffer:
                self._flush_numeric_buffer(numeric_buffer)

        # Module access + config + intercept sections
        self._rebuild_module_access(cfg)
        self._rebuild_module_config(cfg)
        self._rebuild_module_intercept(cfg)

    def _add_bool_row(self, items: list[tuple]):
        """Add 1-4 checkboxes in a single row. Each item is (name, value, is_loading)."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        for name, val, is_loading in items:
            text = f"{name}*" if is_loading else name
            cb = QCheckBox(text)
            cb.setChecked(val)
            tip = PARAM_META.get(name, {}).get("description", "")
            if tip:
                cb.setToolTip(tip)
            if is_loading:
                self._warn_labels[name] = cb
                self._update_warn_label(name)
            self._connect_widget_signal(
                cb, lambda n=name: self._on_param_changed(n))
            self._param_widgets[name] = cb
            row.addWidget(cb)
        wrap = QWidget()
        wrap.setLayout(row)
        self._detail_layout.addWidget(wrap)

    def _make_numeric_half(self, name: str, val, is_loading: bool) -> QWidget:
        """Build a label + editor widget pair, returning the wrapper."""
        w = self._make_editor(name, val)
        self._param_widgets[name] = w
        if is_loading:
            lbl = QLabel(f'{name}<span style="color: #fab387;">*</span>:')
            lbl.setTextFormat(Qt.RichText)
            self._warn_labels[name] = lbl
        else:
            lbl = QLabel(f"{name}:")
        tip = PARAM_META.get(name, {}).get("description", "")
        if tip:
            lbl.setToolTip(tip)
        half = QHBoxLayout()
        half.setContentsMargins(0, 0, 0, 0)
        half.addWidget(lbl)
        half.addWidget(w, stretch=1)
        half_wrap = QWidget()
        half_wrap.setLayout(half)
        if is_loading:
            self._update_warn_label(name)
        return half_wrap

    def _add_numeric_pair(self, item1: tuple, item2: tuple):
        name1, val1, loading1 = item1
        name2, val2, loading2 = item2
        left = self._make_numeric_half(name1, val1, loading1)
        right = self._make_numeric_half(name2, val2, loading2)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(left, stretch=1)
        row.addWidget(right, stretch=1)
        wrap = QWidget()
        wrap.setLayout(row)
        self._detail_layout.addWidget(wrap)

    def _flush_numeric_buffer(self, buffer: list[tuple]):
        """Flush a single buffered numeric as a full-width row."""
        name, val, is_loading = buffer[0]
        half = self._make_numeric_half(name, val, is_loading)
        self._detail_layout.addWidget(half)

    # ---- Widget creation / signals ----

    def _create_widget(self, pname: str, value) -> QWidget:
        """Build a widget for *pname*/*value* without connecting signals."""
        if pname == "chat_template":
            w = NoScrollComboBox()
            w.addItem("none")
            for t in discover_templates(_TEMPLATES_DIR):
                w.addItem(t["name"])
            w.setCurrentText(str(value))
            return w
        if pname == "session_mode":
            w = NoScrollComboBox()
            for mode in ("new", "recent", "file"):
                w.addItem(mode)
            w.setCurrentText(str(value))
            return w
        if pname == "show_stats":
            w = NoScrollComboBox()
            for mode in ("always", "basic", "off"):
                w.addItem(mode)
            w.setCurrentText(str(value))
            return w
        if pname == "system_prompt":
            w = QTextEdit()
            w.setPlainText(str(value) if value else "")
            w.setFixedHeight(80)
            return w
        meta = PARAM_META.get(pname, {})
        if isinstance(value, bool):
            w = QCheckBox()
            w.setChecked(value)
        elif isinstance(value, int):
            w = NoScrollSpinBox()
            w.setRange(meta.get("min", -1), meta.get("max", 999999))
            w.setValue(value)
        elif isinstance(value, float):
            w = NoScrollDoubleSpinBox()
            w.setRange(meta.get("min", -1.0), meta.get("max", 999999.0))
            w.setDecimals(meta.get("decimals", 4))
            w.setSingleStep(meta.get("step", 0.01))
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
        self._mark_pending()

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
        self._detail_layout.addWidget(sep)

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
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(QLabel(f"{mod_name}:"))
            row.addWidget(cb)
            row.addStretch()
            w = QWidget()
            w.setLayout(row)
            self._detail_layout.addWidget(w)

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
                lbl = QLabel(f"{p.name}:")
                if p.description:
                    lbl.setToolTip(p.description)
                row = QHBoxLayout()
                row.setContentsMargins(0, 0, 0, 0)
                row.addWidget(lbl)
                row.addWidget(w, stretch=1)
                rw = QWidget()
                rw.setLayout(row)
                self._detail_layout.addWidget(rw)
            self._module_param_widgets[mod_name] = widgets

    def _on_module_access_changed(self, module_name: str, state: int):
        cfg = self._current_config
        if cfg is None:
            return
        cfg.module_access[module_name] = bool(state)
        self._rebuild_detail()
        self._mark_pending()

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
        self._mark_pending()

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
            self._add_module_header(f"[module] {mod_name} \u2014 intercepts")
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
                    cb.setToolTip(cmd.description)
                row = QHBoxLayout()
                row.setContentsMargins(0, 0, 0, 0)
                row.addWidget(QLabel(f"{label}:"))
                row.addWidget(cb)
                row.addStretch()
                rw = QWidget()
                rw.setLayout(row)
                self._detail_layout.addWidget(rw)
            self._intercept_widgets[mod_name] = widgets

    def _on_intercept_changed(self, module_name: str, command_name: str, state: int):
        cfg = self._current_config
        if cfg is None:
            return
        cfg.module_intercept.setdefault(module_name, {})[command_name] = bool(state)
        self._mark_pending()
