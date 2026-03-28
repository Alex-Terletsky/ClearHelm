import os
import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QLineEdit,
    QScrollArea, QDialog, QFileDialog, QMessageBox,
    QSizePolicy, QButtonGroup,
)
from PySide6.QtGui import QColor, QTextCursor, QFont, QTextCharFormat

from params import RunnerConfig
from runner import ServiceState
from manager import ModelManager, discover_models, discover_configs, load_config

from module_manager import ModuleManager, ModuleContext
from .constants import (
    _MODELS_DIR, _CONFIGS_DIR, _CONFIG_PATH, _MODULES_DIR,
    _AGENT_COLORS, _COMBINED_RE,
    _BASIC_COLOR, _VERBOSE_COLOR, _DEFAULT_COLOR, _parse_segment,
    DARK_STYLE,
)
from .widgets import SignalBridge, NoScrollComboBox
from .agents import _load_agent_configs, _save_agent_config, _delete_agent_config
from .sidebar import ModelSidebar
from .parameter_panel import ParameterPanel
from .module_panel import ModulePanel
from .dialogs import AddAgentDialog


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ClearHelm")
        self.resize(1100, 700)

        self._bridge = SignalBridge()
        self._bridge.text_received.connect(self._on_text_received)

        self._manager = ModelManager(
            models_dir=_MODELS_DIR,
            config_path=_CONFIG_PATH,
            output_callback=self._manager_output,
        )

        self._histories: dict[str, str] = {}
        self._param_model: str | None = None
        self._log_mode: str = "output"   # "output" | "basic" | "verbose"
        self._agent_colors:    dict[str, QColor] = {}
        self._all_events:      list[tuple[str, str]] = []   # (model_name, raw_text) in order
        self._last_individual: str = ""                      # last non-"All" selection

        # ---- Build UI ----
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        # Left: sidebar
        self._sidebar = ModelSidebar()
        self._sidebar.setMinimumWidth(200)
        self._sidebar.setMaximumWidth(320)
        splitter.addWidget(self._sidebar)

        # Right: main content
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(4)

        # Row 1: Active model selector
        active_row = QHBoxLayout()
        active_lbl = QLabel("Chat:")
        active_lbl.setFixedWidth(46)
        self._active_combo = NoScrollComboBox()
        self._active_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        active_row.addWidget(active_lbl)
        active_row.addWidget(self._active_combo)
        right_layout.addLayout(active_row)
        self._active_combo.addItem("All")

        # Row 2: Config preset bar
        config_row = QHBoxLayout()
        config_lbl = QLabel("Config:")
        config_lbl.setFixedWidth(46)
        self._preset_combo = NoScrollComboBox()
        self._preset_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._btn_apply_preset = QPushButton("Apply")
        self._btn_load_config  = QPushButton("Load...")
        self._btn_save_config  = QPushButton("Save...")
        config_row.addWidget(config_lbl)
        config_row.addWidget(self._preset_combo, stretch=1)
        config_row.addWidget(self._btn_apply_preset)
        config_row.addWidget(self._btn_load_config)
        config_row.addWidget(self._btn_save_config)
        right_layout.addLayout(config_row)

        # Console panel: output + input row + verbose toggle
        console_panel = QWidget()
        console_layout = QVBoxLayout(console_panel)
        console_layout.setContentsMargins(0, 0, 0, 0)
        console_layout.setSpacing(4)

        self._output = QTextEdit()
        self._output.setReadOnly(True)
        self._output.setFont(QFont("Consolas", 11))
        self._output.setStyleSheet(
            "QTextEdit { background-color: #11111b; color: #cdd6f4; "
            "border: 1px solid #45475a; border-radius: 4px; }"
        )
        _log_btn_style = (
            "QPushButton { background-color: transparent; color: #6c7086; "
            "border: 1px solid #45475a; border-radius: 3px; padding: 0 6px; font-size: 10px; }"
            "QPushButton:checked { color: #cdd6f4; border-color: #cdd6f4; }"
            "QPushButton:hover { color: #cdd6f4; }"
        )
        _drawer_btn_style = (
            "QPushButton { background-color: transparent; color: #6c7086; "
            "border: 1px solid #45475a; border-radius: 3px; padding: 0 10px; font-size: 12px; }"
            "QPushButton:checked { color: #89b4fa; border-color: #89b4fa; }"
            "QPushButton:hover { color: #cdd6f4; }"
        )
        self._btn_output  = QPushButton("Output")
        self._btn_basic   = QPushButton("Basic Logs")
        self._btn_verbose = QPushButton("Verbose Logs")
        self._log_btn_group = QButtonGroup(self)
        self._log_btn_group.setExclusive(True)
        for i, btn in enumerate((self._btn_output, self._btn_basic, self._btn_verbose)):
            btn.setCheckable(True)
            btn.setFixedHeight(22)
            btn.setStyleSheet(_log_btn_style)
            self._log_btn_group.addButton(btn, i)
        self._btn_output.setChecked(True)

        self._btn_drawer = QPushButton("\u2630")
        self._btn_drawer.setCheckable(True)
        self._btn_drawer.setChecked(False)
        self._btn_drawer.setFixedHeight(22)
        self._btn_drawer.setToolTip("Toggle Actions panel")
        self._btn_drawer.setStyleSheet(_drawer_btn_style)

        console_row = QHBoxLayout()
        console_row.setContentsMargins(0, 0, 0, 0)
        console_row.setSpacing(4)
        console_row.addWidget(self._btn_output)
        console_row.addWidget(self._btn_basic)
        console_row.addWidget(self._btn_verbose)
        console_row.addStretch()
        console_row.addWidget(self._btn_drawer)
        console_row.addSpacing(4)
        console_layout.addLayout(console_row)

        console_layout.addWidget(self._output, stretch=1)

        input_row = QHBoxLayout()
        self._input = QLineEdit()
        self._input.setPlaceholderText("Enter prompt...")
        self._input.setFont(QFont("Consolas", 11))
        self._btn_send = QPushButton("Send")
        self._btn_send.setStyleSheet(
            "QPushButton { background-color: #89b4fa; color: #1e1e2e; "
            "font-weight: bold; }"
            "QPushButton:hover { background-color: #b4d0fb; }"
        )
        input_row.addWidget(self._input, stretch=1)
        input_row.addWidget(self._btn_send)
        console_layout.addLayout(input_row)

        # Param scroll area (independent, below console panel)
        param_scroll = QScrollArea()
        param_scroll.setWidgetResizable(True)
        param_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        param_widget = QWidget()
        param_layout = QVBoxLayout(param_widget)
        param_layout.setContentsMargins(0, 4, 0, 4)
        param_layout.setSpacing(6)
        self._param_panel = ParameterPanel()
        param_layout.addWidget(self._param_panel)
        param_layout.addStretch()
        param_scroll.setWidget(param_widget)

        # Module interception panel (collapsible, right of console)
        self._module_panel = ModulePanel()

        self._console_split = QSplitter(Qt.Horizontal)
        self._console_split.addWidget(console_panel)
        self._console_split.addWidget(self._module_panel)
        self._console_split.setStretchFactor(0, 1)   # console gets all space
        self._console_split.setStretchFactor(1, 0)   # module panel natural width
        self._console_split.setCollapsible(1, False)  # only collapse via drawer button

        # Splitter: console area (output+input+verbose+module panel) vs. param panel
        v_split = QSplitter(Qt.Vertical)
        v_split.addWidget(self._console_split)
        v_split.addWidget(param_scroll)
        v_split.setStretchFactor(0, 3)
        v_split.setStretchFactor(1, 1)

        right_layout.addWidget(v_split, stretch=1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # ---- Signals ----
        self._sidebar.load_requested.connect(self._load_model)
        self._sidebar.unload_requested.connect(self._unload_model)
        self._sidebar.add_requested.connect(self._add_agent)
        self._sidebar.delete_requested.connect(self._delete_agent)
        self._sidebar.selection_changed.connect(self._on_sidebar_selection_changed)
        self._sidebar.save_requested.connect(self._save_agent)
        self._active_combo.currentTextChanged.connect(self._on_active_changed)
        self._btn_send.clicked.connect(self._send_prompt)
        self._input.returnPressed.connect(self._send_prompt)
        self._btn_apply_preset.clicked.connect(self._apply_preset_from_combo)
        self._btn_load_config.clicked.connect(self._load_config_file)
        self._btn_save_config.clicked.connect(self._save_config)
        self._log_btn_group.idClicked.connect(self._on_log_mode_changed)
        self._btn_drawer.clicked.connect(self._on_drawer_toggled)
        self._module_panel.badge_changed.connect(self._on_badge_changed)
        self._module_panel.expanded_changed.connect(self._btn_drawer.setChecked)

        # ---- Status poll timer ----
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_status)
        self._timer.start(500)

        # ---- Load saved agents and presets ----
        self._load_saved_agents()
        self._populate_presets()

        # ---- Built-in virtual agents ----
        self._histories["Echo"] = ""
        self._chat_combo_add("Echo")

        # ---- Module system ----
        _ctx = ModuleContext(
            submit_prompt_fn=self._manager.submit_prompt,
            get_agent_names_fn=lambda: self._manager.model_names,
            ready_models_fn=self._manager.ready_models,
            emit_fn=lambda text: self._manager_output("system", text),
            get_module_config_fn=self._get_module_config,
            set_module_config_fn=self._set_module_config,
            request_action_fn=self._on_action_requested,
        )
        self._module_manager = ModuleManager(
            _MODULES_DIR, _ctx,
            get_agent_config_fn=self._get_agent_config,
        )
        self._module_manager.load_all()
        self._param_panel.set_module_schemas(self._module_manager.module_schemas)
        self._param_panel.set_intercept_schemas(self._module_manager.intercept_schemas)

        # Initialise the "All" view state now that signals are connected and agents loaded
        self._on_active_changed(self._active_combo.currentText())

    # ---- Module config helpers ----

    def _get_agent_config(self, agent_name: str):
        """Return the RunnerConfig for *agent_name*, or None."""
        try:
            return self._manager.get_config(agent_name)
        except KeyError:
            return None

    def _active_agent_name(self) -> str | None:
        """Return the name of the agent currently targeted for interaction."""
        current = self._active_combo.currentText()
        return self._last_individual if current == "All" else (current or None)

    def _get_module_config(self, module_name: str) -> dict:
        """Read the active agent's config for *module_name*, with defaults merged."""
        agent = self._active_agent_name()
        if not agent:
            return {}
        cfg = self._get_agent_config(agent)
        if cfg is None:
            return {}
        merged = dict(cfg.module_config.get(module_name, {}))
        schemas = self._module_manager.module_schemas
        for p in schemas.get(module_name, []):
            merged.setdefault(p.name, p.default)
        return merged

    def _set_module_config(self, module_name: str, key: str, value):
        """Write a single key into the active agent's module config."""
        agent = self._active_agent_name()
        if not agent:
            return
        cfg = self._get_agent_config(agent)
        if cfg is None:
            return
        cfg.module_config.setdefault(module_name, {})[key] = value

    def _on_action_requested(self, module_name, command_name, agent_name,
                             description, action, on_deny):
        """Route an intercepted module action to the module panel."""
        self._module_panel.add_request(
            module_name, command_name, agent_name,
            description, action, on_deny)

    # ---- Chat combo helpers ----

    def _assign_agent_color(self, name: str) -> QColor:
        used = list(self._agent_colors.values())  # list avoids QColor __hash__ requirement
        for c in _AGENT_COLORS:
            if c not in used:
                return c
        return _AGENT_COLORS[len(self._agent_colors) % len(_AGENT_COLORS)]

    def _chat_combo_add(self, name: str):
        self._agent_colors[name] = self._assign_agent_color(name)
        self._active_combo.addItem(name)

    def _chat_combo_remove(self, name: str):
        idx = self._active_combo.findText(name)
        if idx >= 0:
            self._active_combo.removeItem(idx)
        self._agent_colors.pop(name, None)

    # ---- Agent loading / registration ----

    def _load_saved_agents(self):
        for cfg in _load_agent_configs():
            name = cfg.model_name
            path = cfg.model_config.model_path
            try:
                self._manager.add_model(name, path, config=cfg)
                self._histories[name] = ""
                self._sidebar.add_agent(name)
                self._chat_combo_add(name)
            except Exception:
                pass

    def _add_agent(self):
        available = discover_models(_MODELS_DIR)
        presets = [(c["name"], c["path"]) for c in discover_configs(_CONFIGS_DIR)]

        dlg = AddAgentDialog(available, presets, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return

        name = dlg.agent_name()
        model = dlg.selected_model()
        preset_path = dlg.selected_preset()
        if not name or model is None:
            return

        if name in self._manager.model_names:
            QMessageBox.warning(self, "Duplicate Name",
                                f'An agent named "{name}" already exists.')
            return

        if preset_path:
            config = RunnerConfig.from_file(preset_path, model_path=model["path"])
        else:
            config = load_config(self._manager.config_path, model_path=model["path"])
        config.model_name = name

        _save_agent_config(config,
                          module_schemas=self._module_manager.module_schemas,
                          intercept_schemas=self._module_manager.intercept_schemas)
        self._manager.add_model(name, model["path"], config=config)
        self._histories[name] = ""
        self._sidebar.add_agent(name)
        self._chat_combo_add(name)

    def _delete_agent(self, name: str):
        reply = QMessageBox.warning(
            self, "Delete Agent",
            f'Permanently delete agent "{name}"?\n\nThis will remove its saved config and cannot be undone.',
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if reply != QMessageBox.Yes:
            return
        _delete_agent_config(name)
        self._module_panel.remove_requests_for_agent(name)
        self._manager.remove_model(name)
        self._sidebar.remove_agent(name)
        self._histories.pop(name, None)
        self._chat_combo_remove(name)
        self._all_events = [(n, t) for n, t in self._all_events if n != name]
        if self._param_model == name:
            self._param_panel.set_config(None)
            self._param_panel.set_state(None)
            self._param_model = None

    def _save_agent(self, name: str):
        try:
            cfg = self._manager.get_config(name)
            _save_agent_config(cfg,
                              module_schemas=self._module_manager.module_schemas,
                              intercept_schemas=self._module_manager.intercept_schemas)
        except KeyError:
            pass

    # ---- Preset / config ----

    def _populate_presets(self):
        """Scan configs dir and fill the preset combo."""
        self._preset_combo.clear()
        for c in discover_configs(_CONFIGS_DIR):
            self._preset_combo.addItem(c["name"], userData=c["path"])
        default_idx = self._preset_combo.findText("default", Qt.MatchFixedString | Qt.MatchCaseSensitive)
        if default_idx >= 0:
            self._preset_combo.setCurrentIndex(default_idx)

    def _current_target_model(self) -> str | None:
        """Return the model to act on: active READY model, or sidebar selection."""
        name = self._active_combo.currentText()
        return name if name else self._sidebar.selected_model()

    def _apply_preset(self, path: str):
        """Load a config JSON and apply it to the currently targeted model."""
        name = self._current_target_model()
        if not name:
            QMessageBox.information(self, "No Model Selected",
                                    "Select a model in the sidebar first.")
            return
        try:
            cfg = self._manager.get_config(name)
            saved_path = cfg.model_config.model_path
            saved_name = cfg.model_name
            new_cfg = RunnerConfig.from_file(path, model_path=saved_path)
            # Update fields on the existing RunnerConfig instance so RunnerService's
            # reference to the config object stays valid.
            cfg.active_groups      = new_cfg.active_groups
            cfg.model_config       = new_cfg.model_config
            cfg.generation_config  = new_cfg.generation_config
            cfg.module_config      = new_cfg.module_config
            cfg.module_access      = new_cfg.module_access
            cfg.module_intercept   = new_cfg.module_intercept
            cfg.model_name         = saved_name
            self._param_panel.set_config(cfg)
        except Exception as e:
            QMessageBox.warning(self, "Preset Error", str(e))

    def _apply_preset_from_combo(self):
        path = self._preset_combo.currentData()
        if path:
            self._apply_preset(path)

    def _load_config_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Config File", _CONFIGS_DIR, "JSON Files (*.json)"
        )
        if path:
            self._apply_preset(path)

    def _save_config(self):
        name = self._current_target_model()
        if not name:
            QMessageBox.information(self, "No Model Selected",
                                    "Select a model in the sidebar first.")
            return
        try:
            cfg = self._manager.get_config(name)
            default_path = os.path.join(_CONFIGS_DIR, f"{name}.json")
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Config", default_path, "JSON Files (*.json)"
            )
            if path:
                cfg.to_file(path,
                            module_schemas=self._module_manager.module_schemas,
                            intercept_schemas=self._module_manager.intercept_schemas)
                self._populate_presets()  # refresh list if saved to configs dir
        except Exception as e:
            QMessageBox.warning(self, "Save Error", str(e))

    # ---- Load / unload ----

    def _load_model(self, name: str):
        state = self._manager.get_state(name)
        if state not in (ServiceState.IDLE, ServiceState.ERROR):
            return
        self._manager.start_model(name)

    def _unload_model(self, name: str):
        state = self._manager.get_state(name)
        if state in (ServiceState.IDLE, ServiceState.STOPPING):
            return
        self._module_panel.remove_requests_for_agent(name)
        self._manager.stop_model(name)

    # ---- Active model switching ----

    def _on_sidebar_selection_changed(self, name: str):
        if not name:
            return
        try:
            cfg = self._manager.get_config(name)
            state = self._manager.get_state(name)
            self._param_model = name
            self._param_panel.set_config(cfg)
            self._param_panel.set_state(state)
        except KeyError:
            pass

    def _on_active_changed(self, name: str):
        if not name:
            return
        if name == "All":
            self._manager.set_active(None)
            placeholder = (f"Enter prompt to {self._last_individual}..."
                           if self._last_individual else "No agent selected")
            self._input.setPlaceholderText(placeholder)
            QTimer.singleShot(0, self._reload_all)
            self._param_panel.set_config(None)
            self._param_panel.set_state(None)
            self._param_model = None
            self._module_manager.set_target_agent(self._last_individual or None)
            self._module_panel.set_agent_filter(None)
            return
        # Individual agent
        self._last_individual = name
        self._input.setPlaceholderText("Enter prompt...")
        self._module_manager.set_target_agent(name)
        self._module_panel.set_agent_filter(name)
        try:
            self._manager.set_active(name)
        except KeyError:
            pass
        QTimer.singleShot(0, lambda: self._reload_output(name))
        try:
            cfg   = self._manager.get_config(name)
            state = self._manager.get_state(name)
            self._param_model = name
            self._param_panel.set_config(cfg)
            self._param_panel.set_state(state)
        except KeyError:
            self._param_panel.set_config(None)
            self._param_panel.set_state(None)
            self._param_model = None

    # ---- Output routing ----

    def _insert_formatted(self, cursor: QTextCursor, text: str, kind: str,
                          agent_color=None) -> None:
        fmt = QTextCharFormat()
        if kind == 'verbose':
            fmt.setForeground(_VERBOSE_COLOR)
        elif kind == 'basic':
            fmt.setForeground(_BASIC_COLOR)
        elif agent_color is not None:
            fmt.setForeground(agent_color)
        else:
            fmt.setForeground(_DEFAULT_COLOR)
        cursor.insertText(text, fmt)

    def _should_skip(self, kind: str) -> bool:
        if kind == 'basic' and self._log_mode == 'output':
            return True
        if kind == 'verbose' and self._log_mode != 'verbose':
            return True
        return False

    def _render_segments(self, cursor: QTextCursor, text: str,
                         agent_color=None) -> bool:
        """Parse *text* into segments and insert visible ones at *cursor*.

        Returns True if any content was rendered, False if all segments
        were filtered out by the current log mode.
        """
        parts = []
        for part in _COMBINED_RE.split(text):
            if not part:
                continue
            kind, content = _parse_segment(part)
            parts.append((kind, content, self._should_skip(kind)))

        prev_skipped = False
        rendered = False
        for i, (kind, content, skip) in enumerate(parts):
            if skip:
                prev_skipped = True
                continue
            if prev_skipped:
                content = content.lstrip('\n')
                prev_skipped = False
            if (not content.strip('\n')
                    and i + 1 < len(parts) and parts[i + 1][2]):
                prev_skipped = True
                continue
            if not content:
                continue
            self._insert_formatted(cursor, content, kind, agent_color=agent_color)
            rendered = True
        return rendered

    def _reload_output(self, model_name: str) -> None:
        self._output.clear()
        history = self._histories.get(model_name, "")
        if not history:
            return
        cursor = self._output.textCursor()
        self._render_segments(cursor, history)
        self._output.setTextCursor(cursor)
        self._output.ensureCursorVisible()

    def _reload_all(self) -> None:
        self._output.clear()
        if not self._all_events:
            return
        cursor = self._output.textCursor()
        for name, text in self._all_events:
            color = self._agent_colors.get(name, _DEFAULT_COLOR)
            self._render_segments(cursor, text, agent_color=color)
        self._output.setTextCursor(cursor)
        self._output.ensureCursorVisible()

    def _on_log_mode_changed(self, btn_id: int) -> None:
        self._log_mode = ("output", "basic", "verbose")[btn_id]
        active = self._active_combo.currentText()
        if active == "All":
            self._reload_all()
        elif active:
            self._reload_output(active)

    def _on_drawer_toggled(self):
        expanded = self._btn_drawer.isChecked()
        if not expanded:
            self._drawer_saved_sizes = self._console_split.sizes()
        self._module_panel.set_expanded(expanded)
        if expanded and hasattr(self, '_drawer_saved_sizes'):
            self._console_split.setSizes(self._drawer_saved_sizes)
        elif not expanded:
            self._console_split.setSizes([sum(self._drawer_saved_sizes), 0])

    def _on_badge_changed(self, count: int):
        if count > 0:
            self._btn_drawer.setText(f"\u2630 ({count})")
            self._btn_drawer.setToolTip(f"Toggle Actions panel ({count} pending)")
        else:
            self._btn_drawer.setText("\u2630")
            self._btn_drawer.setToolTip("Toggle Actions panel")

    def _manager_output(self, model_name: str, text: str):
        self._bridge.text_received.emit(model_name, text)

    def _on_text_received(self, model_name: str, text: str):
        if model_name not in self._histories:
            self._histories[model_name] = ""
        self._histories[model_name] += text
        self._all_events.append((model_name, text))
        self._module_manager.on_output(model_name, text)

        current = self._active_combo.currentText()
        if current not in ("All", model_name):
            return

        cursor = self._output.textCursor()
        cursor.movePosition(QTextCursor.End)
        color = self._agent_colors.get(model_name, _DEFAULT_COLOR) if current == "All" else None
        if self._render_segments(cursor, text, agent_color=color):
            self._output.setTextCursor(cursor)
            self._output.ensureCursorVisible()

    # ---- Prompt submission ----

    def _send_prompt(self):
        prompt = self._input.text().strip()
        if not prompt:
            return
        current = self._active_combo.currentText()
        active = self._last_individual if current == "All" else current
        if not active:
            return

        if self._module_manager.on_user_input(prompt):
            self._input.clear()
            return

        groups = self._param_panel.active_groups()
        try:
            self._manager.update_active_groups(active, groups)
        except KeyError:
            pass

        self._input.clear()
        echo = f"\n> {prompt}\n"
        self._histories.setdefault(active, "")
        self._histories[active] += echo
        self._all_events.append((active, echo))

        cursor = self._output.textCursor()
        cursor.movePosition(QTextCursor.End)
        if current == "All":
            color = self._agent_colors.get(active, _DEFAULT_COLOR)
            self._insert_formatted(cursor, echo, 'plain', agent_color=color)
        else:
            self._insert_formatted(cursor, echo, 'plain')
        self._output.setTextCursor(cursor)
        self._output.ensureCursorVisible()

        if active == "Echo":
            self._manager_output("Echo", prompt + "\n")
        else:
            self._manager.submit_prompt(prompt, model_name=active)

    # ---- Status polling ----

    def _poll_status(self):
        statuses = self._manager.get_all_status()
        for name, state in statuses.items():
            self._sidebar.update_status(name, state)

        if self._param_model:
            try:
                self._param_panel.set_state(self._manager.get_state(self._param_model))
            except KeyError:
                pass

    # ---- Cleanup ----

    def closeEvent(self, event):
        self._timer.stop()
        self._module_panel.clear_requests()
        self._manager.shutdown()
        self._module_manager.shutdown()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
