import os
import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLineEdit,
    QDialog, QFileDialog, QMessageBox,
    QButtonGroup,
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
    DARK_STYLE, ECHO_NAME, MULTI_SESSION_RENDER_CAP,
)
from .widgets import SignalBridge
from .agents import (
    _load_agent_configs, _save_agent_config, _delete_agent_config,
    _resolve_session, _save_session, _load_session_file,
)
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
        self._bridge.generation_complete.connect(self._on_generation_complete)

        self._manager = ModelManager(
            models_dir=_MODELS_DIR,
            config_path=_CONFIG_PATH,
            output_callback=self._manager_output,
            completion_callback=self._manager_completion,
        )

        self._histories: dict[str, str] = {}
        self._chat_histories: dict[str, list[dict]] = {}
        self._session_paths: dict[str, str] = {}
        self._param_model: str | None = None
        self._log_mode: str = "output"   # "output" | "basic" | "verbose"
        self._agent_colors:    dict[str, QColor] = {}
        self._all_events:      list[tuple[str, str | None]] = []  # (name, text) or (name, None) = session marker
        self._current_chat: str = ""
        self._multi_view_active: bool = False
        self._multi_checked: list[str] = []
        self._session_cursors: list[QTextCursor] = []
        self._model_session_idx: dict[str, int] = {}
        self._session_model_names: list[str] = []

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

        # Console panel: output + input row + log mode toggle
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
        self._input.setPlaceholderText("Select an agent to begin...")
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

        # Parameter panel (now manages its own scroll area)
        self._param_panel = ParameterPanel()

        # Module interception panel (collapsible, right of console)
        self._module_panel = ModulePanel()

        self._console_split = QSplitter(Qt.Horizontal)
        self._console_split.addWidget(console_panel)
        self._console_split.addWidget(self._module_panel)
        self._console_split.setStretchFactor(0, 1)   # console gets all space
        self._console_split.setStretchFactor(1, 0)   # module panel natural width
        self._console_split.setCollapsible(1, False)  # only collapse via drawer button

        # Splitter: console area vs. param panel
        v_split = QSplitter(Qt.Vertical)
        v_split.addWidget(self._console_split)
        v_split.addWidget(self._param_panel)
        v_split.setStretchFactor(0, 3)
        v_split.setStretchFactor(1, 1)

        right_layout.addWidget(v_split, stretch=1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self._module_manager = None

        # ---- Signals ----
        self._sidebar.load_requested.connect(self._load_model)
        self._sidebar.unload_requested.connect(self._unload_model)
        self._sidebar.add_requested.connect(self._add_agent)
        self._sidebar.delete_requested.connect(self._delete_agent)
        self._sidebar.selection_changed.connect(self._on_sidebar_selection_changed)
        self._sidebar.multi_view_changed.connect(self._on_multi_view_changed)
        self._sidebar.multi_selection_changed.connect(self._on_multi_selection_changed)
        self._btn_send.clicked.connect(self._send_prompt)
        self._input.returnPressed.connect(self._send_prompt)
        self._log_btn_group.idClicked.connect(self._on_log_mode_changed)
        self._btn_drawer.clicked.connect(self._on_drawer_toggled)
        self._module_panel.badge_changed.connect(self._on_badge_changed)
        self._module_panel.expanded_changed.connect(self._btn_drawer.setChecked)
        self._param_panel.save_config_requested.connect(self._on_param_save_config)
        self._param_panel.save_agent_requested.connect(self._on_param_save_agent)
        self._param_panel.apply_requested.connect(self._on_param_apply)

        # ---- Status poll timer ----
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_status)
        self._timer.start(500)

        # ---- Load saved agents ----
        self._load_saved_agents()

        # ---- Echo virtual agent ----
        self._sidebar.add_echo()
        self._histories[ECHO_NAME] = ""
        self._agent_colors[ECHO_NAME] = self._assign_agent_color(ECHO_NAME)
        self._sidebar.set_agent_color(ECHO_NAME, self._agent_colors[ECHO_NAME].name())

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

        # Auto-select first agent if any exist
        if self._sidebar.list.count() > 0:
            self._sidebar.list.setCurrentRow(0)

    # ---- Module config helpers ----

    def _get_agent_config(self, agent_name: str):
        """Return the RunnerConfig for *agent_name*, or None."""
        try:
            return self._manager.get_config(agent_name)
        except KeyError:
            return None

    def _active_agent_name(self) -> str | None:
        """Return the name of the agent currently targeted for interaction."""
        return self._current_chat or None

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

    # ---- Agent color assignment ----

    def _assign_agent_color(self, name: str) -> QColor:
        used = list(self._agent_colors.values())
        for c in _AGENT_COLORS:
            if c not in used:
                return c
        return _AGENT_COLORS[len(self._agent_colors) % len(_AGENT_COLORS)]

    # ---- Agent loading / registration ----

    def _load_saved_agents(self):
        for cfg in _load_agent_configs():
            name = cfg.model_name
            path = cfg.model_config.model_path
            try:
                self._manager.add_model(name, path, config=cfg)
                self._histories[name] = ""
                # Resolve session for this agent
                gc = cfg.generation_config
                history, resolved_path = _resolve_session(
                    name, gc.session_mode, gc.session_file,
                    log_fn=lambda text: self._manager_output("system", text),
                )
                if gc.use_history and history:
                    self._chat_histories[name] = history
                self._session_paths[name] = resolved_path
                gc.session_file = resolved_path
                self._sidebar.add_agent(name)
                self._agent_colors[name] = self._assign_agent_color(name)
                self._sidebar.set_agent_color(name, self._agent_colors[name].name())
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
        # Resolve session for the new agent
        gc = config.generation_config
        history, resolved_path = _resolve_session(
            name, gc.session_mode, gc.session_file,
            log_fn=lambda text: self._manager_output("system", text),
        )
        if gc.use_history and history:
            self._chat_histories[name] = history
        self._session_paths[name] = resolved_path
        gc.session_file = resolved_path
        self._sidebar.add_agent(name)
        self._agent_colors[name] = self._assign_agent_color(name)
        self._sidebar.set_agent_color(name, self._agent_colors[name].name())

    def _delete_agent(self, name: str):
        if name == ECHO_NAME:
            return
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
        self._chat_histories.pop(name, None)
        self._session_paths.pop(name, None)
        self._agent_colors.pop(name, None)
        self._all_events = [(n, t) for n, t in self._all_events if n != name]
        if self._multi_view_active:
            self._reload_multi()
        if self._param_model == name:
            self._param_panel.set_config(None)
            self._param_panel.set_state(None)
            self._param_panel.set_agent_name("")
            self._param_model = None
        if self._current_chat == name:
            self._current_chat = ""
            self._output.clear()

    def _save_agent(self, name: str):
        try:
            cfg = self._manager.get_config(name)
            _save_agent_config(cfg,
                              module_schemas=self._module_manager.module_schemas,
                              intercept_schemas=self._module_manager.intercept_schemas)
        except KeyError:
            pass

    # ---- Parameter panel callbacks ----

    def _on_param_save_config(self):
        """Save the staged config to a user-chosen file."""
        name = self._param_model
        if not name:
            return
        staged = self._param_panel.staged_config()
        if staged is None:
            return
        default_path = os.path.join(_CONFIGS_DIR, f"{name}.json")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Config", default_path, "JSON Files (*.json)")
        if path:
            try:
                staged.to_file(path,
                               module_schemas=self._module_manager.module_schemas,
                               intercept_schemas=self._module_manager.intercept_schemas)
            except Exception as e:
                QMessageBox.warning(self, "Save Error", str(e))

    def _on_param_save_agent(self):
        """Save the current live config as the agent's persistent file."""
        name = self._param_model
        if name:
            self._save_agent(name)

    def _on_param_apply(self):
        """Post-apply hook: sync active groups and session path to the RunnerService."""
        name = self._param_model
        if not name:
            return
        try:
            groups = self._param_panel.active_groups()
            self._manager.update_active_groups(name, groups)
        except KeyError:
            pass
        # Redirect session if session_file changed — load new history + redirect saves
        try:
            cfg = self._manager.get_config(name)
            new_path = cfg.generation_config.session_file
            if new_path and new_path != self._session_paths.get(name):
                self._session_paths[name] = new_path
                history = _load_session_file(new_path)
                if history:
                    self._chat_histories[name] = history
                else:
                    self._chat_histories.pop(name, None)
        except KeyError:
            pass

    # ---- Load / unload ----

    def _load_model(self, name: str):
        if name == ECHO_NAME:
            return
        state = self._manager.get_state(name)
        if state not in (ServiceState.IDLE, ServiceState.ERROR):
            return
        self._manager.start_model(name)

    def _unload_model(self, name: str):
        if name == ECHO_NAME:
            return
        state = self._manager.get_state(name)
        if state in (ServiceState.IDLE, ServiceState.STOPPING):
            return
        self._module_panel.remove_requests_for_agent(name)
        self._manager.stop_model(name)

    # ---- Sidebar selection -> chat + param switching ----

    def _on_sidebar_selection_changed(self, name: str):
        if not name:
            return

        # Switch chat view (unless multi view is active)
        self._current_chat = name
        self._input.setPlaceholderText(f"Enter prompt to {name}...")
        if not self._multi_view_active:
            QTimer.singleShot(0, lambda n=name: self._reload_output(n))

        # Update parameter panel
        self._param_panel.set_agent_name(name)
        if name == ECHO_NAME:
            self._param_model = None
            self._param_panel.set_config(None)
            self._param_panel.set_state(None)
            self._sidebar.update_load_button(None)
        else:
            try:
                cfg = self._manager.get_config(name)
                state = self._manager.get_state(name)
                self._param_model = name
                self._param_panel.set_config(cfg)
                self._param_panel.set_state(state)
                self._sidebar.update_load_button(state)
            except KeyError:
                self._param_panel.set_config(None)
                self._param_panel.set_state(None)
                self._param_model = None
                self._sidebar.update_load_button(None)

        # Module panel filter
        self._module_manager.set_target_agent(name)
        self._module_panel.set_agent_filter(name)

        # Set active model in manager
        if name != ECHO_NAME:
            try:
                self._manager.set_active(name)
            except KeyError:
                pass

    # ---- Multi View ----

    def _is_scrolled_to_bottom(self) -> bool:
        sb = self._output.verticalScrollBar()
        return sb.value() >= sb.maximum() - 10

    def _register_session_cursor(self, doc, position: int, model_name: str) -> None:
        """Create a session cursor at *position* and register it for *model_name*."""
        cursor = QTextCursor(doc)
        cursor.setPosition(position)
        self._session_cursors.append(cursor)
        self._session_model_names.append(model_name)
        self._model_session_idx[model_name] = len(self._session_cursors) - 1

    def _append_session_header(self, model_name: str) -> None:
        """Append a bold [ModelName] header at the document end and register a session cursor."""
        doc = self._output.document()
        cursor = QTextCursor(doc)
        cursor.movePosition(QTextCursor.End)
        if doc.characterCount() > 1:
            cursor.insertText("\n")
        fmt = QTextCharFormat()
        color = self._agent_colors.get(model_name, _DEFAULT_COLOR)
        fmt.setForeground(color)
        fmt.setFontWeight(QFont.Bold)
        cursor.insertText(f"[{model_name}]\n", fmt)
        self._register_session_cursor(doc, cursor.position(), model_name)

    def _on_multi_view_changed(self, active: bool):
        self._multi_view_active = active
        if active:
            self._reload_multi()
        else:
            self._session_cursors.clear()
            self._model_session_idx.clear()
            self._session_model_names.clear()
            if self._current_chat:
                QTimer.singleShot(0, lambda: self._reload_output(self._current_chat))

    def _on_multi_selection_changed(self, names: list):
        self._multi_checked = names
        if self._multi_view_active:
            self._reload_multi()

    def _reload_multi(self) -> None:
        """Rebuild console with session-grouped output from checked agents."""
        was_at_bottom = self._is_scrolled_to_bottom()
        saved_scroll = self._output.verticalScrollBar().value()

        self._output.clear()
        self._session_cursors.clear()
        self._model_session_idx.clear()
        self._session_model_names.clear()

        if not self._all_events or not self._multi_checked:
            return

        # Parse _all_events into ordered sessions
        sessions: list[tuple[str, list[str]]] = []
        last_session_for: dict[str, int] = {}

        for name, text in self._all_events:
            if name not in self._multi_checked:
                continue
            if text is None or name not in last_session_for:
                last_session_for[name] = len(sessions)
                sessions.append((name, []))
            if text is not None:
                sessions[last_session_for[name]][1].append(text)

        # Render sessions with cap
        doc = self._output.document()
        cursor = QTextCursor(doc)

        for session_model, chunks in sessions:
            if not chunks:
                continue

            # Header
            cursor.movePosition(QTextCursor.End)
            if doc.characterCount() > 1:
                cursor.insertText("\n")
            hdr_fmt = QTextCharFormat()
            color = self._agent_colors.get(session_model, _DEFAULT_COLOR)
            hdr_fmt.setForeground(color)
            hdr_fmt.setFontWeight(QFont.Bold)
            cursor.insertText(f"[{session_model}]\n", hdr_fmt)

            # Apply render cap
            if len(chunks) > MULTI_SESSION_RENDER_CAP:
                trunc_fmt = QTextCharFormat()
                trunc_fmt.setForeground(QColor("#6c7086"))
                cursor.insertText("[... truncated]\n", trunc_fmt)
                chunks = chunks[-MULTI_SESSION_RENDER_CAP:]

            for text in chunks:
                self._render_segments(cursor, text, agent_color=color)

            # Store session cursor at end of this block
            cursor.movePosition(QTextCursor.End)
            self._register_session_cursor(doc, cursor.position(), session_model)

        # Scroll preservation
        sb = self._output.verticalScrollBar()
        if was_at_bottom:
            sb.setValue(sb.maximum())
        else:
            sb.setValue(saved_scroll)

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

    def _on_log_mode_changed(self, btn_id: int) -> None:
        self._log_mode = ("output", "basic", "verbose")[btn_id]
        if self._multi_view_active:
            self._reload_multi()
        elif self._current_chat:
            self._reload_output(self._current_chat)

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

    def _manager_completion(self, model_name: str, response: str):
        self._bridge.generation_complete.emit(model_name, response)

    def _on_generation_complete(self, model_name: str, response: str):
        """Store the clean assistant response in chat history and persist."""
        try:
            cfg = self._manager.get_config(model_name)
            if not cfg.generation_config.use_history:
                return
        except KeyError:
            return
        hist = self._chat_histories.get(model_name)
        if hist is None:
            return
        hist.append({"role": "assistant", "content": response.strip()})
        session_path = self._session_paths.get(model_name)
        if session_path:
            _save_session(model_name, session_path, hist)

    def _on_text_received(self, model_name: str, text: str):
        if model_name not in self._histories:
            self._histories[model_name] = ""
        self._histories[model_name] += text
        self._all_events.append((model_name, text))
        if self._module_manager is not None:
            self._module_manager.on_output(model_name, text)

        if self._multi_view_active:
            if model_name not in self._multi_checked:
                return
            color = self._agent_colors.get(model_name, _DEFAULT_COLOR)
            idx = self._model_session_idx.get(model_name)
            if idx is None:
                at_bottom = self._is_scrolled_to_bottom()
                self._append_session_header(model_name)
                idx = self._model_session_idx[model_name]
            else:
                at_bottom = self._is_scrolled_to_bottom()
            self._render_segments(self._session_cursors[idx], text, agent_color=color)
            if at_bottom:
                self._output.verticalScrollBar().setValue(
                    self._output.verticalScrollBar().maximum())
            return

        if model_name != self._current_chat:
            return

        cursor = self._output.textCursor()
        cursor.movePosition(QTextCursor.End)
        if self._render_segments(cursor, text):
            self._output.setTextCursor(cursor)
            self._output.ensureCursorVisible()

    # ---- Prompt submission ----

    def _send_prompt(self):
        prompt = self._input.text().strip()
        if not prompt:
            return
        active = self._current_chat
        if not active:
            return

        if self._module_manager.on_user_input(prompt):
            self._input.clear()
            return

        if active != ECHO_NAME:
            groups = self._param_panel.active_groups()
            try:
                self._manager.update_active_groups(active, groups)
            except KeyError:
                pass

        self._input.clear()
        echo = f"\n> {prompt}\n"
        self._histories.setdefault(active, "")
        self._histories[active] += echo
        self._all_events.append((active, None))    # session marker
        self._all_events.append((active, echo))

        if self._multi_view_active:
            if active in self._multi_checked:
                at_bottom = self._is_scrolled_to_bottom()
                self._append_session_header(active)
                session_cursor = self._session_cursors[self._model_session_idx[active]]
                color = self._agent_colors.get(active, _DEFAULT_COLOR)
                self._insert_formatted(session_cursor, echo, 'plain', agent_color=color)
                if at_bottom:
                    self._output.verticalScrollBar().setValue(
                        self._output.verticalScrollBar().maximum())
        else:
            cursor = self._output.textCursor()
            cursor.movePosition(QTextCursor.End)
            self._insert_formatted(cursor, echo, 'plain')
            self._output.setTextCursor(cursor)
            self._output.ensureCursorVisible()

        if active == ECHO_NAME:
            self._manager_output(ECHO_NAME, prompt + "\n")
        else:
            # Only accumulate history when use_history is enabled
            history_arg = None
            try:
                cfg = self._manager.get_config(active)
                use_history = cfg.generation_config.use_history
            except KeyError:
                use_history = False
            if use_history:
                self._chat_histories.setdefault(active, [])
                self._chat_histories[active].append({"role": "user", "content": prompt})
                history_arg = self._chat_histories[active][:-1]  # prior turns only
            self._manager.submit_prompt(prompt, model_name=active, history=history_arg)

    # ---- Status polling ----

    def _poll_status(self):
        statuses = self._manager.get_all_status()
        for name, state in statuses.items():
            self._sidebar.update_status(name, state)

        # Keep Load/Unload button in sync with selected agent
        selected = self._sidebar.selected_model()
        if selected and selected != ECHO_NAME:
            try:
                self._sidebar.update_load_button(self._manager.get_state(selected))
            except KeyError:
                pass

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
