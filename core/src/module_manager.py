"""Lightweight module system for ClearHelm.

Drop a .py file in the project-level ``modules/`` directory.  The file must
expose a ``MODULE_CLASS`` attribute pointing to a ``Module`` subclass.  The
class is instantiated, given a ``ModuleContext``, and receives lifecycle
callbacks as the UI runs.
"""

import importlib.util
import logging
import os

logger = logging.getLogger(__name__)


class ModuleContext:
    """Passed to every module at load time; provides safe access to the app."""

    def __init__(self, submit_prompt_fn, get_agent_names_fn, ready_models_fn, emit_fn):
        self._submit_prompt = submit_prompt_fn
        self._get_agent_names = get_agent_names_fn
        self._ready_models = ready_models_fn
        self._emit = emit_fn

    def message_agent(self, agent_name: str, message: str):
        """Send *message* to *agent_name* via the manager's prompt queue."""
        self._submit_prompt(message, model_name=agent_name)

    def get_agent_names(self) -> list:
        """Return list of all registered agent names."""
        return list(self._get_agent_names())

    def get_ready_agents(self) -> list:
        """Return list of agent names currently in READY state."""
        return list(self._ready_models())

    def emit(self, text: str):
        """Write *text* to the console under the 'system' sender."""
        self._emit(text)


class Module:
    """Base class for ClearHelm modules.  All hooks are no-ops by default."""

    def on_load(self, ctx: ModuleContext):
        """Called once when the module is loaded.  Store *ctx* if needed."""

    def on_output(self, model_name: str, text: str):
        """Called on the Qt main thread for every text chunk emitted by an agent."""

    def on_user_input(self, text: str) -> bool:
        """Called before a user prompt is dispatched.

        Return True to consume the input (clears the field, skips normal send).
        Return False to let normal processing continue.
        """
        return False

    def on_unload(self):
        """Called when the application is closing."""


class ModuleManager:
    """Discovers, loads, and brokers lifecycle events for all active modules."""

    def __init__(self, modules_dir: str, ctx: ModuleContext):
        self._modules_dir = modules_dir
        self._ctx = ctx
        self._modules: list[Module] = []

    # ---- Discovery & loading ----

    def load_all(self):
        """Scan *modules_dir* for ``*.py`` files and load each one."""
        if not os.path.isdir(self._modules_dir):
            return
        for fname in sorted(os.listdir(self._modules_dir)):
            if not fname.endswith(".py") or fname.startswith("_"):
                continue
            path = os.path.join(self._modules_dir, fname)
            self._load_file(path)

    def _load_file(self, path: str):
        module_name = os.path.splitext(os.path.basename(path))[0]
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            py_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(py_mod)
            cls = getattr(py_mod, "MODULE_CLASS", None)
            if cls is None:
                logger.warning("Module %s has no MODULE_CLASS attribute — skipped", path)
                return
            instance: Module = cls()
            instance.on_load(self._ctx)
            self._modules.append(instance)
        except Exception:
            logger.exception("Failed to load module %s", path)

    # ---- Event broadcasting ----

    def on_output(self, model_name: str, text: str):
        """Broadcast an agent output chunk to all modules."""
        for mod in self._modules:
            try:
                mod.on_output(model_name, text)
            except Exception:
                logger.exception("Module %s raised in on_output", type(mod).__name__)

    def on_user_input(self, text: str) -> bool:
        """Offer user input to each module in order.

        Returns True (and short-circuits) when a module consumes the input.
        """
        for mod in self._modules:
            try:
                if mod.on_user_input(text):
                    return True
            except Exception:
                logger.exception("Module %s raised in on_user_input", type(mod).__name__)
        return False

    def shutdown(self):
        """Call on_unload on every module (best-effort)."""
        for mod in self._modules:
            try:
                mod.on_unload()
            except Exception:
                logger.exception("Module %s raised in on_unload", type(mod).__name__)
        self._modules.clear()
