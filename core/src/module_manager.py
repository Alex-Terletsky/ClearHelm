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

    def __init__(self, submit_prompt_fn, get_agent_names_fn, ready_models_fn,
                 emit_fn, get_module_config_fn=None, set_module_config_fn=None,
                 request_action_fn=None):
        self._submit_prompt = submit_prompt_fn
        self._get_agent_names = get_agent_names_fn
        self._ready_models = ready_models_fn
        self._emit = emit_fn
        self._get_module_config = get_module_config_fn
        self._set_module_config = set_module_config_fn
        self._request_action_fn = request_action_fn

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

    def get_config(self, module_name: str) -> dict:
        """Return the active agent's config for *module_name*, with schema defaults merged."""
        if self._get_module_config:
            return self._get_module_config(module_name)
        return {}

    def set_config(self, module_name: str, key: str, value):
        """Write a single key into the active agent's module config."""
        if self._set_module_config:
            self._set_module_config(module_name, key, value)

    def request_action(self, command_name: str, agent_name: str,
                       description: str, action, on_deny=None):
        """Request an interceptable action.  Base: always execute immediately."""
        action()

    def check_intercept(self, command_name: str, agent_name: str,
                        description: str = "") -> bool:
        """Check if a command would be intercepted.  Base: always approve."""
        return True


class BoundModuleContext(ModuleContext):
    """Per-module wrapper.  Delegates to shared context, injects module name."""

    def __init__(self, shared: ModuleContext, module_name: str,
                 get_agent_config_fn=None, intercept_defaults: dict | None = None,
                 module_schemas: dict | None = None):
        # Do NOT call super().__init__() — we delegate everything via __getattr__
        self._shared = shared
        self._module_name = module_name
        self._bound_get_agent_config = get_agent_config_fn
        self._intercept_defaults = intercept_defaults or {}
        self._module_schemas = module_schemas or {}
        self._current_agent = None  # set by ModuleManager during callbacks

    def __getattr__(self, name):
        return getattr(self._shared, name)

    def get_config(self, module_name: str) -> dict:
        """Reads config for the correct agent automatically.

        During on_output: reads from the source agent (set by ModuleManager).
        During on_user_input: falls back to UI-active agent (shared context).
        Modules just call ctx.get_config("my_module") — always correct.
        """
        agent = self._current_agent
        if agent and self._bound_get_agent_config:
            cfg = self._bound_get_agent_config(agent)
            if cfg is not None:
                merged = dict(cfg.module_config.get(module_name, {}))
                # Merge schema defaults so unset params have correct values
                for p in self._module_schemas.get(module_name, []):
                    merged.setdefault(p.name, p.default)
                return merged
        # Fallback: use shared context's active-agent resolution
        return self._shared.get_config(module_name)

    def _should_intercept(self, command_name: str, agent_name: str) -> bool:
        """Check if this command needs user approval for this agent."""
        if not self._bound_get_agent_config or not self._shared._request_action_fn:
            return False
        schema_default = self._intercept_defaults.get(command_name, False)
        cfg = self._bound_get_agent_config(agent_name)
        if not cfg:
            return schema_default
        return cfg.module_intercept.get(
            self._module_name, {}).get(command_name, schema_default)

    def check_intercept(self, command_name: str, agent_name: str,
                        description: str = "") -> bool:
        return not self._should_intercept(command_name, agent_name)

    def request_action(self, command_name, agent_name, description,
                       action, on_deny=None):
        if self._should_intercept(command_name, agent_name):
            self._shared._request_action_fn(
                self._module_name, command_name, agent_name,
                description, action, on_deny)
        else:
            action()


class Module:
    """Base class for ClearHelm modules.  All hooks are no-ops by default."""

    NAME: str = ""
    CONFIG_SCHEMA: list = []
    INTERCEPT_SCHEMA: list = []

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

    def __init__(self, modules_dir: str, ctx: ModuleContext,
                 get_agent_config_fn=None):
        self._modules_dir = modules_dir
        self._ctx = ctx
        self._modules: list[Module] = []
        self._module_names: list[str] = []
        self._module_schemas: dict[str, list] = {}
        self._intercept_schemas: dict[str, list] = {}
        self._bound_contexts: list[BoundModuleContext] = []
        self._get_agent_config = get_agent_config_fn
        self._target_agent: str | None = None

    @property
    def module_names(self) -> list[str]:
        return list(self._module_names)

    @property
    def module_schemas(self) -> dict[str, list]:
        return dict(self._module_schemas)

    @property
    def intercept_schemas(self) -> dict[str, list]:
        return dict(self._intercept_schemas)

    def set_target_agent(self, name: str | None):
        """Set the agent used for access gating in on_user_input."""
        self._target_agent = name

    def _is_module_enabled(self, module_name: str, agent_name: str | None) -> bool:
        """Check if *module_name* is enabled for *agent_name*."""
        if not agent_name or not self._get_agent_config:
            return True
        cfg = self._get_agent_config(agent_name)
        if cfg is None:
            return True
        return cfg.module_access.get(module_name, False)

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
        file_stem = os.path.splitext(os.path.basename(path))[0]
        try:
            spec = importlib.util.spec_from_file_location(file_stem, path)
            py_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(py_mod)
            cls = getattr(py_mod, "MODULE_CLASS", None)
            if cls is None:
                logger.warning("Module %s has no MODULE_CLASS attribute — skipped", path)
                return
            instance: Module = cls()
            name = getattr(cls, "NAME", "") or file_stem

            # Collect CONFIG_SCHEMA
            schema = getattr(cls, "CONFIG_SCHEMA", [])
            if schema:
                self._module_schemas[name] = list(schema)

            # Collect INTERCEPT_SCHEMA
            intercept_schema = getattr(cls, "INTERCEPT_SCHEMA", [])
            if intercept_schema:
                self._intercept_schemas[name] = list(intercept_schema)

            # Build intercept defaults from schema
            intercept_defaults = {
                cmd.name: cmd.intercept
                for cmd in intercept_schema
            }

            # Create BoundModuleContext for this module
            bctx = BoundModuleContext(
                shared=self._ctx,
                module_name=name,
                get_agent_config_fn=self._get_agent_config,
                intercept_defaults=intercept_defaults,
                module_schemas=self._module_schemas,
            )

            instance.on_load(bctx)
            self._modules.append(instance)
            self._module_names.append(name)
            self._bound_contexts.append(bctx)
        except Exception:
            logger.exception("Failed to load module %s", path)

    # ---- Event broadcasting ----

    def on_output(self, model_name: str, text: str):
        """Broadcast an agent output chunk to all modules (access-gated)."""
        for mod, mod_name, bctx in zip(self._modules, self._module_names,
                                       self._bound_contexts):
            if not self._is_module_enabled(mod_name, model_name):
                continue
            try:
                bctx._current_agent = model_name
                mod.on_output(model_name, text)
            except Exception:
                logger.exception("Module %s raised in on_output", type(mod).__name__)
            finally:
                bctx._current_agent = None

    def on_user_input(self, text: str) -> bool:
        """Offer user input to each module in order (access-gated by target agent).

        Returns True (and short-circuits) when a module consumes the input.
        """
        for mod, mod_name in zip(self._modules, self._module_names):
            if not self._is_module_enabled(mod_name, self._target_agent):
                continue
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
        self._module_names.clear()
        self._module_schemas.clear()
        self._intercept_schemas.clear()
        self._bound_contexts.clear()
