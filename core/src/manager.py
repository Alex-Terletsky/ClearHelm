"""Orchestration layer for managing multiple models simultaneously.

ModelManager owns a dict of named RunnerService instances and tracks which
one is "active". Handles model discovery, registration, lifecycle, prompt
routing, and config/group propagation.
"""

import os
from typing import Callable

from params import (
    RunnerConfig, ModelConfig, GenerationConfig,
    PARAMETER_GROUPS,
)
from runner import RunnerService, ServiceState


def _discover_files(directory: str, extension: str) -> list[dict]:
    """Scan *directory* for files ending with *extension*.

    Returns ``[{name, path}]`` sorted by filename, where *name* is the
    stem (filename without extension).
    """
    if not os.path.isdir(directory):
        return []
    results = []
    for fname in sorted(os.listdir(directory)):
        if fname.lower().endswith(extension):
            results.append({
                "name": os.path.splitext(fname)[0],
                "path": os.path.join(directory, fname),
            })
    return results


def discover_models(models_dir: str) -> list[dict]:
    """Scan *models_dir* for .gguf files.  Returns ``[{name, path}]``."""
    return _discover_files(models_dir, ".gguf")


def discover_configs(configs_dir: str) -> list[dict]:
    """Scan *configs_dir* for .json config files.  Returns ``[{name, path}]``."""
    return _discover_files(configs_dir, ".json")


def load_config(config_path: str, model_path: str = "") -> RunnerConfig:
    """Load RunnerConfig from JSON, or create defaults."""
    if os.path.isfile(config_path):
        return RunnerConfig.from_file(config_path, model_path=model_path)
    return RunnerConfig(
        model_config=ModelConfig(model_path=model_path),
        model_name=(
            os.path.splitext(os.path.basename(model_path))[0]
            if model_path else "model"
        ),
        active_groups=["essential", "visibility"],
    )


class ModelManager:
    """Manages multiple RunnerService instances."""

    def __init__(self, models_dir: str, config_path: str = "",
                 output_callback: Callable[[str, str], None] | None = None):
        """
        Parameters
        ----------
        models_dir : str
            Directory to scan for .gguf model files.
        config_path : str
            Path to a default ``runner_config.json`` (used when adding a
            model that has no per-model config).
        output_callback : callable(model_name, text)
            Receives ``(model_name, text)`` for every chunk of output
            from any managed service.
        """
        self.models_dir = models_dir
        self.config_path = config_path
        self._output_callback = output_callback or self._default_output
        self._services: dict[str, RunnerService] = {}
        self._configs: dict[str, RunnerConfig] = {}
        self._active: str | None = None

    # ---- Output ----

    @staticmethod
    def _default_output(model_name: str, text: str):
        print(f"[{model_name}] {text}", end="")

    def _make_service_callback(self, name: str) -> Callable[[str], None]:
        """Return a single-arg callback that tags output with *name*."""
        def _cb(text: str):
            self._output_callback(name, text)
        return _cb

    # ---- Discovery ----

    def discover_models(self) -> list[dict]:
        return discover_models(self.models_dir)

    # ---- Lifecycle ----

    def add_model(self, name: str, model_path: str,
                  config: RunnerConfig | None = None) -> RunnerConfig:
        """Register a model.  Creates a ``RunnerService`` in IDLE state."""
        if name in self._services:
            raise ValueError(f"Model '{name}' already added")

        if config is None:
            config = load_config(self.config_path, model_path=model_path)
        config.model_name = name
        config.model_config.model_path = model_path

        svc = RunnerService(
            config=config,
            output_callback=self._make_service_callback(name),
        )
        self._services[name] = svc
        self._configs[name] = config
        return config

    def remove_model(self, name: str):
        """Stop (if running) and remove a model entirely."""
        svc = self._services.pop(name, None)
        self._configs.pop(name, None)
        if svc is not None:
            svc.stop()
        if self._active == name:
            self._active = None

    def start_model(self, name: str):
        """Start (load) a registered model."""
        svc = self._services.get(name)
        if svc is None:
            raise KeyError(f"Unknown model: {name}")
        svc.start()

    def stop_model(self, name: str):
        """Stop (unload) a registered model."""
        svc = self._services.get(name)
        if svc is None:
            raise KeyError(f"Unknown model: {name}")
        svc.stop()
        if self._active == name:
            self._active = None

    # ---- Active model ----

    def set_active(self, name: str | None):
        if name is not None and name not in self._services:
            raise KeyError(f"Unknown model: {name}")
        self._active = name

    @property
    def active(self) -> str | None:
        return self._active

    # ---- Prompt routing ----

    def submit_prompt(self, prompt: str, model_name: str | None = None):
        """Route a prompt to the active (or specified) model."""
        target = model_name or self._active
        if target is None:
            self._output_callback("system", "No active model selected.\n")
            return
        svc = self._services.get(target)
        if svc is None:
            self._output_callback("system", f"Unknown model: {target}\n")
            return
        svc.submit_prompt(prompt)

    # ---- Status / introspection ----

    def get_state(self, name: str) -> ServiceState:
        svc = self._services.get(name)
        if svc is None:
            raise KeyError(f"Unknown model: {name}")
        return svc.state

    def get_all_status(self) -> dict[str, ServiceState]:
        return {name: svc.state for name, svc in self._services.items()}

    def get_config(self, name: str) -> RunnerConfig:
        cfg = self._configs.get(name)
        if cfg is None:
            raise KeyError(f"Unknown model: {name}")
        return cfg

    def get_service(self, name: str) -> RunnerService:
        svc = self._services.get(name)
        if svc is None:
            raise KeyError(f"Unknown model: {name}")
        return svc

    @property
    def model_names(self) -> list[str]:
        return list(self._services.keys())

    def ready_models(self) -> list[str]:
        """Return names of models in READY state."""
        return [
            name for name, svc in self._services.items()
            if svc.state == ServiceState.READY
        ]

    # ---- Parameter groups ----

    def update_active_groups(self, name: str, groups: list[str]):
        """Update the active parameter groups for a specific model."""
        cfg = self._configs.get(name)
        svc = self._services.get(name)
        if cfg is None or svc is None:
            raise KeyError(f"Unknown model: {name}")
        cfg.active_groups = list(groups)
        svc.update_active_groups(groups)

    # ---- Cleanup ----

    def shutdown(self):
        """Stop all services."""
        for name in list(self._services):
            self._services[name].stop()
        self._services.clear()
        self._configs.clear()
        self._active = None
