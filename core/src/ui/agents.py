"""Agent config load/save/delete helpers with per-agent subdirectories.

Directory layout::

    agents/
      <slug>/               # slugified agent name
        config.json          # RunnerConfig
        sessions/
          2026-04-12_143022_a1b2c3.json
          ...
"""

import json
import os
import re
import secrets
import shutil
from datetime import datetime
from typing import Callable

from params import RunnerConfig

from .constants import _AGENTS_DIR


# ---- Slug helper ----

def _slugify(name: str) -> str:
    """Convert a display name to a filesystem-safe directory slug."""
    slug = name.lower().strip()
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    slug = slug.strip('-')
    return slug or 'agent'


# ---- Directory mapping (display name -> slug dir) ----

# Maps agent display name to its actual directory path on disk.
# Populated by _load_agent_configs, updated by save/delete.
_agent_dir_map: dict[str, str] = {}


def _find_agent_dir(name: str) -> str | None:
    """Look up the directory for an existing agent by display name."""
    # Fast path: cached
    if name in _agent_dir_map:
        path = _agent_dir_map[name]
        if os.path.isdir(path):
            return path
        del _agent_dir_map[name]

    # Slow path: scan for a config with matching model_name
    if os.path.isdir(_AGENTS_DIR):
        for entry in os.listdir(_AGENTS_DIR):
            entry_path = os.path.join(_AGENTS_DIR, entry)
            config_path = os.path.join(entry_path, "config.json")
            if not os.path.isfile(config_path):
                continue
            try:
                with open(config_path, encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("model_name") == name:
                    _agent_dir_map[name] = entry_path
                    return entry_path
            except (json.JSONDecodeError, OSError):
                continue
    return None


def _agent_dir(name: str) -> str:
    """Return the directory for an agent, creating a deduplicated slug if new."""
    existing = _find_agent_dir(name)
    if existing:
        return existing

    # New agent — generate slug with deduplication
    slug = _slugify(name)
    path = os.path.join(_AGENTS_DIR, slug)
    if not os.path.exists(path):
        _agent_dir_map[name] = path
        return path

    # Slug collision — append incrementing suffix
    i = 2
    while os.path.exists(f"{path}-{i}"):
        i += 1
    deduped = f"{path}-{i}"
    _agent_dir_map[name] = deduped
    return deduped


def _sessions_dir(name: str) -> str:
    return os.path.join(_agent_dir(name), "sessions")


# ---- Config functions ----

def _load_agent_configs() -> list[RunnerConfig]:
    if not os.path.isdir(_AGENTS_DIR):
        return []

    # Scan subdirectories for config.json, populating the dir map
    configs = []
    for entry in sorted(os.listdir(_AGENTS_DIR)):
        entry_path = os.path.join(_AGENTS_DIR, entry)
        if not os.path.isdir(entry_path):
            continue
        config_path = os.path.join(entry_path, "config.json")
        if os.path.isfile(config_path):
            try:
                cfg = RunnerConfig.from_file(config_path)
                _agent_dir_map[cfg.model_name] = entry_path
                configs.append(cfg)
            except Exception:
                pass
    return configs


def _save_agent_config(cfg: RunnerConfig, module_schemas: dict | None = None,
                       intercept_schemas: dict | None = None):
    agent_path = _agent_dir(cfg.model_name)
    os.makedirs(agent_path, exist_ok=True)
    cfg.to_file(os.path.join(agent_path, "config.json"),
                module_schemas=module_schemas,
                intercept_schemas=intercept_schemas)


def _delete_agent_config(name: str):
    agent_path = _find_agent_dir(name)
    if agent_path and os.path.isdir(agent_path):
        shutil.rmtree(agent_path)
    _agent_dir_map.pop(name, None)


# ---- Session functions ----

def _new_session_path(name: str) -> str:
    """Generate a timestamped session path with random hash."""
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    tag = secrets.token_hex(3)
    return os.path.join(_sessions_dir(name), f"{ts}_{tag}.json")


def _load_session_file(path: str) -> list[dict]:
    """Safely load a session JSON file."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _resolve_session(
    name: str,
    session_mode: str,
    session_file: str,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[list[dict], str]:
    """Return (history, resolved_path) based on session mode."""
    def _log(msg: str):
        if log_fn:
            log_fn(f"[session] [{name}] {msg}")

    if session_mode == "recent":
        sdir = _sessions_dir(name)
        if os.path.isdir(sdir):
            files = sorted(
                (f for f in os.listdir(sdir) if f.endswith(".json")),
                key=lambda f: os.path.getmtime(os.path.join(sdir, f)),
                reverse=True,
            )
            if files:
                path = os.path.join(sdir, files[0])
                history = _load_session_file(path)
                _log(f"Loaded recent session: {files[0]} ({len(history)} turns)")
                return history, path
        _log("No recent session found, starting new")
    elif session_mode == "file" and session_file:
        history = _load_session_file(session_file)
        _log(f"Loaded session file: {session_file} ({len(history)} turns)")
        return history, session_file

    # "new" or fallback
    path = _new_session_path(name)
    _log(f"New session: {os.path.basename(path)}")
    return [], path


def _save_session(name: str, session_path: str, history: list[dict]):
    """Write history to the session file."""
    os.makedirs(_sessions_dir(name), exist_ok=True)
    with open(session_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
