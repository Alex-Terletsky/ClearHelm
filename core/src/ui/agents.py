import os

from params import RunnerConfig

from .constants import _AGENTS_DIR


def _load_agent_configs() -> list[RunnerConfig]:
    if not os.path.isdir(_AGENTS_DIR):
        return []
    configs = []
    for fname in sorted(os.listdir(_AGENTS_DIR)):
        if fname.lower().endswith(".json"):
            try:
                cfg = RunnerConfig.from_file(os.path.join(_AGENTS_DIR, fname))
                configs.append(cfg)
            except Exception:
                pass
    return configs


def _save_agent_config(cfg: RunnerConfig):
    os.makedirs(_AGENTS_DIR, exist_ok=True)
    cfg.to_file(os.path.join(_AGENTS_DIR, f"{cfg.model_name}.json"))


def _delete_agent_config(name: str):
    path = os.path.join(_AGENTS_DIR, f"{name}.json")
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
