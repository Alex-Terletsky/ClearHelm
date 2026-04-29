"""Chat template discovery, loading, and prompt formatting.

Templates are JSON files in the ``templates/`` directory.  Each defines
prefix/suffix strings for system, user, and assistant turns plus optional
stop tokens.  The ``"none"`` sentinel bypasses formatting entirely.
"""

import json
import os

_REQUIRED_KEYS = {
    "name", "system_prefix", "system_suffix",
    "user_prefix", "user_suffix", "assistant_prefix",
}


def _extract_text(content) -> str:
    """Extract text from content that may be a string or a list of content blocks.

    Handles both plain string content and OpenAI-style content arrays
    (e.g. [{"type": "text", "text": "..."}, {"type": "image_path", ...}]).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return " ".join(parts) if parts else ""
    return str(content)

# Module-level cache: {abs_path: parsed_dict}
_template_cache: dict[str, dict] = {}


def discover_templates(templates_dir: str) -> list[dict]:
    """Scan *templates_dir* for ``.json`` files and return ``[{name, path}]``."""
    results = []
    if not os.path.isdir(templates_dir):
        return results
    for fname in sorted(os.listdir(templates_dir)):
        if not fname.endswith(".json"):
            continue
        stem = os.path.splitext(fname)[0]
        results.append({"name": stem, "path": os.path.join(templates_dir, fname)})
    return results


def load_template(templates_dir: str, name: str) -> dict:
    """Load and validate a template by *name* (filename stem).

    Returns the parsed dict.  Results are cached so repeated calls don't
    re-read the file.
    """
    path = os.path.join(templates_dir, f"{name}.json")
    if path in _template_cache:
        return _template_cache[path]

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    missing = _REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Template {name!r} missing keys: {missing}")

    _template_cache[path] = data
    return data


def apply_template(template: dict, system_prompt: str, user_message: str) -> str:
    """Build a formatted prompt string from template fields."""
    parts = []
    if system_prompt:
        parts.append(template["system_prefix"] + system_prompt + template["system_suffix"])
    parts.append(template["user_prefix"] + user_message + template["user_suffix"])
    parts.append(template["assistant_prefix"])
    return "".join(parts)


def apply_multiturn_template(
    template: dict,
    system_prompt: str,
    history: list[dict],
    user_message: str,
) -> str:
    """Build a multi-turn prompt from prior conversation turns.

    *history* is ``[{"role": "user"|"assistant", "content": str}, ...]``.
    An empty *history* produces output identical to ``apply_template()``.
    """
    assistant_suffix = template.get("assistant_suffix", "")
    parts = []
    if system_prompt:
        parts.append(template["system_prefix"] + system_prompt + template["system_suffix"])
    for turn in history:
        text = _extract_text(turn["content"])
        if turn["role"] == "user":
            parts.append(template["user_prefix"] + text + template["user_suffix"])
        elif turn["role"] == "assistant":
            parts.append(template["assistant_prefix"] + text + assistant_suffix)
    parts.append(template["user_prefix"] + user_message + template["user_suffix"])
    parts.append(template["assistant_prefix"])
    return "".join(parts)


def trim_history(history: list[dict], max_turns: int) -> list[dict]:
    """Keep the last *max_turns* complete user+assistant pairs."""
    h = history[-(max_turns * 2):]
    if h and h[0]["role"] != "user":
        h = h[1:]
    return h


def get_stop_tokens(template: dict) -> list[str]:
    """Return the template's stop token list (may be empty)."""
    return list(template.get("stop_tokens", []))
