"""Migration for versions older than v1.3.

Moves flat agent config files (agents/*.json) to the per-agent subdirectory
layout introduced in v1.3 (agents/<slug>/config.json).

Run from the project root:
    python scripts/migrate_agents.py

Not needed for fresh installs or projects created on v1.3+.
Safe to re-run — skips files that have already been migrated.
"""

import os
import re
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AGENTS_DIR = os.path.join(_PROJECT_ROOT, "agents")


def _slugify(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    slug = slug.strip('-')
    return slug or 'agent'


def migrate():
    if not os.path.isdir(_AGENTS_DIR):
        print("agents/ directory not found, nothing to do.")
        return

    moved = 0
    skipped = 0
    for fname in sorted(os.listdir(_AGENTS_DIR)):
        if not fname.lower().endswith(".json"):
            continue
        display_name = fname[:-5]
        slug = _slugify(display_name)
        slug_dir = os.path.join(_AGENTS_DIR, slug)
        src = os.path.join(_AGENTS_DIR, fname)

        # Deduplicate if slug already taken
        if os.path.exists(slug_dir):
            config_path = os.path.join(slug_dir, "config.json")
            if os.path.exists(config_path):
                print(f"  SKIP  {fname} -> {slug}/config.json (already exists)")
                skipped += 1
                continue
        # Check for slug collision with a different agent
        if os.path.exists(slug_dir):
            i = 2
            while os.path.exists(f"{slug_dir}-{i}"):
                i += 1
            slug_dir = f"{slug_dir}-{i}"
            slug = os.path.basename(slug_dir)

        dst = os.path.join(slug_dir, "config.json")
        os.makedirs(slug_dir, exist_ok=True)
        os.rename(src, dst)
        print(f"  MOVED {fname} -> {slug}/config.json")
        moved += 1

    print(f"\nDone. Moved: {moved}, Skipped: {skipped}")


if __name__ == "__main__":
    migrate()
