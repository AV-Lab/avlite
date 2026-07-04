#!/usr/bin/env python3
"""One-time migration: per-layer YAML configs -> per-profile YAML files.

Old layout (one file per layer/plugin, profiles as top-level keys)::

    configs/c10_perception.yaml   -> { default: {...}, B: {...}, ... }
    configs/c59_apps.yaml
    configs/plugin_p50_visualizer_tk.yaml

New layout (one file per profile, sections keyed by layer/app/plugin)::

    configs/default.yaml -> {c10_perception: {...}, ..., c69_apps: {...}, plugins: {...}}

The app section is renamed ``c59_apps`` -> ``c69_apps`` and its field prefixes
(``c52_*`` -> ``c62_*``, ``c50_selected_profile`` -> ``c60_selected_profile``) are
updated. Built-in plugin directories ``p50_*`` become ``p60_*`` and the visualizer
plugin's field prefixes (``p50_/p56_/p57_/p58_`` -> ``p60_/p66_/p67_/p68_``) are
renamed. Layer sections (``c10``-``c40``) are copied unchanged.

Usage::

    python scripts/migrate_configs.py [--dir configs] [--dry-run] [--keep-old]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

LAYER_FILES = ("c10_perception", "c20_planning", "c30_control", "c40_execution")
APP_FILE = "c59_apps"
APP_SECTION = "c69_apps"

APP_FIELD_RENAMES = {
    "c52_load_plugins": "c62_load_plugins",
    "c52_default_plugins": "c62_default_plugins",
    "c52_community_plugins": "c62_community_plugins",
    "c50_selected_profile": "c60_selected_profile",
}

# Visualizer field prefix renames (consumer module numbers shift 5x -> 6x).
VIZ_PREFIX_RENAMES = tuple((f"p5{d}_", f"p6{d}_") for d in range(10))


def rename_plugin_dir(name: str) -> str:
    return "p60_" + name[len("p50_"):] if name.startswith("p50_") else name


def rename_plugin_name_value(value: str) -> str:
    return rename_plugin_dir(value)


def migrate_app_section(data: dict) -> dict:
    out: dict = {}
    for key, value in data.items():
        new_key = APP_FIELD_RENAMES.get(key, key)
        if new_key == "c62_default_plugins" and isinstance(value, list):
            value = [rename_plugin_name_value(v) for v in value]
        out[new_key] = value
    return out


def migrate_viz_section(data: dict) -> dict:
    out: dict = {}
    for key, value in data.items():
        new_key = key
        for old, new in VIZ_PREFIX_RENAMES:
            if new_key.startswith(old):
                new_key = new + new_key[len(old):]
                break
        out[new_key] = value
    return out


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def collect(config_dir: Path) -> dict[str, dict]:
    """Return {profile_name: {section: data, ..., plugins: {...}}}."""
    profiles: dict[str, dict] = {}

    def ensure(profile: str) -> dict:
        return profiles.setdefault(profile, {})

    # Core layer files (unchanged section keys / fields).
    for layer in LAYER_FILES:
        data = load_yaml(config_dir / f"{layer}.yaml")
        for profile, section in data.items():
            if isinstance(section, dict):
                ensure(profile)[layer] = section

    # App file -> c69_apps section with field renames.
    for profile, section in load_yaml(config_dir / f"{APP_FILE}.yaml").items():
        if isinstance(section, dict):
            ensure(profile)[APP_SECTION] = migrate_app_section(section)

    # Plugin files -> plugins.<new_name> subsections.
    for path in sorted(config_dir.glob("plugin_*.yaml")):
        old_name = path.stem[len("plugin_"):]
        new_name = rename_plugin_dir(old_name)
        is_viz = old_name == "p50_visualizer_tk"
        for profile, section in load_yaml(path).items():
            if not isinstance(section, dict):
                continue
            section = migrate_viz_section(section) if is_viz else section
            ensure(profile).setdefault("plugins", {})[new_name] = section

    return profiles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="configs", help="Config directory to migrate (default: configs)")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing")
    parser.add_argument("--keep-old", action="store_true", help="Do not delete the old per-layer files")
    args = parser.parse_args()

    config_dir = Path(args.dir)
    if not config_dir.is_dir():
        print(f"Config directory not found: {config_dir}", file=sys.stderr)
        return 1

    old_files = [config_dir / f"{n}.yaml" for n in (*LAYER_FILES, APP_FILE)]
    old_files += sorted(config_dir.glob("plugin_*.yaml"))
    old_files = [p for p in old_files if p.is_file()]
    if not old_files:
        print("No legacy per-layer config files found; nothing to migrate.")
        return 0

    profiles = collect(config_dir)
    if not profiles:
        print("No profiles found in legacy files; nothing to migrate.")
        return 0

    for profile, sections in sorted(profiles.items()):
        dest = config_dir / f"{profile}.yaml"
        print(f"{'[dry-run] ' if args.dry_run else ''}write {dest} ({len(sections)} sections)")
        if not args.dry_run:
            with open(dest, "w", encoding="utf-8") as f:
                yaml.dump(sections, f, default_flow_style=False, sort_keys=True)

    if not args.keep_old:
        for path in old_files:
            print(f"{'[dry-run] ' if args.dry_run else ''}remove {path}")
            if not args.dry_run:
                path.unlink()

    print(f"Migrated {len(profiles)} profile(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
