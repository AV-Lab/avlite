"""YAML profile load/save for AVLite settings classes (one file per profile).

Each profile is a single ``configs/<profile>.yaml`` file whose top-level keys are
the stack layer sections (``c10_perception`` … ``c40_execution``), the app section
(``c69_apps``), and a ``plugins`` mapping keyed by plugin directory name::

    c10_perception: { ... }
    c40_execution:  { ... }
    c69_apps:       { c60_selected_profile: default, c62_load_plugins: true, ... }
    plugins:
      p60_visualizer_tk: { ... }
      <community_plugin>: { ... }
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Protocol

import yaml

from avlite.c60_apps.c64_settings_schema import (
    SETTINGS_META,
    PlainBinder,
    SettingsValidationError,
    apply_validated_to_setting,
    dump_from_setting,
    schema_of,
    setting_key,
    validate_profile,
)
from avlite.c60_apps.c68_paths import (
    COMMUNITY_DEV_SUBDIR,
    PRIVATE_DEV_SUBDIR,
    ConfigPaths,
    PluginPaths,
)

log = logging.getLogger(__name__)

PLUGINS_SECTION = "plugins"

STACK_LAYER_SECTIONS = frozenset(
    {"c10_perception", "c20_planning", "c30_control", "c40_execution"}
)

class SettingsBinder(Protocol):
    def get_value(self, setting: Any, attr_name: str) -> Any: ...
    def set_value(self, setting: Any, attr_name: str, value: Any) -> None: ...


def profile_file_path(profile: str, *, for_write: bool = False) -> str:
    """Resolve the YAML path for *profile* (``configs/<profile>.yaml``)."""
    return ConfigPaths.effective_path(f"configs/{profile}.yaml", for_write=for_write)


def setting_section(setting) -> tuple[str | None, str]:
    """Return ``(group, key)`` locating *setting* inside a profile file.

    ``group`` is ``"plugins"`` for plugin settings (nested under the ``plugins``
    mapping), or ``None`` for layer/app settings (top-level). ``key`` is the
    section name (layer/app stem, or plugin directory name).
    """
    plugin_dir = _SettingResolution.plugin_dir_from_module(setting)
    if plugin_dir:
        return (PLUGINS_SECTION, plugin_dir)

    fp = getattr(setting, "filepath", "") or ""
    base = Path(fp).name
    if base.startswith("plugin_") and base.endswith(".yaml"):
        return (PLUGINS_SECTION, base[len("plugin_") : -len(".yaml")])
    if fp:
        return (None, Path(fp).stem)
    return (None, setting_key(setting))


def _load_profile_config(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _get_section(config: dict, group: str | None, key: str) -> Any:
    if group == PLUGINS_SECTION:
        plugins = config.get(PLUGINS_SECTION)
        return plugins.get(key) if isinstance(plugins, dict) else None
    return config.get(key)


def _set_section(config: dict, group: str | None, key: str, value: dict) -> None:
    if group == PLUGINS_SECTION:
        config.setdefault(PLUGINS_SECTION, {})[key] = value
    else:
        config[key] = value


def save_setting(
    setting,
    profile: str = "default",
    *,
    binder: SettingsBinder | None = None,
) -> None:
    """Save *setting* into its section of ``configs/<profile>.yaml``."""
    bind = binder or PlainBinder()
    group, key = setting_section(setting)
    write_path = profile_file_path(profile, for_write=True)
    config = _load_profile_config(profile_file_path(profile, for_write=False))
    schema = _SettingResolution.schema_for(setting)

    if schema is not None:
        section_data = dump_from_setting(
            setting, schema, filepath=write_path, profile=profile, binder=bind
        )
    else:
        section_data = {}
        exclude = _SettingResolution.setting_exclude(setting)
        for attr_name, attr_value in vars(setting).items():
            if callable(attr_value) or attr_name.startswith("_") or attr_name in exclude:
                continue
            section_data[attr_name] = bind.get_value(setting, attr_name)

    _set_section(config, group, key, section_data)
    os.makedirs(os.path.dirname(write_path) or ".", exist_ok=True)
    with open(write_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    log.info("Saved %s to %s (profile '%s')", key, write_path, profile)


def load_setting(
    setting,
    profile: str = "default",
    *,
    strict: bool = False,
    binder: SettingsBinder | None = None,
) -> bool:
    """Load *setting* from its section of ``configs/<profile>.yaml``. True on success."""
    bind = binder or PlainBinder()
    group, key = setting_section(setting)
    filepath = profile_file_path(profile, for_write=False)
    try:
        with open(filepath, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        log.error("YAML syntax error in %s: %s", filepath, e)
        return False
    except (FileNotFoundError, OSError) as e:
        if isinstance(e, FileNotFoundError) or getattr(e, "errno", None) == 2:
            log.debug("No profile file at %s; using defaults", filepath)
            return False
        log.error("Failed to read %s: %s", filepath, e)
        return False

    if not config:
        log.warning("Empty or invalid profile file: %s", filepath)
        return False

    section = _get_section(config, group, key)
    schema = _SettingResolution.schema_for(setting)
    try:
        if schema is not None:
            if not isinstance(section, dict) or not section:
                if not isinstance(section, dict):
                    log.debug(
                        "Section '%s' not found in %s (profile '%s'); applying schema defaults",
                        key,
                        filepath,
                        profile,
                    )
                else:
                    log.debug("Section '%s' empty; applying schema defaults", key)
                section = {}
            try:
                validated = validate_profile(schema, section, filepath=filepath, profile=profile)
            except SettingsValidationError as e:
                log.error(str(e))
                if strict:
                    raise
                return False
            apply_validated_to_setting(setting, validated, binder=bind)
        else:
            if not isinstance(section, dict) or not section:
                log.debug("Section '%s' not found in %s (profile '%s')", key, filepath, profile)
                return False
            exclude = _SettingResolution.setting_exclude(setting)
            for attr_name, value in section.items():
                if attr_name in exclude:
                    continue
                if not hasattr(setting, attr_name):
                    log.warning("Skipping unknown attribute: %s", attr_name)
                    continue
                bind.set_value(setting, attr_name, value)
        log.info("Loaded %s from %s (profile '%s')", key, filepath, profile)
        return True
    except SettingsValidationError:
        raise
    except Exception as e:
        log.error("Failed to load %s: %s", key, e)
        return False


def can_remove_builtin_plugin(plugin_name: str, hosting_plugin_name: str) -> str | None:
    """Return an error message if *plugin_name* must not be removed, else ``None``."""
    if plugin_name == hosting_plugin_name:
        return (
            f"Cannot remove {plugin_name!r} while this settings window is running. "
            "Remove other plugins, save, and restart with a different app if needed."
        )
    return None


def delete_profile(profile: str) -> bool:
    """Delete the ``configs/<profile>.yaml`` file. The ``default`` profile is protected."""
    if profile == "default":
        log.warning("Cannot delete the 'default' profile.")
        return False
    path = Path(profile_file_path(profile, for_write=True))
    if not path.is_file():
        log.warning("Profile '%s' does not exist at %s", profile, path)
        return False
    path.unlink()
    log.info("Deleted profile '%s' (%s)", profile, path)
    return True


def rename_profile(old_profile: str, new_profile: str) -> bool:
    """Rename ``configs/<old>.yaml`` to ``configs/<new>.yaml`` and refresh selection."""
    if old_profile == "default":
        log.warning("Cannot rename the 'default' profile.")
        return False
    src = Path(profile_file_path(old_profile, for_write=True))
    dst = Path(profile_file_path(new_profile, for_write=True))
    if not src.is_file():
        log.warning("Profile '%s' does not exist at %s", old_profile, src)
        return False
    if dst.exists():
        log.warning("Profile '%s' already exists at %s", new_profile, dst)
        return False
    src.rename(dst)

    config = _load_profile_config(str(dst))
    apps = config.get("c69_apps")
    if isinstance(apps, dict) and "c60_selected_profile" in apps:
        apps["c60_selected_profile"] = new_profile
        with open(dst, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    log.info("Renamed profile '%s' to '%s'", old_profile, new_profile)
    return True


def order_profiles_for_dropdown(profiles: list[str]) -> list[str]:
    """Return profile names with ``default`` first for UI dropdowns."""
    if not profiles:
        return []
    rest = [p for p in profiles if p != "default"]
    return (["default"] if "default" in profiles else []) + rest


def list_profiles(setting: Any = None) -> list:
    """List available profiles from the active config target (dev clone) or user+repo (pip)."""
    if ConfigPaths.can_edit_bundled() and not ConfigPaths.is_repo_target():
        ConfigPaths.copy_bundled_profiles_to_user()
    if ConfigPaths.can_edit_bundled():
        directories = (
            [ConfigPaths.bundled_dir()]
            if ConfigPaths.is_repo_target()
            else [ConfigPaths.user_dir()]
        )
    else:
        directories = [ConfigPaths.user_dir(), ConfigPaths.bundled_dir()]
    names: set[str] = set()
    for directory in directories:
        for path in ConfigPaths.iter_profile_paths(directory):
            names.add(path.stem)
    profiles = order_profiles_for_dropdown(sorted(names))
    log.debug("Available profiles: %s", profiles)
    return profiles


def dev_mode_export_warning(community_plugins: dict[str, str] | None = None) -> str | None:
    """Return a user-facing warning when exporting under Plugins dev mode, else None."""
    if not PluginPaths.is_dev_mode():
        return None
    lines = [
        "Plugins dev mode is enabled. This profile uses repo-relative plugin paths "
        f"({COMMUNITY_DEV_SUBDIR}/, {PRIVATE_DEV_SUBDIR}/).",
        "",
        "On the target machine you must:",
        "  \u2022 Enable Plugins dev mode (same checkout directories)",
        "  \u2022 Install all community and member plugins referenced in the profile",
        "",
        "Plugin paths will not resolve on a machine that uses only "
        f"{PluginPaths.format_display(PluginPaths.install_dir())}/.",
    ]
    plugins = community_plugins or {}
    missing = [
        name
        for name, stored in sorted(plugins.items())
        if PluginPaths.load_path(name, stored) is None
    ]
    if missing:
        lines.extend(["", "Plugins not found locally:", ", ".join(missing)])
    return "\n".join(lines)


def dev_mode_uninstall_warning(plugins_dir: Path, name: str) -> str | None:
    """Return a user-facing warning when uninstalling a dev checkout, else None."""
    if not PluginPaths.is_dev_mode():
        return None
    resolved_dir = plugins_dir.resolve()
    dev_roots = (
        PluginPaths.community_dev_dir().resolve(),
        PluginPaths.private_dev_dir().resolve(),
    )
    if resolved_dir not in dev_roots:
        return None
    checkout = (resolved_dir / name).resolve()
    path_display = PluginPaths.format_display(checkout)
    return "\n".join(
        [
            "Plugins dev mode is enabled.",
            "",
            "Uninstall will permanently delete the plugin source checkout at:",
            f"  {path_display}",
            "",
            "Commit or back up any local changes in git before continuing.",
        ]
    )


def export_profile(
    profile: str,
    out_path: Path | str,
    *,
    include_stack: bool = True,
    include_app: bool = True,
    include_plugins: bool = True,
) -> int:
    """Export *profile* to a single ``.yaml`` file. Returns the number of sections written.

    ``include_stack`` controls the four core layer sections (``c10_perception`` …
    ``c40_execution``); ``include_app`` controls ``c69_apps``; ``include_plugins``
    controls the ``plugins`` mapping.
    """
    if not include_stack and not include_app and not include_plugins:
        raise ValueError("Nothing selected to export")

    read_path = profile_file_path(profile, for_write=False)
    if not os.path.exists(read_path):
        raise ValueError(f"Profile '{profile}' not found ({read_path})")
    config = _load_profile_config(read_path)
    if not config:
        raise ValueError(f"Profile '{profile}' is empty ({read_path})")

    out: dict[str, Any] = {}
    for key, value in config.items():
        if key == PLUGINS_SECTION:
            if include_plugins:
                out[key] = value
            continue
        if key == "c69_apps" and not include_app:
            continue
        if key in STACK_LAYER_SECTIONS and not include_stack:
            continue
        out[key] = value

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        yaml.dump(out, f, default_flow_style=False)
    log.info("Exported profile '%s' to %s (%d section(s))", profile, out_file, len(out))
    return len(out)


def _validate_profile_sections(data: dict) -> None:
    """Validate known layer/app/built-in-plugin sections against their schemas."""
    from avlite.c60_apps.c62_factory import get_stack_settings_classes

    schema_by_section: dict[tuple[str | None, str], Any] = {}
    for cls in get_stack_settings_classes():
        schema = schema_of(cls)
        if schema is None:
            continue
        schema_by_section[setting_section(cls)] = schema

    for key, value in data.items():
        if key == PLUGINS_SECTION:
            if isinstance(value, dict):
                for plugin_key, plugin_value in value.items():
                    schema = schema_by_section.get((PLUGINS_SECTION, plugin_key))
                    if schema is not None and isinstance(plugin_value, dict):
                        validate_profile(schema, plugin_value, profile=plugin_key)
            continue
        schema = schema_by_section.get((None, key))
        if schema is not None and isinstance(value, dict):
            validate_profile(schema, value, profile=key)


def import_profile(
    in_path: Path | str,
    *,
    overwrite: bool = False,
    profile: str | None = None,
    validate: bool = True,
) -> str:
    """Import a profile ``.yaml`` file into ``configs/``. Returns the profile name.

    The target profile name defaults to the imported file's stem. Existing sections
    are merged unless *overwrite* replaces the whole profile. Known sections are
    validated against their schemas before anything is written.
    """
    src = Path(in_path)
    with open(src, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Invalid or empty profile file: {src}")

    if validate:
        _validate_profile_sections(data)

    profile_name = profile or src.stem
    dest = profile_file_path(profile_name, for_write=True)
    dest_exists = os.path.exists(dest)
    if dest_exists and not overwrite:
        raise ValueError(f"Profile '{profile_name}' already exists")

    existing = _load_profile_config(dest) if dest_exists else {}
    merged: dict[str, Any] = {} if overwrite else dict(existing)
    for key, value in data.items():
        if key == PLUGINS_SECTION and isinstance(value, dict):
            plugins = merged.setdefault(PLUGINS_SECTION, {})
            if isinstance(plugins, dict):
                plugins.update(value)
            else:
                merged[PLUGINS_SECTION] = dict(value)
        else:
            merged[key] = value

    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        yaml.dump(merged, f, default_flow_style=False)
    log.info("Imported profile '%s' from %s", profile_name, src)
    return profile_name


class _SettingResolution:
    """Resolve setting module paths, excludes, and schemas (module-private)."""

    @staticmethod
    def setting_module(setting) -> str:
        return setting.__module__ if isinstance(setting, type) else type(setting).__module__

    @staticmethod
    def plugin_dir_from_module(setting) -> str | None:
        """Return the plugin directory name if *setting* lives under ``avlite.plugins``."""
        parts = _SettingResolution.setting_module(setting).split(".")
        if len(parts) >= 3 and parts[0] == "avlite" and parts[1] == "plugins":
            return parts[2]
        return None

    @staticmethod
    def setting_exclude(setting) -> set[str]:
        exclude = set(getattr(setting, "exclude", []))
        exclude.update(SETTINGS_META)
        return exclude

    @staticmethod
    def schema_for(setting):
        return schema_of(setting)
