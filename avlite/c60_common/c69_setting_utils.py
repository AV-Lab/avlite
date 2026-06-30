"""YAML profile load/save for AVLite settings classes."""

from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path
from typing import Any, Protocol

import yaml

from avlite.c60_common.c66_plugins import load_plugin_settings_class
from avlite.c60_common.c68_settings_schema import (
    SETTINGS_META,
    PlainBinder,
    SettingsValidationError,
    apply_validated_to_setting,
    dump_from_setting,
    schema_of,
    validate_profile,
)
from avlite.c60_common.c67_paths import (
    ConfigPaths,
    PluginPaths,
)

log = logging.getLogger(__name__)


def _setting_module(setting) -> str:
    return setting.__module__ if isinstance(setting, type) else type(setting).__module__


def _plugin_dir_from_module(setting) -> str | None:
    """Return the plugin directory name if *setting* lives under ``avlite.plugins``."""
    parts = _setting_module(setting).split(".")
    if len(parts) >= 3 and parts[0] == "avlite" and parts[1] == "plugins":
        return parts[2]
    return None


def stored_filepath(setting) -> str:
    """Resolve the YAML filepath token for *setting*.

    Uses an explicit ``filepath`` when set; otherwise, for plugin settings imported
    directly (no loader patch), derives ``configs/plugin_<dir>.yaml`` from the module.
    """
    fp = getattr(setting, "filepath", "") or ""
    if fp:
        return fp
    plugin_dir = _plugin_dir_from_module(setting)
    return PluginPaths.settings_filepath(plugin_dir) if plugin_dir else fp


class SettingsBinder(Protocol):
    def get_value(self, setting: Any, attr_name: str) -> Any: ...
    def set_value(self, setting: Any, attr_name: str, value: Any) -> None: ...


def _setting_exclude(setting) -> set[str]:
    exclude = set(getattr(setting, "exclude", []))
    exclude.update(SETTINGS_META)
    return exclude


def _get_schema(setting):
    return schema_of(setting)


def save_setting(
    setting,
    profile: str = "default",
    *,
    binder: SettingsBinder | None = None,
) -> None:
    """Save current configuration to a YAML file."""
    bind = binder or PlainBinder()
    stored = stored_filepath(setting)
    filepath = ConfigPaths.effective_path(stored, for_write=True)
    schema = _get_schema(setting)

    read_path = ConfigPaths.effective_path(stored, for_write=False)
    if os.path.exists(read_path):
        with open(read_path, "r") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    if schema is not None:
        config[profile] = dump_from_setting(
            setting, schema, filepath=filepath, profile=profile, binder=bind
        )
    else:
        config[profile] = {}
        exclude = _setting_exclude(setting)
        target = setting if not isinstance(setting, type) else setting
        for attr_name, attr_value in vars(target).items():
            if callable(attr_value) or attr_name.startswith("_") or attr_name in exclude:
                continue
            config[profile][attr_name] = bind.get_value(setting, attr_name)

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    log.info("Configuration saved to %s for profile '%s'", filepath, profile)


def load_setting(
    setting,
    profile: str = "default",
    *,
    strict: bool = False,
    binder: SettingsBinder | None = None,
) -> bool:
    """Load configuration from a YAML file. Returns True on success."""
    bind = binder or PlainBinder()
    stored = stored_filepath(setting)
    filepath = ConfigPaths.effective_path(stored, for_write=False)
    schema = _get_schema(setting)
    try:
        with open(filepath, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        log.error("YAML syntax error in %s: %s", filepath, e)
        return False
    except OSError as e:
        log.error("Failed to read configuration %s: %s", filepath, e)
        return False

    try:
        if not config:
            log.warning("Empty or invalid configuration file: %s", filepath)
            return False

        profile_dict = config.get(profile, "")
        if not profile_dict:
            log.warning("Profile '%s' not found in %s", profile, filepath)
            return False

        if schema is not None:
            known = set(schema.model_fields.keys())
            unknown = set(profile_dict.keys()) - known - SETTINGS_META
            for key in sorted(unknown):
                log.debug("Skipping unknown key in %s profile '%s': %s", filepath, profile, key)
            try:
                validated = validate_profile(
                    schema, profile_dict, filepath=filepath, profile=profile
                )
            except SettingsValidationError as e:
                log.error(str(e))
                if strict:
                    raise
                return False
            apply_validated_to_setting(setting, validated, binder=bind)
        else:
            exclude = _setting_exclude(setting)
            for attr_name, value in profile_dict.items():
                if attr_name in exclude:
                    continue
                if not hasattr(setting, attr_name):
                    log.warning("Skipping unknown attribute: %s", attr_name)
                    continue
                bind.set_value(setting, attr_name, value)

        log.info("Configuration loaded from %s for profile '%s'", filepath, profile)
        return True
    except SettingsValidationError:
        raise
    except Exception as e:
        log.error("Failed to load configuration: %s", e)
        return False


def delete_setting_profile(setting, profile) -> bool:
    """Delete a profile from the configuration file."""
    filepath = ConfigPaths.effective_path(stored_filepath(setting), for_write=True)
    if profile == "default":
        log.warning("Cannot delete the 'default' profile.")
        return False

    try:
        with open(filepath, "r") as f:
            config = yaml.safe_load(f) or {}

        if profile not in config:
            log.warning("Profile '%s' does not exist in %s", profile, profile)
            return False

        del config[profile]

        with open(filepath, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        log.info("Profile '%s' deleted from %s", profile, filepath)
        return True
    except Exception as e:
        log.error("Failed to delete profile: %s", e)
        return False


def rename_setting_profile(setting, old_profile, new_profile) -> bool:
    """Rename a profile in the configuration file."""
    filepath = ConfigPaths.effective_path(stored_filepath(setting), for_write=True)
    if old_profile == "default":
        log.warning("Cannot rename the 'default' profile.")
        return False

    try:
        with open(filepath, "r") as f:
            config = yaml.safe_load(f) or {}

        if old_profile not in config:
            log.warning("Profile '%s' does not exist in %s", old_profile, filepath)
            return False
        if new_profile in config:
            log.warning("Profile '%s' already exists in %s", new_profile, filepath)
            return False

        config[new_profile] = config.pop(old_profile)

        with open(filepath, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        log.info("Profile '%s' renamed to '%s' in %s", old_profile, new_profile, filepath)
        return True
    except Exception as e:
        log.error("Failed to rename profile: %s", e)
        return False


def list_profiles(setting) -> list:
    """List all profiles in the configuration file."""
    filepath = ConfigPaths.effective_path(stored_filepath(setting), for_write=False)
    try:
        with open(filepath, "r") as f:
            config = yaml.safe_load(f)
        if not config:
            log.warning("Empty or invalid configuration file: %s", filepath)
            return []

        profiles = list(config.keys())
        log.info("Available profiles: %s", profiles)
        return profiles
    except Exception as e:
        log.error("Failed to list profiles: %s", e)
        return []


def _settings_by_basename_map(settings_classes: list[Any]) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for cls in settings_classes:
        mapping[Path(stored_filepath(cls)).name] = cls
    return mapping


def _settings_class_for_basename(
    basename: str,
    settings_classes: list[Any],
) -> Any | None:
    return _settings_by_basename_map(settings_classes).get(basename)


def _validated_profile_dict(
    settings_cls: Any | None,
    profile: str,
    prof_data: dict,
    *,
    filepath: str,
) -> dict:
    if settings_cls is None:
        return prof_data
    schema = _get_schema(settings_cls)
    if schema is None:
        return prof_data
    try:
        validated = validate_profile(schema, prof_data, filepath=filepath, profile=profile)
    except SettingsValidationError as exc:
        raise ValueError(str(exc)) from exc
    return validated.model_dump()


def _profile_dict_in_file(path: Path, profile: str) -> dict | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    prof = data.get(profile)
    if not isinstance(prof, dict):
        return None
    return prof


def _community_sources_for_profile(
    profile: str,
    community_plugins: dict[str, str],
) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for name in community_plugins:
        read_path = Path(
            ConfigPaths.effective_path(PluginPaths.settings_filepath(name), for_write=False)
        )
        if _profile_dict_in_file(read_path, profile) is not None:
            sources[PluginPaths.settings_basename(name)] = read_path
    return sources


def iter_profile_sources(
    profile: str,
    *,
    settings_classes: list[Any],
    community_plugins: dict[str, str] | None = None,
) -> list[tuple[str, Path]]:
    """Return ``(zip_entry_name, read_path)`` for every YAML containing *profile*."""
    sources: dict[str, Path] = {}

    for cls in settings_classes:
        read_path = Path(ConfigPaths.effective_path(stored_filepath(cls), for_write=False))
        if _profile_dict_in_file(read_path, profile) is not None:
            sources[read_path.name] = read_path

    config_dir = ConfigPaths.user_dir()
    if config_dir.is_dir():
        for path in sorted(config_dir.glob("*.yaml")):
            if path.name not in sources and _profile_dict_in_file(path, profile) is not None:
                sources[path.name] = path

    if community_plugins:
        for basename, read_path in _community_sources_for_profile(profile, community_plugins).items():
            sources.setdefault(basename, read_path)
    return sorted(sources.items())


def export_profile(
    profile: str,
    zip_path: Path | str,
    *,
    settings_classes: list[Any],
    community_plugins: dict[str, str] | None = None,
) -> int:
    """Export *profile* from all YAML sources into a zip file. Returns entry count."""
    sources = iter_profile_sources(
        profile, settings_classes=settings_classes, community_plugins=community_plugins
    )
    if not sources:
        raise ValueError(f"Profile '{profile}' not found in any configuration file")

    community_basenames = {
        PluginPaths.settings_basename(name): name
        for name in (community_plugins or {})
    }

    out = Path(zip_path)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for entry_name, read_path in sources:
            with open(read_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            prof_data = data[profile]
            if entry_name in community_basenames:
                plugin_name = community_basenames[entry_name]
                stored = community_plugins[plugin_name]
                cls = load_plugin_settings_class(
                    plugin_name, str(PluginPaths.resolve(plugin_name, stored))
                )
            else:
                cls = _settings_class_for_basename(read_path.name, settings_classes)
            snippet_data = _validated_profile_dict(
                cls, profile, prof_data, filepath=str(read_path)
            )
            snippet = {profile: snippet_data}
            zf.writestr(entry_name, yaml.dump(snippet, default_flow_style=False))
    log.info("Exported profile '%s' to %s (%d file(s))", profile, out, len(sources))
    return len(sources)


def _merge_profile_into_file(
    filepath: str | Path,
    profile: str,
    prof_data: dict,
    *,
    overwrite: bool,
) -> None:
    path = Path(filepath)
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    if profile in config and not overwrite:
        raise ValueError(f"Profile '{profile}' already exists in {path}")

    config[profile] = prof_data
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)
    log.info("Merged profile '%s' into %s", profile, path)


def _community_settings_dest(name: str) -> Path:
    return Path(
        ConfigPaths.effective_path(PluginPaths.settings_filepath(name), for_write=True)
    )


def _community_plugin_name_for_basename(
    basename: str,
    community_plugins: dict[str, str],
) -> str | None:
    for name in community_plugins:
        if PluginPaths.settings_basename(name) == basename:
            return name
    return None


def import_profile(
    zip_path: Path | str,
    *,
    settings_classes: list[Any],
    overwrite: bool = False,
) -> str:
    """Import a profile zip; merge each entry into the corresponding YAML. Returns profile name."""
    src = Path(zip_path)
    standard_entries: list[tuple[str, dict]] = []
    community_entries: list[tuple[str, dict]] = []
    profile_name: str | None = None

    with zipfile.ZipFile(src, "r") as zf:
        for name in zf.namelist():
            if not name.endswith(".yaml"):
                continue
            snippet = yaml.safe_load(zf.read(name)) or {}
            if not isinstance(snippet, dict) or len(snippet) != 1:
                raise ValueError(f"Invalid profile entry in zip: {name}")
            prof, prof_data = next(iter(snippet.items()))
            if not isinstance(prof_data, dict):
                raise ValueError(f"Invalid profile data in zip: {name}")
            if profile_name is None:
                profile_name = prof
            elif profile_name != prof:
                raise ValueError(
                    f"Inconsistent profile names in zip: expected '{profile_name}', got '{prof}' in {name}"
                )
            if name.startswith("community/"):
                community_entries.append((name, prof_data))
            else:
                standard_entries.append((Path(name).name, prof_data))

    if profile_name is None:
        raise ValueError("No YAML entries found in zip")

    standard_entries.sort(key=lambda item: (0 if item[0] == "c40_execution.yaml" else 1, item[0]))

    exec_prof_data: dict | None = None
    for basename, prof_data in standard_entries:
        if basename == "c40_execution.yaml":
            exec_prof_data = prof_data
            break

    community_plugins_map: dict[str, str] = {}
    if isinstance(exec_prof_data, dict):
        raw = exec_prof_data.get("c40_community_plugins") or {}
        if isinstance(raw, dict):
            community_plugins_map = raw

    validated_standard: list[tuple[str, dict]] = []
    deferred_community_standard: list[tuple[str, dict]] = []
    for basename, prof_data in standard_entries:
        plugin_name = _community_plugin_name_for_basename(basename, community_plugins_map)
        if plugin_name is not None:
            deferred_community_standard.append((plugin_name, prof_data))
            continue
        dest = ConfigPaths.effective_path(f"configs/{basename}", for_write=True)
        cls = _settings_class_for_basename(basename, settings_classes)
        validated_standard.append(
            (dest, _validated_profile_dict(cls, profile_name, prof_data, filepath=dest))
        )

    validated_community: list[tuple[Path, dict]] = []
    for entry_name, prof_data in community_entries:
        plugin_name = Path(entry_name).stem
        if plugin_name not in community_plugins_map:
            log.warning(
                "Skipping legacy community plugin '%s': not referenced in execution profile '%s'",
                plugin_name,
                profile_name,
            )
            continue
        stored = community_plugins_map[plugin_name]
        dest = _community_settings_dest(plugin_name)
        cls = load_plugin_settings_class(
            plugin_name, str(PluginPaths.resolve(plugin_name, stored))
        )
        validated_community.append(
            (
                dest,
                _validated_profile_dict(cls, profile_name, prof_data, filepath=str(dest)),
            )
        )

    for plugin_name, prof_data in deferred_community_standard:
        if plugin_name not in community_plugins_map:
            log.warning(
                "Skipping community plugin '%s': not referenced in execution profile '%s'",
                plugin_name,
                profile_name,
            )
            continue
        stored = community_plugins_map[plugin_name]
        dest = _community_settings_dest(plugin_name)
        cls = load_plugin_settings_class(
            plugin_name, str(PluginPaths.resolve(plugin_name, stored))
        )
        validated_community.append(
            (
                dest,
                _validated_profile_dict(cls, profile_name, prof_data, filepath=str(dest)),
            )
        )

    for dest, validated in validated_standard:
        _merge_profile_into_file(dest, profile_name, validated, overwrite=overwrite)

    for dest, validated in validated_community:
        _merge_profile_into_file(dest, profile_name, validated, overwrite=overwrite)

    log.info("Imported profile '%s' from %s", profile_name, src)
    return profile_name
