"""YAML profile load/save for AVLite settings classes."""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol

import yaml

from avlite.c60_common.c68_settings_schema import (
    SETTINGS_META,
    PlainBinder,
    SettingsValidationError,
    apply_validated_to_setting,
    dump_from_setting,
    validate_profile,
)
from avlite.c60_common.c67_paths import effective_config_path

log = logging.getLogger(__name__)


class SettingsBinder(Protocol):
    def get_value(self, setting: Any, attr_name: str) -> Any: ...
    def set_value(self, setting: Any, attr_name: str, value: Any) -> None: ...


def _setting_exclude(setting) -> set[str]:
    exclude = set(getattr(setting, "exclude", []))
    exclude.update(SETTINGS_META)
    return exclude


def _get_schema(setting):
    if isinstance(setting, type):
        return getattr(setting, "schema", None)
    return getattr(type(setting), "schema", None)


def save_setting(
    setting,
    profile: str = "default",
    *,
    binder: SettingsBinder | None = None,
) -> None:
    """Save current configuration to a YAML file."""
    bind = binder or PlainBinder()
    stored = setting.filepath if not isinstance(setting, type) else setting.filepath
    filepath = effective_config_path(stored, for_write=True)
    schema = _get_schema(setting)

    read_path = effective_config_path(stored, for_write=False)
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
    stored = setting.filepath if not isinstance(setting, type) else setting.filepath
    filepath = effective_config_path(stored, for_write=False)
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
    filepath = effective_config_path(setting.filepath, for_write=True)
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
    filepath = effective_config_path(setting.filepath, for_write=True)
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
    filepath = effective_config_path(setting.filepath, for_write=False)
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
