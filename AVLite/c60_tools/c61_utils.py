import yaml
import logging
import json
import os
from pathlib import Path

import sys
import pkg_resources
import logging
import importlib


import tkinter as tk

log = logging.getLogger(__name__)


# def load_config(config_path):
#     if os.path.isabs(config_path):
#         raise ValueError("config_path should be relative to the project directory")
#
#     config_file = get_absolute_path(config_path)
#
#     with open(config_file, "r") as f:
#         config_data = yaml.safe_load(f)
#
#     return config_data

def get_absolute_path(relative_path: str) -> str:
    """Convert a relative path to an absolute path based on the current file location."""
    if os.path.isabs(relative_path):
        return relative_path
    current_file = os.path.realpath(__file__) if sys.argv[0] == "" else sys.argv[0]
    project_dir = Path(current_file).parent.parent
    return str(project_dir / relative_path)



def reload_lib():
    """Dynamically reload all modules in the project."""
    log.info("Reloading imports...")

    # Get the base package name (AVLite) and all submodules
    project_modules = []
    base_prefixes = ["c10_perception", "c20_planning", "c30_control", "c40_execution", "c50_visualization", "c60_tools",]

    # Find all loaded modules that belong to our project
    for module_name in list(sys.modules.keys()):
        if any(module_name.startswith(prefix) for prefix in base_prefixes):
            project_modules.append(module_name)

    # Sort modules to ensure proper reload order (parent modules before child modules)
    project_modules.sort(key=lambda x: x.count('.'))

    # Reload each module
    for module_name in project_modules:
        if module_name in sys.modules:
            try:
                module = sys.modules[module_name]
                importlib.reload(module)
                log.debug(f"Reloaded: {module_name}")
            except Exception as e:
                log.warning(f"Failed to reload {module_name}: {e}")

    log.info(f"Reloaded {len(project_modules)} modules")



def save_setting(setting, profile="default") -> None:
    """Save current visualization configuration to a YAML file. """
    filepath=setting.filepath
    # first load the current configuration if it exists
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f) or {}
        config[profile] = {}
    else:
        config = {profile: {}}

    # Extract all attributes from the data
    for attr_name, attr_value in vars(setting).items():
        if callable(attr_value) or attr_name.startswith('_') or attr_name in setting.exclude:
            continue

        # Handle tkinter variables
        if isinstance(attr_value, tk.Variable):
            # Extract the actual value
            config[profile][attr_name] = attr_value.get()
        # Handle regular Python values (non-tkinter)
        else:
            config[profile][attr_name] = attr_value

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save to YAML file
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    log.info(f"Visualization configuration saved to {filepath} for profile '{profile}'")


def load_setting(setting, profile="default") -> None:
    """Load visualization configuration from a YAML file. """
    filepath=setting.filepath
    try:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)

        if not config:
            log.warning(f"Empty or invalid configuration file: {filepath}")
            return

        profile_dict = config.get(profile,"")
        if not profile_dict:
            log.warning(f"Profile '{profile}' not found in {filepath}")
            return

        for attr_name, value in profile_dict.items():
            if not hasattr(setting, attr_name):  # Changed from app.data to data
                log.warning(f"Skipping unknown attribute: {attr_name}")
                continue

            attr_value = getattr(setting, attr_name)  

            # Handle tkinter variables
            if isinstance(attr_value, tk.Variable):
                if isinstance(attr_value, tk.BooleanVar):
                    attr_value.set(bool(value))
                else:
                    attr_value.set(value)
            # Handle regular Python values
            elif not callable(attr_value):
                setattr(setting, attr_name, value)
        
        log.info(f"Configuration loaded from {filepath} for profile '{profile}'")
    except Exception as e:
        log.error(f"Failed to load configuration: {e}")

def delete_profile(setting, profile) -> bool:
    """Delete a profile from the configuration file."""
    filepath = setting.filepath
    if profile == "default":
        log.warning("Cannot delete the 'default' profile.")
        return False

    try:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f) or {}

        if profile not in config:
            log.warning(f"Profile '{profile}' does not exist in {filepath}")
            return False

        del config[profile]

        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        log.info(f"Profile '{profile}' deleted from {filepath}")
        return True
    except Exception as e:
        log.error(f"Failed to delete profile: {e}")
        return False

def list_profiles(setting) -> list:
    """List all profiles in the configuration file."""
    filepath = setting.filepath
    try:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        if not config:
            log.warning(f"Empty or invalid configuration file: {filepath}")
            return []

        profiles = list(config.keys())
        log.info(f"Available profiles: {profiles}")
        return profiles
    except Exception as e:
        log.error(f"Failed to list profiles: {e}")
        return []
