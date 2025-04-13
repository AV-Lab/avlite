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

class Config:
    pass

def load_config(config_path, source_run=True):
    if os.path.isabs(config_path):
        raise ValueError("config_path should be relative to the project directory")

    # pkg_config = pkg_resources.resource_filename("AVLite", "config.yaml")
    if source_run:
        current_file_name = os.path.realpath(__file__) if sys.argv[0] == "" else sys.argv[0]
    else:
        current_file_name = pkg_config if sys.argv[0] == "" else sys.argv[0]

    log.info(f"current_file_name: {current_file_name}")
    if source_run:
        project_dir = Path(current_file_name).parent.parent
    else:
        project_dir = Path(current_file_name).parent.parent / "share/AVLite"

    config_file = project_dir / config_path

    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)
        path_to_track = project_dir / config_data["global_trajectory"]

    # loading race trajectory data
    with open(path_to_track, "r") as f:
        track_data = json.load(f)
        reference_path = [point[:2] for point in track_data["ReferenceLine"]]  # ignoring z
        reference_velocity = track_data["ReferenceSpeed"]
        ref_left_boundary_d = track_data["LeftBound"]
        ref_right_boundary_d = track_data["RightBound"]
    logging.info(f"Track data loaded from {path_to_track}")
    return reference_path, reference_velocity, ref_left_boundary_d, ref_right_boundary_d, config_data


def reload_lib():
    """Dynamically reload all modules in the project."""
    log.info("Reloading imports...")

    # Get the base package name (AVLite) and all submodules
    project_modules = []
    base_prefixes = ["c10_perception", "c20_planning", "c30_control", "c40_execution", "c50_visualization", "c60_utils",]

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



def save_config(setting) -> None:
    """Save current visualization configuration to a YAML file.

    Args:
        data: The VisualizerData instance
        filepath: Path where to save the configuration
    """
    filepath=setting.filepath
    config = {}

    # Extract all attributes from the data
    for attr_name, attr_value in vars(setting).items():
        if callable(attr_value) or attr_name.startswith('_') or attr_name in setting.exclude:
            continue

        # Handle tkinter variables
        if isinstance(attr_value, tk.Variable):
            # Extract the actual value
            config[attr_name] = attr_value.get()
        # Handle regular Python values (non-tkinter)
        else:
            config[attr_name] = attr_value

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save to YAML file
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    log.info(f"Visualization configuration saved to {filepath}")


def load_setting(setting) -> None:
    """Load visualization configuration from a YAML file.

    Args:
        data: The VisualizerData instance
        filepath: Path from where to load the configuration
    """
    filepath=setting.filepath
    try:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)

        if not config:
            log.warning(f"Empty or invalid configuration file: {filepath}")
            return

        # Load all settings
        for attr_name, value in config.items():
            # Check if attribute exists in the data class
            if not hasattr(setting, attr_name):  # Changed from app.data to data
                log.warning(f"Skipping unknown attribute: {attr_name}")
                continue

            attr_value = getattr(setting, attr_name)  # Changed from app.data to data

            # Handle tkinter variables
            if isinstance(attr_value, tk.Variable):
                if isinstance(attr_value, tk.BooleanVar):
                    attr_value.set(bool(value))
                else:
                    attr_value.set(value)
            # Handle regular Python values
            elif not callable(attr_value):
                setattr(setting, attr_name, value)
        
        log.info(f"Configuration loaded from {filepath}")
    except Exception as e:
        log.error(f"Failed to load configuration: {e}")

