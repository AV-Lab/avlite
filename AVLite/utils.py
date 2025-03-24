import yaml
import logging
import json
import os
from pathlib import Path

import sys
import pkg_resources
import logging
import importlib

from c50_visualize.c59_data import VisualizerData


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
    base_prefixes = ["c10_perceive", "c20_plan", "c30_control", "c40_execute", "c50_visualize"]
    
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



def save_visualizer_config(data:VisualizerData, filepath: str="configs/c50_visualize.yaml") -> None:
    """Save current visualization configuration to a YAML file.
    
    Args:
        data: The VisualizerData instance
        filepath: Path where to save the configuration
    """
    config = {}
    
    # Extract all attributes from the data
    for attr_name, attr_value in vars(data).items():
        # Skip methods, private attributes, and non-serializable objects
        if callable(attr_value) or attr_name.startswith('_'):
            continue
            
        # Handle tkinter variables
        if isinstance(attr_value, tk.Variable):
            # Extract the actual value
            config[attr_name] = attr_value.get()
        # Handle regular Python values (non-tkinter)
        elif not isinstance(attr_value, tk.Tk):  # Fixed: using tk.Tk instead of tk._tkinter.Tk
            config[attr_name] = attr_value
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save to YAML file
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    log.info(f"Visualization configuration saved to {filepath}")

def load_visualizer_config(data:VisualizerData, filepath: str="configs/c50_visualize.yaml") -> None:
    """Load visualization configuration from a YAML file.
    
    Args:
        data: The VisualizerData instance
        filepath: Path from where to load the configuration
    """
    try:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        
        if not config:
            log.warning(f"Empty or invalid configuration file: {filepath}")
            return
            
        # Load all settings
        for attr_name, value in config.items():
            # Check if attribute exists in the data class
            if not hasattr(data, attr_name):  # Changed from app.data to data
                log.warning(f"Skipping unknown attribute: {attr_name}")
                continue
                
            attr_value = getattr(data, attr_name)  # Changed from app.data to data
            
            # Handle tkinter variables
            if isinstance(attr_value, tk.Variable):
                if isinstance(attr_value, tk.BooleanVar):
                    attr_value.set(bool(value))
                else:
                    attr_value.set(value)
            # Handle regular Python values
            elif not callable(attr_value):
                setattr(data, attr_name, value)
        
        log.info(f"Visualization configuration loaded from {filepath}")
    except Exception as e:
        log.error(f"Failed to load visualization configuration: {e}")

def joy_stick_test():
    import pygame
    import time

    pygame.init()
    pygame.joystick.init()

    # Check for joystick
    if pygame.joystick.get_count() == 0:
        log.warning("No joystick connected")
        return

    # Initialize the first joystick
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    log.info(f"Joystick name: {joystick.get_name()}")
    log.info(f"Number of axes: {joystick.get_numaxes()}")

    try:
        while True:
            # Pump the event loop
            pygame.event.pump()

            # Read axis values
            for i in range(joystick.get_numaxes()):
                axis_value = joystick.get_axis(i)
                print(f"Axis {i} value: {axis_value:.3f}")

            # Add a small delay to make the output readable
            time.sleep(0.1)

    except KeyboardInterrupt:
        log.info("Exiting...")

    finally:
        pygame.quit()

if __name__ == "__main__":
    main()
