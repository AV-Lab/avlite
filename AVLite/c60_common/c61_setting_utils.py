import yaml
import logging
import os
from pathlib import Path

import sys
import logging
import importlib
import importlib.util

import tkinter as tk

log = logging.getLogger(__name__)

EXTENSIONS_DIR = Path(__file__).parent.parent / "extensions"

def get_absolute_path(relative_path: str) -> str:
    """Convert a relative path to an absolute path based on the current file location."""
    if os.path.isabs(relative_path):
        return relative_path
    current_file = os.path.realpath(__file__) if sys.argv[0] == "" else sys.argv[0]
    project_dir = Path(current_file).parent.parent
    return str(project_dir / relative_path)


def reload_lib(reload_extensions: bool = True, exclude_settings=False, exclude_stack=False) -> None:
    """Dynamically reload all modules in the project."""
    log.info("Reloading imports...")

    # Get the base package name (AVLite) and all submodules
    project_modules = []
    base_prefixes = ["c10_perception", "c20_planning", "c30_control", "c40_execution", "c50_visualization", "c60_utils"]
    extension_base_prefixes = ["e10_perception", "e20_planning", "e30_control", "e40_execution"]
    stack_settings = ["c10_perception.c19_settings", "c20_planning.c29_settings", "c30_control.c39_settings",
                             "c40_execution.c49_settings"]


    # base_prefixes = [p.replace("\\", ".").replace("/", ".") for p in base_prefixes]
    if exclude_stack:
        project_modules = stack_settings
        
        if reload_extensions:
            log.debug("Reloading extensions...")
            ext = list_extensions()
            project_modules += [f"extensions.{ext}.settings" for ext in ext]
    
    else:
        # Find all loaded modules that belong to our project
        for module_name in list(sys.modules.keys()):
            if any(module_name.startswith(prefix) for prefix in base_prefixes):
                project_modules.append(module_name)

            # ext = list_extensions()
            # ext.extend([x+"."+p for p in extension_base_prefixes for x in ext])
            # print(ext)
            #
            # # import_all_modules()
            # if reload_extensions and any(module_name.startswith(name) for name in ext):
            #     # Check if the module is an extension settings module
            #     project_modules.append(module_name)


        # Sort modules to ensure proper reload order (parent modules before child modules)
        project_modules.sort(key=lambda x: x.count('.'))

        if exclude_settings:
            project_modules = [mod for mod in project_modules if mod not in stack_settings]


    #################################################
    ## Reloading Settings Modules ###################
    #################################################

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


def list_extensions() -> list:
    """List all available extensions in the extensions directory."""
    avlite_dir = Path(__file__).parent.parent  # Go up to AVLite directory
    extensions_dir = avlite_dir / "extensions"
    extensions = []

    if extensions_dir.exists() and extensions_dir.is_dir():
        for ext_dir in extensions_dir.iterdir():
            if ext_dir.is_dir() and not ext_dir.name.startswith('.'):
                extensions.append(ext_dir.name)
    else:
        log.warning(f"Extensions directory not found at: {extensions_dir}")

    if not extensions:
        log.warning("No extensions found in the specified directories.")

    extensions = [x for x in extensions if x != "__pycache__"]
    return extensions


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

def delete_setting_profile(setting, profile) -> bool:
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

def import_all_modules(directory:str = ""):
    """Import all Python modules from a directory."""

    log.debug(f"Importing all modules from directory: {directory}")

    if not directory:
        ext_directory = EXTENSIONS_DIR
    else: 
        ext_directory = Path(directory)
        # sys.path.insert(0, str(directory))
        sys.path.append(directory)
    
    for py_file in ext_directory.rglob('*.py'):
        if py_file.name == '__init__.py':
            continue
            
        # Create module name from relative path
        relative_path = py_file.relative_to(ext_directory)
        module_name = str(relative_path.with_suffix('')).replace('/', '.').replace('\\', '.')
        
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sys.modules[module_name] = module
                log.debug(f"Loaded module: {module_name} from {py_file}")
        except Exception as e:
            log.error(f"Failed to load module {module_name} from {py_file}: {e}")
    if module_name not in sys.modules:
        log.warning(f"Module {module_name} is NOT loaded.")


def load_all_stack_settings(profile="default", load_extensions=True, all_extension_dirs=[]): 
    from c10_perception.c19_settings import PerceptionSettings
    from c20_planning.c29_settings import PlanningSettings
    from c30_control.c39_settings import ControlSettings
    from c40_execution.c49_settings import ExecutionSettings
    load_setting(PerceptionSettings, profile=profile)
    load_setting(PlanningSettings, profile=profile)
    load_setting(ControlSettings, profile=profile)
    load_setting(ExecutionSettings, profile=profile)

    log.debug(f"all extension_dirs: {all_extension_dirs}")
    extensions = list_extensions()
    for ext in extensions:
        module = importlib.import_module(f"extensions.{ext}.settings")
        ExtensionSettings = getattr(module, "ExtensionSettings")
        load_setting(ExtensionSettings, profile=profile)



