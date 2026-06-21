import yaml
import logging
import os
from pathlib import Path
import types
import sys
import importlib
import importlib.util
import tkinter as tk

log = logging.getLogger(__name__)


def get_absolute_path(relative_path: str) -> str:
    """Convert a relative path to an absolute path based on the current file location."""
    if os.path.isabs(relative_path):
        return relative_path
    # current_file = os.path.realpath(__file__) if sys.argv[0] == "" else sys.argv[0]
    # project_dir = Path(current_file).parent.parent
    project_dir = Path(__file__).resolve().parent.parent.parent
    log.warning(f"Converting relative path {relative_path} to absolute path based on project dir {str(project_dir/ relative_path)}")
    return str(project_dir / relative_path)


def reload_lib(reload_extensions: bool = True, exclude_settings=False, exclude_stack=False) -> None:
    """Dynamically reload all modules in the project."""
    log.info("Reloading imports...")

    # Get the base package name (AVLite) and all submodules
    project_modules = []
    base_prefixes = ["avlite.c10_perception", "avlite.c20_planning", "avlite.c30_control", "avlite.c40_execution", "avlite.c50_visualization", "avlite.c60_common"]
    stack_settings = ["avlite.c10_perception.c19_settings", "avlite.c20_planning.c29_settings", "avlite.c30_control.c39_settings",
                             "avlite.c40_execution.c49_settings"]

    if exclude_stack:
        project_modules = stack_settings
        
        if reload_extensions:
            log.debug("Reloading extensions...")
            ext = list_extensions()
            project_modules += [f"extensions.{ext}.settings" for ext in ext]
    
    else:
        if reload_extensions:
            ext = ["avlite.extensions." + e for e in list_extensions()]
            project_modules.extend(ext)
        else:
            ext = []

        # Find all loaded modules that belong to our project
        for module_name in list(sys.modules.keys()):
            if any(module_name.startswith(prefix) for prefix in base_prefixes):
                project_modules.append(module_name)

            elif reload_extensions and module_name.startswith("avlite.extensions"):
                project_modules.append(module_name)


        # Sort modules to ensure proper reload order (parent modules before child modules)
        project_modules.sort(key=lambda x: x.count('.'))

        if exclude_settings:
            project_modules = [mod for mod in project_modules if mod not in stack_settings]


    #################################################
    ## Reloading Settings Modules ###################
    #################################################
    for module_name in project_modules:
        if module_name in sys.modules:
            try:
                module = sys.modules[module_name]
                importlib.reload(module)
                log.debug(f"Reloaded: {module_name}")
            except Exception as e:
                log.warning(f"Failed to reload {module_name}: {e}")


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


def load_plugin_settings_class(name: str, plugin_path: str):
    """Load ``PluginSettings`` from ``<plugin_path>/settings.py``, or return ``None``."""
    settings_file = Path(plugin_path) / "settings.py"
    if not settings_file.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(
            f"_avlite_plugin_{name}_settings", settings_file
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, "PluginSettings", None)
    except Exception as e:
        log.warning("Could not load PluginSettings for '%s': %s", name, e)
        return None


def patch_plugin_settings(cls, name: str, plugin_path: str) -> None:
    """Inject ``filepath`` and ``exclude`` onto *cls* so save/load_setting work."""
    config_dir = Path(plugin_path) / "config"
    cls.filepath = str(config_dir / f"{name}.yaml")
    if not hasattr(cls, "exclude"):
        cls.exclude = ["exclude", "filepath"]


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
                # Properly handle collections (including empty ones)
                if value is None and isinstance(attr_value, (list, dict)):
                    # Create an empty collection of the same type
                    setattr(setting, attr_name, type(attr_value)())
                else:
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

def rename_setting_profile(setting, old_profile, new_profile) -> bool:
    """Rename a profile in the configuration file."""
    filepath = setting.filepath
    if old_profile == "default":
        log.warning("Cannot rename the 'default' profile.")
        return False

    try:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f) or {}

        if old_profile not in config:
            log.warning(f"Profile '{old_profile}' does not exist in {filepath}")
            return False
        if new_profile in config:
            log.warning(f"Profile '{new_profile}' already exists in {filepath}")
            return False

        config[new_profile] = config.pop(old_profile)

        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        log.info(f"Profile '{old_profile}' renamed to '{new_profile}' in {filepath}")
        return True
    except Exception as e:
        log.error(f"Failed to rename profile: {e}")
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

def load_all_stack_settings(profile="default", load_extensions=True): 
    """Load all stack settings and extension settings."""
    from avlite.c10_perception.c19_settings import PerceptionSettings
    from avlite.c20_planning.c29_settings import PlanningSettings
    from avlite.c30_control.c39_settings import ControlSettings
    from avlite.c40_execution.c49_settings import ExecutionSettings
    load_setting(PerceptionSettings, profile=profile)
    load_setting(PlanningSettings, profile=profile)
    load_setting(ControlSettings, profile=profile)
    load_setting(ExecutionSettings, profile=profile)

    extensions = list_extensions()
    for ext in extensions:
        try:
            module = importlib.import_module(f"avlite.extensions.{ext}.settings")
            ExtensionSettings = getattr(module, "ExtensionSettings")
            load_setting(ExtensionSettings, profile=profile)
        except (ImportError, AttributeError) as e:
            log.error(f"Failed to load settings for extension {ext}: {e}")


def _ensure_extensions_package(extensions_directory: Path) -> None:
    """Ensure `avlite.extensions` exists as an importable package."""
    existing = sys.modules.get("avlite.extensions")
    if existing is not None:
        package_paths = getattr(existing, "__path__", None)
        if package_paths is None:
            existing.__path__ = [str(extensions_directory)]
        elif str(extensions_directory) not in package_paths:
            package_paths.append(str(extensions_directory))
        return

    extensions_init = extensions_directory / "__init__.py"
    if extensions_init.exists():
        spec = importlib.util.spec_from_file_location("avlite.extensions", extensions_init)
        if spec and spec.loader:
            ext_module = importlib.util.module_from_spec(spec)
            ext_module.__path__ = [str(extensions_directory)]
            sys.modules["avlite.extensions"] = ext_module
            spec.loader.exec_module(ext_module)
            return

    ext_module = types.ModuleType("avlite.extensions")
    ext_module.__path__ = [str(extensions_directory)]
    sys.modules["avlite.extensions"] = ext_module


def import_all_modules(directory:str = "", pkg_name="", extensions_filter: list[str] = None):
    """Import all Python modules from a directory.
    
    Args:
        directory: Path to the external extension directory.
        pkg_name: Package name for external extensions.
        extensions_filter: If provided, only load these extensions. If empty list, load nothing.
                          If None, load all discovered extensions.
    """

    if not directory:
        extensions_directory = Path(__file__).parent.parent / "extensions"
        if extensions_filter is not None:
            pkgs = extensions_filter
        else:
            pkgs = list_extensions()
        pkg_paths = [extensions_directory / pkg for pkg in pkgs]
    else:
        extensions_directory = Path(directory).parent # to get the parent directory
        if not extensions_directory.exists():
            log.error(f"Extensions directory does not exist: {extensions_directory}")
            return
        pkg_paths = [Path(directory)]
    
    _ensure_extensions_package(extensions_directory)
    
    for pkg_path in pkg_paths:
        if not pkg_path.exists():
            log.warning(f"Package path does not exist: {pkg_path}")
            continue
        package_prefix = "avlite.extensions." + (pkg_name if directory else pkg_path.name)
        log.info(f"Importing package: {package_prefix} from {pkg_path}")
        

        init_py_path = pkg_path / "__init__.py"
        if not init_py_path.exists():
            log.warning(f"No __init__.py found for {package_prefix}, creating empty module")
            # Create an empty module without requiring the file
            module = types.ModuleType(package_prefix)
            module.__path__ = [str(pkg_path)]
            sys.modules[package_prefix] = module
        else:
            spec = importlib.util.spec_from_file_location(
                package_prefix,
                init_py_path,
                submodule_search_locations=[str(pkg_path)],
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                module.__path__ = [str(pkg_path)]
                sys.modules[package_prefix] = module
                spec.loader.exec_module(module)
            else:
                log.error(f"Failed to create module spec for {package_prefix}")
        
        files = list(pkg_path.rglob('*.py'))

        for f in files:
            if f.name == '__init__.py':
                continue
            if "test" in f.parts:
                continue
                
            # Create module name from relative path
            relative_path = f.relative_to(pkg_path)
            module_name = package_prefix + "." + str(relative_path.with_suffix('')).replace('/', '.').replace('\\', '.')
            
            # Ensure all parent packages exist in sys.modules
            parts = module_name.split('.')
            for i in range(1, len(parts)):
                parent_name = '.'.join(parts[:i])
                if parent_name not in sys.modules:
                    parent_module = types.ModuleType(parent_name)
                    relative_parent_parts = parts[len(package_prefix.split('.')):i]
                    if relative_parent_parts:
                        parent_module.__path__ = [str(pkg_path.joinpath(*relative_parent_parts))]
                    else:
                        parent_module.__path__ = [str(pkg_path)]
                    sys.modules[parent_name] = parent_module
            
            try:
                if module_name in sys.modules:
                    log.debug(f"Skipping already loaded module: {module_name}")
                    continue
                spec = importlib.util.spec_from_file_location(module_name, f)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    log.debug(f"Loaded module: {module_name} from {f}")
            except Exception as e:
                log.error(f"Failed to load module {module_name} from {f}: {e}")#, stack_info=True)

