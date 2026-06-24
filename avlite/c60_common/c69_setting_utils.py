import yaml
import logging
import os
import shutil
from pathlib import Path
import types
import sys
import importlib
import importlib.util
import tkinter as tk

from avlite.c60_common.c68_settings_schema import (
    SETTINGS_META,
    SettingsValidationError,
    apply_validated_to_setting,
    dump_from_setting,
    validate_profile,
)

log = logging.getLogger(__name__)


def get_config_dir() -> Path:
    """User AVLite config root (~/.config/avlite by default)."""
    env = os.environ.get("AVLITE_CONFIG_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return (base / "avlite").resolve()


def _user_meta_dir() -> Path:
    """Preferences always under ~/.config/avlite (ignores ``AVLITE_CONFIG_DIR``)."""
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return (base / "avlite").resolve()


def bundled_config_dir() -> Path:
    """Shipped YAML directory (git clone ``configs/``); missing after plain pip install."""
    return Path(__file__).resolve().parent.parent.parent / "configs"


def get_repo_config_dir() -> Path:
    """Repository shipped defaults (…/configs)."""
    return bundled_config_dir()


def can_edit_repo_configs() -> bool:
    d = bundled_config_dir()
    return d.is_dir() and any(d.glob("*.yaml"))


def is_repo_config_target() -> bool:
    path = _user_meta_dir() / "config_target"
    if not path.is_file():
        return False
    return path.read_text(encoding="utf-8").strip() == "repo"


def set_repo_config_target(enabled: bool) -> None:
    _user_meta_dir().mkdir(parents=True, exist_ok=True)
    value = "repo" if enabled else "user"
    (_user_meta_dir() / "config_target").write_text(value + "\n", encoding="utf-8")


def copy_repo_configs_to_user() -> list[str]:
    """Copy bundled ``configs/*.yaml`` into ``get_config_dir()``; overwrite existing."""
    src_dir = bundled_config_dir()
    if not can_edit_repo_configs():
        raise FileNotFoundError(f"No bundled configs in {src_dir}")
    dest_dir = get_config_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for src in sorted(src_dir.glob("*.yaml")):
        dest = dest_dir / src.name
        shutil.copy2(src, dest)
        copied.append(str(dest))
    return copied


def get_user_configs_dir() -> Path:
    """User YAML profiles directory (same as ``get_config_dir()``)."""
    return get_config_dir()


def effective_config_path(stored_filepath: str, *, for_write: bool = False) -> str:
    """Resolve a settings filepath: user config dir for writes; user if present else repo for reads."""
    path = Path(stored_filepath)
    if path.is_absolute():
        if for_write:
            path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
    name = path.name
    user = get_config_dir() / name
    repo = get_repo_config_dir() / name
    if is_repo_config_target() and can_edit_repo_configs():
        bundled = get_repo_config_dir() / name
        if for_write:
            bundled.parent.mkdir(parents=True, exist_ok=True)
        return str(bundled)
    if for_write:
        get_config_dir().mkdir(parents=True, exist_ok=True)
        return str(user)
    return str(user if user.is_file() else repo)


def clear_user_configs() -> list[str]:
    """Remove user-local stack YAML files so loads fall back to repository defaults."""
    repo = get_repo_config_dir()
    names = {p.name for p in repo.glob("*.yaml")} if repo.is_dir() else set()
    deleted: list[str] = []
    config_dir = get_config_dir()
    for name in sorted(names):
        path = config_dir / name
        if path.is_file():
            path.unlink()
            deleted.append(str(path))
    nested = config_dir / "configs"
    if nested.is_dir() and not any(nested.iterdir()):
        nested.rmdir()
    startup = config_dir / "startup_profile"
    if startup.is_file():
        startup.unlink()
        deleted.append(str(startup))
    return deleted


def get_startup_profile() -> str | None:
    """Last GUI profile name to load on startup (``~/.config/avlite/startup_profile``)."""
    path = get_config_dir() / "startup_profile"
    if not path.is_file():
        return None
    name = path.read_text(encoding="utf-8").strip()
    return name or None


def set_startup_profile(profile: str) -> None:
    """Remember ``profile`` as the GUI startup profile for the next session."""
    get_config_dir().mkdir(parents=True, exist_ok=True)
    (get_config_dir() / "startup_profile").write_text(profile.strip() + "\n", encoding="utf-8")


def resolve_plugin_path(name: str, stored: str) -> Path:
    """Resolve a community plugin install path (name-only, relative, or legacy absolute)."""
    from avlite.c50_visualization.c50_community_plugins_app import get_plugins_dir

    if not stored or stored == name:
        return get_plugins_dir() / name
    path = Path(stored)
    if path.is_absolute():
        return path
    return get_plugins_dir() / stored


def get_data_dir() -> Path:
    """User data directory (~/.local/share/avlite/data by default)."""
    env = os.environ.get("AVLITE_DATA_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return (base / "avlite" / "data").resolve()


def get_absolute_path(relative_path: str, *, for_write: bool = False) -> str:
    """Resolve a data path: user dir first on read, repo ``data/`` fallback; writes go to user dir."""
    if os.path.isabs(relative_path):
        if for_write:
            Path(relative_path).parent.mkdir(parents=True, exist_ok=True)
        return relative_path

    repo_root = Path(__file__).resolve().parent.parent.parent
    repo_path = repo_root / relative_path

    rel = relative_path
    if rel.startswith("data/"):
        rel = rel[5:]
    elif rel == "data":
        rel = ""

    if for_write:
        user_path = get_data_dir() / rel
        user_path.parent.mkdir(parents=True, exist_ok=True)
        return str(user_path)

    user_path = get_data_dir() / rel
    if user_path.is_file():
        return str(user_path)
    return str(repo_path)


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
        cls = getattr(module, "PluginSettings", None)
        if cls is not None:
            plugin_schema = getattr(module, "PluginSettingsSchema", None)
            if plugin_schema is not None:
                cls.schema = plugin_schema
        return cls
    except Exception as e:
        log.warning("Could not load PluginSettings for '%s': %s", name, e)
        return None


def patch_plugin_settings(cls, name: str, plugin_path: str) -> None:
    """Inject ``filepath`` and ``exclude`` onto *cls* so save/load_setting work."""
    config_dir = Path(plugin_path) / "config"
    cls.filepath = str(config_dir / f"{name}.yaml")
    if not hasattr(cls, "exclude"):
        cls.exclude = ["exclude", "filepath", "schema"]
    else:
        cls.exclude = list(cls.exclude)
        for key in ("filepath", "schema"):
            if key not in cls.exclude:
                cls.exclude.append(key)


def _setting_exclude(setting) -> set[str]:
    exclude = set(getattr(setting, "exclude", []))
    exclude.update(SETTINGS_META)
    return exclude


def _get_schema(setting):
    if isinstance(setting, type):
        return getattr(setting, "schema", None)
    return getattr(type(setting), "schema", None)


def save_setting(setting, profile="default") -> None:
    """Save current configuration to a YAML file."""
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
            setting, schema, filepath=filepath, profile=profile
        )
    else:
        config[profile] = {}
        exclude = _setting_exclude(setting)
        target = setting if not isinstance(setting, type) else setting
        for attr_name, attr_value in vars(target).items():
            if callable(attr_value) or attr_name.startswith("_") or attr_name in exclude:
                continue
            if isinstance(attr_value, tk.Variable):
                config[profile][attr_name] = attr_value.get()
            else:
                config[profile][attr_name] = attr_value

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    log.info(f"Configuration saved to {filepath} for profile '{profile}'")


def load_setting(setting, profile="default", *, strict: bool = False) -> bool:
    """Load configuration from a YAML file. Returns True on success."""
    stored = setting.filepath if not isinstance(setting, type) else setting.filepath
    filepath = effective_config_path(stored, for_write=False)
    schema = _get_schema(setting)
    try:
        with open(filepath, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        log.error(f"YAML syntax error in {filepath}: {e}")
        return False
    except OSError as e:
        log.error(f"Failed to read configuration {filepath}: {e}")
        return False

    try:
        if not config:
            log.warning(f"Empty or invalid configuration file: {filepath}")
            return False

        profile_dict = config.get(profile, "")
        if not profile_dict:
            log.warning(f"Profile '{profile}' not found in {filepath}")
            return False

        if schema is not None:
            known = set(schema.model_fields.keys())
            unknown = set(profile_dict.keys()) - known - SETTINGS_META
            for key in sorted(unknown):
                log.debug(f"Skipping unknown key in {filepath} profile '{profile}': {key}")
            try:
                validated = validate_profile(
                    schema, profile_dict, filepath=filepath, profile=profile
                )
            except SettingsValidationError as e:
                log.error(str(e))
                if strict:
                    raise
                return False
            apply_validated_to_setting(setting, validated)
        else:
            exclude = _setting_exclude(setting)
            for attr_name, value in profile_dict.items():
                if attr_name in exclude:
                    continue
                if not hasattr(setting, attr_name):
                    log.warning(f"Skipping unknown attribute: {attr_name}")
                    continue
                attr_value = getattr(setting, attr_name)
                if isinstance(attr_value, tk.Variable):
                    if isinstance(attr_value, tk.BooleanVar):
                        attr_value.set(bool(value))
                    else:
                        attr_value.set(value)
                elif not callable(attr_value):
                    if value is None and isinstance(attr_value, (list, dict)):
                        setattr(setting, attr_name, type(attr_value)())
                    else:
                        setattr(setting, attr_name, value)

        log.info(f"Configuration loaded from {filepath} for profile '{profile}'")
        return True
    except SettingsValidationError:
        raise
    except Exception as e:
        log.error(f"Failed to load configuration: {e}")
        return False

def delete_setting_profile(setting, profile) -> bool:
    """Delete a profile from the configuration file."""
    filepath = effective_config_path(setting.filepath, for_write=True)
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
    filepath = effective_config_path(setting.filepath, for_write=True)
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
    filepath = effective_config_path(setting.filepath, for_write=False)
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

def load_extension_settings_class(ext: str):
    """Load ``ExtensionSettings`` from ``settings.py`` without importing the extension package."""
    ext_dir = Path(__file__).resolve().parent.parent / "extensions" / ext
    settings_file = ext_dir / "settings.py"
    if not settings_file.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(
            f"_avlite_ext_{ext}_settings", settings_file
        )
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = getattr(module, "ExtensionSettings", None)
        if cls is not None:
            plugin_schema = getattr(module, "ExtensionSettingsSchema", None)
            if plugin_schema is not None and not hasattr(cls, "schema"):
                cls.schema = plugin_schema
        return cls
    except Exception as e:
        log.warning("Could not load ExtensionSettings for '%s': %s", ext, e)
        return None


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

    if not load_extensions:
        return

    for ext in list_extensions():
        cls = load_extension_settings_class(ext)
        if cls is None:
            continue
        load_setting(cls, profile=profile)


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

