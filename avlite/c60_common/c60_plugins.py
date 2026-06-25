"""Built-in and community plugin discovery, loading, and log-routing helpers."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import re
import sys
import types
from pathlib import Path

from avlite.c60_common.c67_paths import (
    builtin_plugins_dir,
    community_plugin_settings_filepath,
    effective_config_path,
    legacy_community_plugin_settings_path,
    resolve_plugin_path,
)

log = logging.getLogger(__name__)

PLUGIN_NAMESPACE = "avlite.plugins"

_LAYER_DIGIT_TO_KEY = {
    "1": "perception",
    "2": "planning",
    "3": "control",
    "4": "execution",
    "5": "visualization",
    "6": "common",
}

_PNX_PREFIX = re.compile(r"^p(\d)", re.IGNORECASE)

_SKIP_PLUGIN_DIRS = frozenset({".venv", "venv", "__pycache__", "site-packages", "dist", "build", ".git"})


def plugin_module_prefix(name: str) -> str:
    """Full module prefix for built-in or community plugin *name*."""
    return f"{PLUGIN_NAMESPACE}.{name}"


def is_plugin_logger(record_name: str) -> bool:
    """True if logging record comes from a plugin module."""
    return record_name.startswith(PLUGIN_NAMESPACE + ".")


def plugin_package_from_logger(logger_name: str) -> str | None:
    """Return top-level plugin package segment from logging record name."""
    prefix = PLUGIN_NAMESPACE + "."
    if not logger_name.startswith(prefix):
        return None
    rest = logger_name[len(prefix) :]
    return rest.split(".", 1)[0] if rest else None


def plugin_module_from_logger(logger_name: str) -> str | None:
    """Return first module segment under the plugin package, e.g. p31_joystick_controller."""
    pkg = plugin_package_from_logger(logger_name)
    if pkg is None:
        return None
    prefix = f"{PLUGIN_NAMESPACE}.{pkg}."
    if not logger_name.startswith(prefix):
        return None
    rest = logger_name[len(prefix) :]
    return rest.split(".", 1)[0] if rest else None


def layer_key_for_plugin_package(package: str) -> str | None:
    """Map p10_foo / p40_bar → layer key, or None if not pNx-shaped."""
    m = _PNX_PREFIX.match(package)
    if not m:
        return None
    return _LAYER_DIGIT_TO_KEY.get(m.group(1)[0])


def layer_key_for_plugin_log_record(logger_name: str) -> str | None:
    """Layer for log routing: module pNx first, else package pNx, else None."""
    pkg = plugin_package_from_logger(logger_name)
    if pkg is None:
        return None
    module_seg = plugin_module_from_logger(logger_name)
    if module_seg is not None:
        layer = layer_key_for_plugin_package(module_seg)
        if layer is not None:
            return layer
    return layer_key_for_plugin_package(pkg)


def list_plugins() -> list[str]:
    """List built-in plugin package names under ``avlite/plugins/``."""
    plugins_dir = builtin_plugins_dir()
    plugins: list[str] = []
    if plugins_dir.is_dir():
        for plugin_dir in plugins_dir.iterdir():
            if plugin_dir.is_dir() and not plugin_dir.name.startswith("."):
                plugins.append(plugin_dir.name)
    else:
        log.warning("Plugins directory not found at: %s", plugins_dir)
    if not plugins:
        log.warning("No built-in plugins found.")
    return [x for x in plugins if x != "__pycache__"]


def load_builtin_plugin_settings(plugin: str):
    """Load ``PluginSettings`` from ``settings.py`` without importing the plugin package."""
    plugin_dir = builtin_plugins_dir() / plugin
    settings_file = plugin_dir / "settings.py"
    if not settings_file.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(
            f"_avlite_plugin_{plugin}_settings", settings_file
        )
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = getattr(module, "PluginSettings", None)
        if cls is not None:
            plugin_schema = getattr(module, "PluginSettingsSchema", None)
            if plugin_schema is not None and not hasattr(cls, "schema"):
                cls.schema = plugin_schema
        return cls
    except Exception as e:
        log.warning("Could not load PluginSettings for '%s': %s", plugin, e)
        return None


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
    cls.filepath = community_plugin_settings_filepath(name)
    if not hasattr(cls, "exclude"):
        cls.exclude = ["exclude", "filepath", "schema"]
    else:
        cls.exclude = list(cls.exclude)
        for key in ("filepath", "schema"):
            if key not in cls.exclude:
                cls.exclude.append(key)


def load_community_plugin_setting(
    name: str,
    stored: str,
    profile: str = "default",
    *,
    binder=None,
):
    """Load ``PluginSettings`` for a community plugin (user config, legacy install-dir fallback)."""
    from avlite.c60_common.c69_setting_utils import load_setting

    install_path = str(resolve_plugin_path(name, stored))
    cls = load_plugin_settings_class(name, install_path)
    if cls is None:
        return None
    patch_plugin_settings(cls, name, install_path)
    user_filepath = community_plugin_settings_filepath(name)
    user_path = Path(effective_config_path(user_filepath, for_write=False))
    if not user_path.is_file():
        legacy = legacy_community_plugin_settings_path(name, stored)
        if legacy.is_file():
            cls.filepath = str(legacy)
    load_setting(cls, profile=profile, binder=binder)
    if cls.filepath != user_filepath:
        patch_plugin_settings(cls, name, install_path)
    return cls


def load_all_stack_settings(profile: str = "default", load_plugins: bool = True) -> None:
    """Load all stack settings and built-in plugin settings."""
    from avlite.c10_perception.c19_settings import PerceptionSettings
    from avlite.c20_planning.c29_settings import PlanningSettings
    from avlite.c30_control.c39_settings import ControlSettings
    from avlite.c40_execution.c49_settings import ExecutionSettings
    from avlite.c60_common.c69_setting_utils import load_setting

    load_setting(PerceptionSettings, profile=profile)
    load_setting(PlanningSettings, profile=profile)
    load_setting(ControlSettings, profile=profile)
    load_setting(ExecutionSettings, profile=profile)

    from avlite.c60_common.c67_paths import bootstrap_reference_point_from_maps

    bootstrap_reference_point_from_maps()

    if not load_plugins:
        return

    for plugin in ExecutionSettings.c40_default_plugins:
        cls = load_builtin_plugin_settings(plugin)
        if cls is None:
            continue
        load_setting(cls, profile=profile)


def unregister_plugin_package(plugin_name: str) -> None:
    """Remove strategy classes registered by a plugin and purge its modules."""
    from avlite.c10_perception.c12_perception_strategy import PerceptionStrategy
    from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
    from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy
    from avlite.c30_control.c32_control_strategy import ControlStrategy
    from avlite.c40_execution.c41_world_bridge import WorldBridge
    from avlite.c40_execution.c42_executer import Executer

    prefix = plugin_module_prefix(plugin_name)
    for registry in (
        PerceptionStrategy.registry,
        GlobalPlannerStrategy.registry,
        LocalPlanningStrategy.registry,
        ControlStrategy.registry,
        Executer.registry,
        WorldBridge.registry,
    ):
        to_remove = [
            name
            for name, cls in registry.items()
            if cls.__module__.startswith(prefix)
        ]
        for name in to_remove:
            del registry[name]
            log.info("Unregistered %s from %s", name, prefix)

    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith(prefix):
            del sys.modules[mod_name]
            log.debug("Removed %s from sys.modules", mod_name)


def sync_builtin_plugins(allowed: list[str]) -> None:
    """Unload built-in plugins not in *allowed*, then import those that are."""
    allowed_set = set(allowed)
    for name in list_plugins():
        if name not in allowed_set:
            unregister_plugin_package(name)
    if allowed:
        import_plugin_modules(plugins_filter=allowed)


def _ensure_plugins_package(plugins_directory: Path) -> None:
    """Ensure ``avlite.plugins`` exists as an importable package."""
    existing = sys.modules.get(PLUGIN_NAMESPACE)
    if existing is not None:
        package_paths = getattr(existing, "__path__", None)
        if package_paths is None:
            existing.__path__ = [str(plugins_directory)]
        elif str(plugins_directory) not in package_paths:
            package_paths.append(str(plugins_directory))
        return

    plugins_init = plugins_directory / "__init__.py"
    if plugins_init.exists():
        spec = importlib.util.spec_from_file_location(PLUGIN_NAMESPACE, plugins_init)
        if spec and spec.loader:
            plugin_module = importlib.util.module_from_spec(spec)
            plugin_module.__path__ = [str(plugins_directory)]
            sys.modules[PLUGIN_NAMESPACE] = plugin_module
            spec.loader.exec_module(plugin_module)
            return

    plugin_module = types.ModuleType(PLUGIN_NAMESPACE)
    plugin_module.__path__ = [str(plugins_directory)]
    sys.modules[PLUGIN_NAMESPACE] = plugin_module


def import_plugin_modules(
    directory: str = "",
    pkg_name: str = "",
    plugins_filter: list[str] | None = None,
) -> None:
    """Import all Python modules from a built-in or community plugin directory."""
    if not directory:
        plugins_directory = builtin_plugins_dir()
        if plugins_filter is not None:
            pkgs = plugins_filter
        else:
            pkgs = list_plugins()
        pkg_paths = [plugins_directory / pkg for pkg in pkgs]
    else:
        plugins_directory = Path(directory).parent
        if not plugins_directory.exists():
            log.error("Plugins directory does not exist: %s", plugins_directory)
            return
        pkg_paths = [Path(directory)]

    _ensure_plugins_package(plugins_directory)

    for pkg_path in pkg_paths:
        if not pkg_path.exists():
            log.warning("Package path does not exist: %s", pkg_path)
            continue
        package_prefix = plugin_module_prefix(pkg_name if directory else pkg_path.name)
        log.info("Importing package: %s from %s", package_prefix, pkg_path)

        init_py_path = pkg_path / "__init__.py"
        if not init_py_path.exists():
            log.warning("No __init__.py found for %s, creating empty module", package_prefix)
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
                log.error("Failed to create module spec for %s", package_prefix)

        files = list(pkg_path.rglob("*.py"))

        for f in files:
            if f.name == "__init__.py":
                continue
            relative_path = f.relative_to(pkg_path)
            if any(part in _SKIP_PLUGIN_DIRS for part in relative_path.parts):
                continue
            if any(part in ("test", "tests") for part in relative_path.parts):
                continue

            module_name = (
                package_prefix
                + "."
                + str(relative_path.with_suffix("")).replace("/", ".").replace("\\", ".")
            )

            parts = module_name.split(".")
            for i in range(1, len(parts)):
                parent_name = ".".join(parts[:i])
                if parent_name not in sys.modules:
                    parent_module = types.ModuleType(parent_name)
                    relative_parent_parts = parts[len(package_prefix.split(".")) : i]
                    if relative_parent_parts:
                        parent_module.__path__ = [str(pkg_path.joinpath(*relative_parent_parts))]
                    else:
                        parent_module.__path__ = [str(pkg_path)]
                    sys.modules[parent_name] = parent_module

            try:
                if module_name in sys.modules:
                    log.debug("Skipping already loaded module: %s", module_name)
                    continue
                spec = importlib.util.spec_from_file_location(module_name, f)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    log.debug("Loaded module: %s from %s", module_name, f)
            except Exception as e:
                log.error("Failed to load module %s from %s: %s", module_name, f, e)


def reload_lib(
    reload_plugins: bool = True,
    exclude_settings: bool = False,
    exclude_stack: bool = False,
) -> None:
    """Dynamically reload stack and plugin modules."""
    log.info("Reloading imports...")

    project_modules: list[str] = []
    base_prefixes = [
        "avlite.c10_perception",
        "avlite.c20_planning",
        "avlite.c30_control",
        "avlite.c40_execution",
        "avlite.c50_visualization",
        "avlite.c60_common",
    ]
    stack_settings = [
        "avlite.c10_perception.c19_settings",
        "avlite.c20_planning.c29_settings",
        "avlite.c30_control.c39_settings",
        "avlite.c40_execution.c49_settings",
    ]

    if exclude_stack:
        project_modules = stack_settings
        if reload_plugins:
            log.debug("Reloading plugins...")
            project_modules += [f"plugins.{p}.settings" for p in list_plugins()]
    else:
        if reload_plugins:
            plugins = [plugin_module_prefix(p) for p in list_plugins()]
            project_modules.extend(plugins)
        else:
            plugins = []

        for module_name in list(sys.modules.keys()):
            if any(module_name.startswith(prefix) for prefix in base_prefixes):
                project_modules.append(module_name)
            elif reload_plugins and is_plugin_logger(module_name):
                project_modules.append(module_name)

        project_modules.sort(key=lambda x: x.count("."))

        if exclude_settings:
            project_modules = [mod for mod in project_modules if mod not in stack_settings]

    for module_name in project_modules:
        if module_name in sys.modules:
            try:
                module = sys.modules[module_name]
                importlib.reload(module)
                log.debug("Reloaded: %s", module_name)
            except Exception as e:
                log.warning("Failed to reload %s: %s", module_name, e)
