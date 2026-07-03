"""Built-in and community plugin discovery, loading, and log-routing helpers."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import re
import sys
import types
from pathlib import Path

from avlite.c50_apps.c58_paths import (
    ConfigPaths,
    PluginPaths,
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
_AVLITE_PLUGIN_PREFIX = re.compile(r"^avlite[-_](controller|bridge|executer)", re.IGNORECASE)

_SKIP_PLUGIN_DIRS = frozenset({".venv", "venv", "__pycache__", "site-packages", "dist", "build", ".git"})


def plugin_import_name(name: str) -> str:
    """Python-safe plugin package segment (repo names may use dashes)."""
    return name.replace("-", "_")


def plugin_module_prefix(name: str) -> str:
    """Full module prefix for built-in or community plugin *name*."""
    return f"{PLUGIN_NAMESPACE}.{plugin_import_name(name)}"


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
    """Map p10_foo / p40_bar / avlite-bridge-* → layer key, or None if unknown."""
    m = _PNX_PREFIX.match(package)
    if m:
        return _LAYER_DIGIT_TO_KEY.get(m.group(1)[0])
    av = _AVLITE_PLUGIN_PREFIX.match(package)
    if av:
        kind = av.group(1).lower()
        if kind == "controller":
            return "control"
        if kind in ("bridge", "executer"):
            return "execution"
    return None


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
    plugins_dir = PluginPaths.builtin_dir()
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
    plugin_dir = PluginPaths.builtin_dir() / plugin
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
            _PluginSettingsIO.set_settings_filepath(cls, PluginPaths.settings_filepath(plugin))
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
        return cls
    except Exception as e:
        log.warning("Could not load PluginSettings for '%s': %s", name, e)
        return None


def patch_plugin_settings(setting, name: str, plugin_path: str) -> None:
    """Derive and set the settings ``filepath`` so save/load_setting work.

    *setting* may be a class singleton or a settings instance; the path is derived
    from the plugin name.
    """
    _PluginSettingsIO.set_settings_filepath(setting, PluginPaths.settings_filepath(name))


def load_community_plugin_setting(
    name: str,
    stored: str,
    profile: str = "default",
    *,
    binder=None,
):
    """Load ``PluginSettings`` for a community plugin from user config."""
    from avlite.c50_apps.c55_setting_utils import load_setting

    install_path = str(PluginPaths.resolve(name, stored))
    if not Path(install_path).is_dir():
        return None
    settings_mod_name = f"{plugin_module_prefix(name)}.settings"
    if settings_mod_name not in sys.modules:
        import_plugin_modules(install_path, pkg_name=name)

    cls = _PluginSettingsIO.plugin_settings_from_module(name)
    if cls is None:
        cls = load_plugin_settings_class(name, install_path)
    if cls is None:
        return None
    patch_plugin_settings(cls, name, install_path)
    load_setting(cls, profile=profile, binder=binder)
    return cls


def unregister_plugin_package(plugin_name: str) -> None:
    """Remove strategy classes registered by a plugin and purge its modules."""
    from avlite.c10_perception.c12_perception_strategy import PerceptionStrategy
    from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
    from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy
    from avlite.c30_control.c32_control_strategy import ControlStrategy
    from avlite.c40_execution.c41_world_bridge import WorldBridge
    from avlite.c40_execution.c42_execution_strategy import ExecutionStrategy

    registries = [
        PerceptionStrategy.registry,
        GlobalPlannerStrategy.registry,
        LocalPlanningStrategy.registry,
        ControlStrategy.registry,
        ExecutionStrategy.registry,
        WorldBridge.registry,
    ]
    # App registry, only if the app layer was loaded (c60 must not import c50_apps).
    app_strategy_mod = sys.modules.get("avlite.c50_apps.c51_app_strategy")
    if app_strategy_mod is not None:
        registries.append(app_strategy_mod.AppStrategy.registry)

    prefix = plugin_module_prefix(plugin_name)
    for registry in registries:
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


def sync_community_plugins(allowed: dict[str, str]) -> None:
    """Unload community plugins not in *allowed*, then import those that are."""
    allowed_import = {plugin_import_name(n) for n in allowed}
    builtin_import = {plugin_import_name(n) for n in list_plugins()}
    for import_name in _CommunityPluginPaths.loaded_plugin_import_names() - builtin_import - allowed_import:
        unregister_plugin_package(import_name)
    for name, stored in allowed.items():
        path = PluginPaths.resolve(name, stored)
        if not path.is_dir():
            continue
        import_plugin_modules(str(path), pkg_name=name)


def find_community_plugin_dir(import_seg: str) -> Path | None:
    """Map a Python import segment back to a community plugin install directory."""
    for root in (
        PluginPaths.install_dir(),
        PluginPaths.community_dev_dir(),
        PluginPaths.private_dev_dir(),
    ):
        found = _CommunityPluginPaths.match_community_plugin_dir(import_seg, root)
        if found is not None:
            return found
    for name, stored in _CommunityPluginPaths.community_plugins_from_execution_yaml().items():
        if plugin_import_name(name) == import_seg:
            path = PluginPaths.resolve(name, stored)
            if path.is_dir():
                return path.resolve()
    return None


class CommunityPluginFinder:
    """Resolve ``avlite.plugins.<import_name>`` to dashed community plugin directories."""

    def find_spec(self, fullname: str, path, target=None):
        prefix = PLUGIN_NAMESPACE + "."
        if not fullname.startswith(prefix):
            return None
        rest = fullname[len(prefix) :]
        if not rest:
            return None

        import_seg = rest.split(".", 1)[0]
        builtin = PluginPaths.builtin_dir() / import_seg
        if builtin.is_dir():
            return None

        plugin_dir = find_community_plugin_dir(import_seg)
        if plugin_dir is None:
            return None

        subparts = rest.split(".")[1:]
        if not subparts:
            init_py = plugin_dir / "__init__.py"
            return importlib.util.spec_from_file_location(
                fullname,
                str(init_py) if init_py.is_file() else None,
                submodule_search_locations=[str(plugin_dir)],
            )

        mod_path = plugin_dir.joinpath(*subparts).with_suffix(".py")
        if not mod_path.is_file():
            return None
        return importlib.util.spec_from_file_location(fullname, mod_path)


_community_import_hook_registered = False


def register_community_plugin_import_hook() -> None:
    """Register ``CommunityPluginFinder`` so worker subprocesses can import community plugins."""
    global _community_import_hook_registered
    if _community_import_hook_registered:
        return
    sys.meta_path.insert(0, CommunityPluginFinder())
    _community_import_hook_registered = True


def import_plugin_modules(
    directory: str = "",
    pkg_name: str = "",
    plugins_filter: list[str] | None = None,
) -> None:
    """Import all Python modules from a built-in or community plugin directory."""
    if not directory:
        plugins_directory = PluginPaths.builtin_dir()
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

    _CommunityPluginPaths.ensure_plugins_package(plugins_directory)

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

        display_name = pkg_name if directory else pkg_path.name
        settings_mod = sys.modules.get(f"{package_prefix}.settings")
        ps = getattr(settings_mod, "PluginSettings", None) if settings_mod else None
        if ps is not None:
            patch_plugin_settings(ps, display_name, str(pkg_path))


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
        "avlite.c60_common",
    ]
    c50_modules = [
        "avlite.c50_apps.c51_app_strategy",
        "avlite.c50_apps.c52_factory",
        "avlite.c50_apps.c53_plugins",
        "avlite.c50_apps.c54_settings_schema",
        "avlite.c50_apps.c55_setting_utils",
        "avlite.c50_apps.c58_paths",
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
            elif any(module_name.startswith(prefix) for prefix in c50_modules):
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
                log.error("Failed to reload %s: %s", module_name, e)


class _PluginSettingsIO:
    """Plugin settings filepath and module lookup (module-private)."""

    @staticmethod
    def set_settings_filepath(setting, filepath: str) -> None:
        """Assign ``filepath`` on a plugin settings class or singleton instance."""
        if isinstance(setting, type):
            setting.filepath = filepath
        else:
            type(setting).filepath = filepath

    @staticmethod
    def plugin_settings_from_module(name: str):
        """Return ``PluginSettings`` from an already-imported plugin settings module."""
        mod = sys.modules.get(f"{plugin_module_prefix(name)}.settings")
        return getattr(mod, "PluginSettings", None) if mod else None


class _CommunityPluginPaths:
    """Community plugin path resolution and package bootstrap (module-private)."""

    @staticmethod
    def loaded_plugin_import_names() -> set[str]:
        """Package segments currently loaded under ``avlite.plugins``."""
        names: set[str] = set()
        prefix = f"{PLUGIN_NAMESPACE}."
        for mod_name in sys.modules:
            if not mod_name.startswith(prefix):
                continue
            tail = mod_name[len(prefix):]
            if tail:
                names.add(tail.split(".", 1)[0])
        return names

    @staticmethod
    def community_plugins_from_execution_yaml() -> dict[str, str]:
        """Read ``c50_community_plugins`` from the active app profile YAML."""
        import yaml

        profile = ConfigPaths.startup_profile() or "default"
        for config_path in (
            ConfigPaths.user_dir() / "c59_apps.yaml",
            ConfigPaths.bundled_dir() / "c59_apps.yaml",
        ):
            if not config_path.is_file():
                continue
            try:
                with open(config_path, encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            except OSError:
                continue
            prof = data.get(profile, {})
            if not isinstance(prof, dict):
                continue
            raw = prof.get("c50_community_plugins") or {}
            if isinstance(raw, dict):
                return raw
        return {}

    @staticmethod
    def match_community_plugin_dir(import_seg: str, parent: Path) -> Path | None:
        """Return a plugin directory whose import segment matches *import_seg*."""
        if not parent.is_dir():
            return None
        for entry in parent.iterdir():
            if (
                entry.is_dir()
                and not entry.name.startswith(".")
                and plugin_import_name(entry.name) == import_seg
            ):
                return entry.resolve()
        return None

    @staticmethod
    def ensure_plugins_package(plugins_directory: Path) -> None:
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
