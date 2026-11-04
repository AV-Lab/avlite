"""Base class and registry for AVLite apps (CLI/GUI entry points).

An *app* is anything launchable as ``avlite <cli_name>``: the visualizer
(default), the plugin manager, the settings GUI, the headless runner, etc.

To add a new app, subclass :class:`AppStrategy` and implement ``run``::

    from avlite.c60_apps.c61_app_strategy import AppStrategy

    class MyToolApp(AppStrategy):
        cli_name = "my-tool"
        help = "Short description for avlite --help"

        def run(self, args, unknown):
            ...
            return 0   # optional; None = exit 0

Importing the module auto-registers the app (same pattern as
``PerceptionStrategy``). Override :meth:`AppStrategy.configure_parser` only
when the app needs CLI flags or nested subcommands.

Built-in ``p60_*`` plugin packages are imported at startup via :func:`~avlite.c60_apps.c61_app_strategy.import_app_plugins`.

This module stays free of heavy imports (no tkinter); plugins may import it
without pulling in the GUI.
"""

from __future__ import annotations

import argparse
import logging
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)


class AppStrategy(ABC):
    """Base class for AVLite apps. Subclass, set ``cli_name``/``help``, implement ``run``."""

    registry: dict[str | None, type["AppStrategy"]] = {}
    cli_name: str | None = None  # None = default app (runs when no subcommand is given)
    help: str = ""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Optional. Add arguments to this app's subcommand."""

    @abstractmethod
    def run(self, args: argparse.Namespace, unknown: list[str]) -> int | None:
        """Run the app. Return exit code, or None for 0."""

    def __init_subclass__(cls, abstract: bool = False, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if not abstract:
            AppStrategy.registry[cls.cli_name] = cls


# ---------------------------------------------------------------------------
# Internal wiring used by avlite/__main__.py (not part of the plugin API)
# ---------------------------------------------------------------------------


_BUILTIN_APPS = {"headless", "setting", "setting-cli", "plugins"}


def _apps_file():
    from pathlib import Path

    return Path(__file__).resolve().parent.parent / "config" / "apps.yaml"


def bootstrap_apps() -> None:
    """Import all app modules so their ``AppStrategy`` subclasses register."""
    import importlib
    from pathlib import Path

    import yaml

    from avlite.c60_apps.c63_plugins import import_plugin_modules, list_plugins

    for name in list_plugins():
        if not name.startswith("p60"):
            continue
        try:
            import_plugin_modules(plugins_filter=[name])
        except Exception as e:
            log.warning("Could not load app plugin '%s': %s", name, e)
    try:
        apps = yaml.safe_load(_apps_file().read_text(encoding="utf-8"))
        if not isinstance(apps, dict):
            return
    except Exception:
        return
    protected = dict(AppStrategy.registry)
    for name, target in apps.items():
        if name in protected:
            if name not in _BUILTIN_APPS:
                log.warning("Skipping app '%s': command already registered", name)
            continue
        try:
            plugin = Path(str(target)).expanduser()
            if plugin.is_dir():
                import_plugin_modules(str(plugin))
            else:
                importlib.import_module(str(target))
        except Exception as e:
            log.warning("Could not load app '%s': %s", name, e)
    for name, cls in protected.items():
        current = AppStrategy.registry.get(name)
        if current is not cls:
            log.warning("Skipping app '%s': command '%s' already registered", getattr(current, "__name__", name), name)
            AppStrategy.registry[name] = cls


def note_app(plugin_path) -> None:
    """Append an installed AppStrategy plugin to ``avlite/config/apps.yaml``."""
    import re
    import yaml
    from avlite.c60_apps.c68_paths import PluginPaths

    try:
        root = PluginPaths.plugin_load_dir(plugin_path)
        text = "".join(
            chunk for py in root.glob("*.py")
            if "AppStrategy" in (chunk := py.read_text(encoding="utf-8", errors="ignore"))
        )
        path = _apps_file()
        current = path.read_text(encoding="utf-8") if path.is_file() else ""
        extra = ""
        for cli in dict.fromkeys(re.findall(r"(?m)^\s*cli_name\s*=\s*['\"]([^'\"]+)['\"]", text)):
            if cli in _BUILTIN_APPS or re.search(rf"(?m)^{re.escape(cli)}\s*:", current + extra):
                continue
            extra += yaml.safe_dump({cli: str(root)}, sort_keys=False)
        if not extra:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text((current if not current or current.endswith("\n") else current + "\n") + extra, encoding="utf-8")
    except Exception as e:
        log.warning("Could not register app '%s': %s", plugin_path, e)


def forget_app(name: str) -> None:
    """Drop one installed plugin from ``avlite/config/apps.yaml``."""
    from pathlib import Path
    import yaml

    try:
        path = _apps_file()
        text = path.read_text(encoding="utf-8")
        apps = yaml.safe_load(text) or {}
        if not isinstance(apps, dict):
            return
        drop = {k for k, v in apps.items() if k not in _BUILTIN_APPS and name in Path(str(v)).parts}
        if not drop:
            return
        path.write_text(
            "".join(ln for ln in text.splitlines(True) if ln.split(":", 1)[0].strip().strip("'\"") not in drop),
            encoding="utf-8",
        )
    except Exception as e:
        log.warning("Could not unregister app '%s': %s", name, e)


def register_parsers(sub) -> None:
    """Add one subcommand per registered app (skips the default app)."""
    for cls in AppStrategy.registry.values():
        if cls.cli_name is None:
            continue
        parser = sub.add_parser(cls.cli_name, help=cls.help)
        cls().configure_parser(parser)


def run_app(command: str | None, args: argparse.Namespace, unknown: list[str]) -> int | None:
    """Dispatch to the app registered for *command* (None = default app)."""
    cls = AppStrategy.registry.get(command) or AppStrategy.registry.get(None)
    if cls is None:
        raise SystemExit(f"No app registered for command {command!r}.")
    return cls().run(args, unknown)
