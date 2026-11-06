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


class AppsApp(AppStrategy):
    """``avlite apps`` — list every registered app."""

    cli_name = "apps"
    help = "List registered apps"

    def run(self, args: argparse.Namespace, unknown: list[str]) -> int | None:
        rows = [
            ("default" if name is None else name, cls.help)
            for name, cls in AppStrategy.registry.items()
        ]
        width = max(len(label) for label, _help in rows)
        for label, help_text in sorted(rows):
            print(f"{label:<{width}}  {help_text}")
        return 0


# ---------------------------------------------------------------------------
# Internal wiring used by avlite/__main__.py (not part of the plugin API)
# ---------------------------------------------------------------------------


def bootstrap_apps() -> None:
    """Import built-in ``p60`` apps, then ``configs/apps.yaml``.

    A missing or unreadable file keeps the built-in commands. A plugin cannot
    replace a command that is already registered.
    """
    import importlib
    from pathlib import Path

    import yaml

    from avlite.c60_apps.c63_plugins import import_plugin_modules, list_plugins
    from avlite.c60_apps.c68_paths import ConfigPaths

    for name in list_plugins():
        if not name.startswith("p60"):
            continue
        try:
            import_plugin_modules(plugins_filter=[name])
        except Exception as e:
            log.warning("Could not load app plugin '%s': %s", name, e)

    apps_file = Path(ConfigPaths.effective_path("configs/apps.yaml"))
    bundled = ConfigPaths.bundled_dir() / "apps.yaml"
    if not apps_file.is_file() and bundled.is_file() and bundled.resolve() != apps_file.resolve():
        apps_file.parent.mkdir(parents=True, exist_ok=True)
        apps_file.write_text(bundled.read_text(encoding="utf-8"), encoding="utf-8")
    try:
        listed = yaml.safe_load(apps_file.read_text(encoding="utf-8"))
        if not isinstance(listed, dict):
            return
    except Exception:
        return

    protected = dict(AppStrategy.registry)
    for name, target in listed.items():
        if name in protected:
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
            log.warning(
                "Skipping app '%s': command '%s' already registered",
                getattr(current, "__name__", name),
                name,
            )
            AppStrategy.registry[name] = cls


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
