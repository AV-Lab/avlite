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

Built-in ``p60_*`` packages and community plugins from the active profile are
imported at startup via :func:`bootstrap_apps`, before subcommands are parsed.

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


def bootstrap_apps() -> None:
    """Import built-in ``p60_*`` apps and profile community plugins so they register."""
    from avlite.c60_apps.c63_plugins import import_plugin_modules, list_plugins
    from avlite.c60_apps.c68_paths import PluginPaths
    from avlite.c60_apps.c69_settings import AppSettings

    for name in list_plugins():
        if not name.startswith("p60"):
            continue
        try:
            import_plugin_modules(plugins_filter=[name])
        except Exception as e:
            log.warning("Could not load app plugin '%s': %s", name, e)

    # Profile is already loaded. Import listed community plugins so their
    # AppStrategy commands exist before register_parsers(). One failure must
    # not take down the CLI, and a community app must not replace a built-in name.
    if not AppSettings.c62_load_plugins:
        return
    builtin = dict(AppStrategy.registry)
    for name, stored in AppSettings.c62_community_plugins.items():
        path = PluginPaths.load_path(name, stored)
        if path is None:
            log.warning("Could not load community app plugin '%s': not installed", name)
            continue
        try:
            import_plugin_modules(str(path), pkg_name=name)
        except Exception as e:
            log.warning("Could not load community app plugin '%s': %s", name, e)
        for cli_name, cls in builtin.items():
            if AppStrategy.registry.get(cli_name) is cls:
                continue
            AppStrategy.registry[cli_name] = cls
            log.warning(
                "Community plugin '%s' skipped command %r; built-in already registered",
                name,
                cli_name,
            )


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
