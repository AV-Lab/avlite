"""Built-in shim that registers the Qt5 visualizer AppStrategy at CLI bootstrap.

The full plugin lives in the repo-root community package
``avlite-app-visualizer-qt/`` (same layout as ``avlite-bridge-carla``) so it can
be split into its own repository later. This ``p60_*`` package exists only so
``bootstrap_apps()`` discovers ``avlite qt`` without requiring the community
plugin to already be listed in a profile.

Qt / pyqtgraph are imported lazily by the community package when the app runs.
If the package or its deps are missing, a stub still registers the subcommand
with a clear install error.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

_PLUGIN_NAME = "avlite-app-visualizer-qt"


def _candidate_dirs() -> list[Path]:
    from avlite.c60_apps.c68_paths import PluginPaths

    here = Path(__file__).resolve()
    # avlite/plugins/p60_visualizer_qt -> repo root sibling
    monorepo = here.parents[3] / _PLUGIN_NAME
    return [
        monorepo,
        PluginPaths.install_dir() / _PLUGIN_NAME,
        PluginPaths.community_dev_dir() / _PLUGIN_NAME,
    ]


def _load_community_package() -> bool:
    from avlite.c60_apps.c63_plugins import import_plugin_modules

    for path in _candidate_dirs():
        if (path / "__init__.py").is_file():
            try:
                import_plugin_modules(str(path), pkg_name=_PLUGIN_NAME)
                log.info("Loaded Qt visualizer plugin from %s", path)
                return True
            except Exception as exc:
                log.warning("Failed loading Qt visualizer from %s: %s", path, exc)
    return False


def _register_stub() -> None:
    from avlite.c60_apps.c61_app_strategy import AppStrategy

    class VisualizationQtAppStub(AppStrategy):
        cli_name = "qt"
        help = "Real-time Qt5 visualizer (install avlite-app-visualizer-qt + PyQt5)"

        def run(self, args, unknown):
            print(
                "Qt visualizer plugin not available.\n"
                f"  Expected package at one of: {[str(p) for p in _candidate_dirs()]}\n"
                "  pip install -r avlite-app-visualizer-qt/requirements.txt"
            )
            return 1

    # Ensure registry entry even if a previous stub exists.
    AppStrategy.registry["qt"] = VisualizationQtAppStub


if not _load_community_package():
    _register_stub()

__all__ = []
