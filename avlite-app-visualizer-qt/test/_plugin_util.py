from __future__ import annotations

from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
PLUGIN_NAME = "avlite-app-visualizer-qt"


def ensure_qt_plugin_loaded() -> None:
    from avlite.c60_apps.c61_app_strategy import bootstrap_apps
    from avlite.c60_apps.c63_plugins import import_plugin_modules

    bootstrap_apps()
    import_plugin_modules(str(PLUGIN_DIR), pkg_name=PLUGIN_NAME)
