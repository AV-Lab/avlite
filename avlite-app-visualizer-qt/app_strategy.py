"""AppStrategy entry point: ``avlite qt``."""

from __future__ import annotations

import logging
import os

from avlite.c60_apps.c61_app_strategy import AppStrategy

log = logging.getLogger(__name__)


class VisualizationQtApp(AppStrategy):
    """High-performance Qt5 / pyqtgraph visualizer (alternative to the Tk app)."""

    cli_name = "qt"
    help = "Real-time Qt5 visualizer (pyqtgraph; BasicSim and other bridges)"

    def configure_parser(self, parser) -> None:
        parser.add_argument(
            "--offscreen",
            action="store_true",
            help="Force QT_QPA_PLATFORM=offscreen (CI / headless).",
        )
        parser.add_argument(
            "--profile",
            default=None,
            help="Startup profile name (default: remembered / 'default').",
        )

    def run(self, args, unknown) -> int | None:
        offscreen = bool(getattr(args, "offscreen", False)) or os.environ.get("QT_QPA_PLATFORM") == "offscreen"
        if getattr(args, "profile", None):
            from avlite.c60_apps.c68_paths import ConfigPaths

            ConfigPaths.set_startup_profile(args.profile)

        try:
            from .main_window import launch
        except ImportError as exc:
            log.error("%s", exc)
            print(exc)
            return 1
        return launch(offscreen=offscreen)
