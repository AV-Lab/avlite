"""AVLite headless-mode extension (headless runner + config CLI)."""

from avlite.extensions.e50_headless_mode.e52_config_cli import register_config_parser, run_config_command
from avlite.extensions.e50_headless_mode.e51_headless import register_parser, run_headless

__all__ = [
    "register_parser",
    "run_headless",
    "register_config_parser",
    "run_config_command",
]
