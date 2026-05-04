
import argparse
import logging
import sys

log = logging.getLogger(__name__)


def _setup_dpi() -> None:
    import platform
    import os

    if platform.system() == "Linux":
        os.environ["TK_WINDOWS_FORCE_OPENGL"] = "1"
    else:
        import ctypes

        try:  # >= win 8.
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):  # win 8.0 or less
            ctypes.windll.user32.SetProcessDPIAware()
        os.environ["TK_WINDOWS_FORCE_OPENGL"] = "1"


def _run_visualizer() -> None:
    from avlite.c50_visualization.c51_visualizer_app import VisualizerApp

    _setup_dpi()
    app = VisualizerApp()
    app.mainloop()


def _run_plugins() -> None:
    from avlite.c50_visualization.c50_community_plugins_app import main as plugins_main

    _setup_dpi()
    plugins_main()


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the AVLite application."""
    parser = argparse.ArgumentParser(prog="avlite", description="AVLite")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("plugins", help="Open the community plugins manager")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if args.command == "plugins":
        _run_plugins()
    else:
        _run_visualizer()


if __name__ == "__main__":
    main()
