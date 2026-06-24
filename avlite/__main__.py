
import argparse
import sys


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
    from avlite.plugins.p50_headless_mode import (
        register_parser,
        run_headless,
        register_config_parser,
        run_config_command,
    )

    parser = argparse.ArgumentParser(prog="avlite", description="AVLite")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("plugins", help="Open the community plugins manager")
    register_config_parser(sub)
    register_parser(sub)

    try:
        args, unknown = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as exc:
        if exc.code not in (0, None):
            sys.stderr.write("\nError parsing arguments. Use --help for usage.\n")
        raise

    if args.command == "plugins":
        _run_plugins()
    elif args.command == "config":
        sys.exit(run_config_command(args))
    elif args.command == "headless":
        if unknown:
            sys.stderr.write(f"Ignoring unknown arguments: {unknown}\n")
        run_headless(
            profile=args.profile or args.profile_pos or "default",
            control_dt=args.control_dt,
            replan_dt=args.replan_dt,
            perceive=args.perceive,
            log_level=args.log_level,
        )
    else:
        _run_visualizer()


if __name__ == "__main__":
    main()
