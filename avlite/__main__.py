
import argparse
import sys


def _run_visualizer() -> None:
    from avlite.c50_visualization.c51_visualizer_app import VisualizerApp

    app = VisualizerApp()
    app.mainloop()


def _run_plugins() -> None:
    from avlite.c50_visualization.c54_plugins import main as plugins_main

    plugins_main()


def _run_headless(args: argparse.Namespace, unknown: list[str]) -> None:
    try:
        from avlite.plugins.p50_headless_mode import run_config_command, run_headless
    except ImportError as exc:
        sys.stderr.write(
            "Headless mode plugin is not available (missing optional dependencies).\n"
            "Install with:  pip install rich\n"
            f"Detail: {exc}\n"
        )
        sys.exit(1)

    if args.command == "config":
        sys.exit(run_config_command(args))

    if unknown:
        sys.stderr.write(f"Ignoring unknown arguments: {unknown}\n")
    run_headless(
        profile=args.profile or args.profile_pos or "default",
        control_dt=args.control_dt,
        replan_dt=args.replan_dt,
        perceive=args.perceive,
        log_level=args.log_level,
    )


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the AVLite application."""
    parser = argparse.ArgumentParser(prog="avlite", description="AVLite")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("plugins", help="Open the community plugins manager")
    try:
        from avlite.plugins.p50_headless_mode import register_config_parser, register_parser

        register_config_parser(sub)
        register_parser(sub)
    except ImportError:
        sub.add_parser("headless", help="Run headless (requires optional deps: rich)")
        sub.add_parser("config", help="Manage profiles (requires headless plugin)")

    try:
        args, unknown = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as exc:
        if exc.code not in (0, None):
            sys.stderr.write("\nError parsing arguments. Use --help for usage.\n")
        raise

    if args.command == "plugins":
        _run_plugins()
    elif args.command in ("config", "headless"):
        _run_headless(args, unknown)
    else:
        _run_visualizer()


if __name__ == "__main__":
    main()
