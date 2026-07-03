import argparse
import sys

from avlite.c50_apps.c51_app_strategy import bootstrap_apps, register_parsers, run_app


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the AVLite application."""
    parser = argparse.ArgumentParser(prog="avlite", description="AVLite")
    sub = parser.add_subparsers(dest="command")

    bootstrap_apps()
    register_parsers(sub)

    try:
        args, unknown = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as exc:
        if exc.code not in (0, None):
            sys.stderr.write("\nError parsing arguments. Use --help for usage.\n")
        raise

    if unknown:
        sys.stderr.write(f"Ignoring unknown arguments: {unknown}\n")

    exit_code = run_app(args.command, args, unknown)
    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
