"""CLI for validating and describing YAML settings profiles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from avlite.c10_perception.c19_settings import PerceptionSettings
from avlite.c20_planning.c29_settings import PlanningSettings
from avlite.c30_control.c39_settings import ControlSettings
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c60_apps.c69_settings import AppSettings
from avlite.c60_apps.c61_app_strategy import AppStrategy
from avlite.c60_apps.c65_setting_utils import (
    dev_mode_export_warning,
    export_profile,
    import_profile,
    list_profiles,
    profile_file_path,
    setting_section,
)
from avlite.c60_apps.c63_plugins import list_plugins, load_builtin_plugin_settings
from avlite.c60_apps.c64_settings_schema import (
    SettingsValidationError,
    describe_schema,
    schema_of,
    validate_profile,
)

_LAYER_REGISTRY: dict[str, Any] = {
    "perception": PerceptionSettings,
    "planning": PlanningSettings,
    "control": ControlSettings,
    "execution": ExecutionSettings,
    "apps": AppSettings,
}
_setting_parser: argparse.ArgumentParser | None = None


class SettingCliApp(AppStrategy):
    """``avlite setting-cli`` — validate, describe, and import/export profiles in the terminal."""

    cli_name = "setting-cli"
    help = "Validate or describe settings profiles (terminal)"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        configure_parser(parser)

    def run(self, args: argparse.Namespace, unknown: list[str]) -> int | None:
        return run_setting_command(args)


def _layers() -> dict[str, Any]:
    return _LAYER_REGISTRY


def _stack_settings() -> list[Any]:
    return list(_layers().values())


def _plugin_settings() -> list[Any]:
    classes = []
    for plugin in list_plugins():
        cls = load_builtin_plugin_settings(plugin)
        if cls is not None:
            classes.append(cls)
    return classes


def cmd_validate(args: argparse.Namespace) -> int:
    errors: list[str] = []
    settings_classes = _stack_settings() + _plugin_settings()
    profiles = [args.profile] if args.profile else list_profiles()

    for prof in profiles:
        filepath = Path(profile_file_path(prof, for_write=False))
        if not filepath.exists():
            errors.append(f"{filepath}: profile '{prof}' not found")
            continue
        try:
            with open(filepath) as f:
                config = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            errors.append(f"{filepath}: YAML syntax error: {exc}")
            continue

        for settings_cls in settings_classes:
            schema = schema_of(settings_cls)
            if schema is None:
                continue
            group, key = setting_section(settings_cls)
            if group == "plugins":
                section = (config.get("plugins") or {}).get(key)
            else:
                section = config.get(key)
            if not isinstance(section, dict):
                continue
            try:
                validate_profile(schema, section, filepath=str(filepath), profile=prof)
            except SettingsValidationError as exc:
                errors.append(str(exc))

    if errors:
        for line in errors:
            print(line, file=sys.stderr)
        return 1
    print("All profiles valid.")
    return 0


def cmd_describe(args: argparse.Namespace) -> int:
    layers = _layers()
    if args.layer:
        key = args.layer.lower()
        if key not in layers:
            print(f"Unknown layer '{args.layer}'. Choose from: {', '.join(sorted(layers))}", file=sys.stderr)
            return 1
        schema = schema_of(layers[key])
        for line in describe_schema(schema, field_filter=args.field):
            print(line)
        return 0

    if args.field:
        print(f"Specify --layer with --field; layers: {', '.join(sorted(layers))}", file=sys.stderr)
        return 1

    for name, settings_cls in sorted(layers.items()):
        schema = schema_of(settings_cls)
        if schema is None:
            continue
        print(f"[{name}]")
        for line in describe_schema(schema):
            print(line)
        print()
    return 0


def cmd_help(_args: argparse.Namespace) -> int:
    if _setting_parser is not None:
        _setting_parser.print_help()
    return 0


def cmd_export_profile(args: argparse.Namespace) -> int:
    output = args.output or f"{args.profile}.yaml"
    try:
        warning = dev_mode_export_warning(AppSettings.c62_community_plugins)
        if warning:
            print(warning, file=sys.stderr)
        count = export_profile(
            args.profile,
            output,
        include_stack=not args.no_stack,
            include_app=not args.no_app,
            include_plugins=not args.no_plugins,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Failed to write profile: {exc}", file=sys.stderr)
        return 1
    print(f"Exported profile '{args.profile}' ({count} section(s)) to {output}")
    return 0


def cmd_import_profile(args: argparse.Namespace) -> int:
    try:
        profile_name = import_profile(args.path, overwrite=args.force)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Failed to import profile: {exc}", file=sys.stderr)
        return 1
    print(f"Imported profile '{profile_name}'")
    return 0


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Add nested subcommands to the ``setting-cli`` app parser."""
    global _setting_parser
    _setting_parser = parser
    setting_sub = parser.add_subparsers(dest="setting_command")

    help_p = setting_sub.add_parser("help", help="Show setting-cli command usage")
    help_p.set_defaults(setting_handler=cmd_help)

    validate = setting_sub.add_parser("validate", help="Validate YAML profiles against schemas")
    validate.add_argument("--profile", help="Validate only this profile name")
    validate.set_defaults(setting_handler=cmd_validate)

    describe = setting_sub.add_parser("describe", help="Print field types, defaults, and descriptions")
    describe.add_argument("--layer", help="Stack layer: perception, planning, control, execution")
    describe.add_argument("--field", help="Single field name to describe (requires --layer)")
    describe.set_defaults(setting_handler=cmd_describe)

    export_p = setting_sub.add_parser("export-profile", help="Export a profile to a YAML file")
    export_p.add_argument("profile", help="Profile name to export")
    export_p.add_argument("-o", "--output", help="Output YAML path (default: {profile}.yaml)")
    export_p.add_argument("--no-stack", action="store_true", help="Exclude stack layer sections (c10–c40)")
    export_p.add_argument("--no-app", action="store_true", help="Exclude the c69_apps section")
    export_p.add_argument("--no-plugins", action="store_true", help="Exclude the plugins section")
    export_p.set_defaults(setting_handler=cmd_export_profile)

    import_p = setting_sub.add_parser("import-profile", help="Import a profile from a YAML file")
    import_p.add_argument("path", help="Path to profile YAML file")
    import_p.add_argument("--force", action="store_true", help="Overwrite an existing profile")
    import_p.set_defaults(setting_handler=cmd_import_profile)


def run_setting_command(args: argparse.Namespace) -> int:
    handler = getattr(args, "setting_handler", None)
    if handler is None:
        if _setting_parser is not None:
            _setting_parser.print_help()
        return 0
    return handler(args)
